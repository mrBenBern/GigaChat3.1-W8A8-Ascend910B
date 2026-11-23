
import torch, torch_npu, functools, traceback
print = functools.partial(print, flush=True)
torch.npu.set_device(0)
torch.manual_seed(0)
H, Dq, Dv = 32, 192, 192
scale = Dq ** -0.5
B, L = 2, 1040
T = B * L
q = torch.randn(T, H, Dq, dtype=torch.bfloat16, device='npu')
k = torch.randn(T, H, Dq, dtype=torch.bfloat16, device='npu')
v = torch.randn(T, H, Dv, dtype=torch.bfloat16, device='npu')
ends = [L, 2 * L]

m2048 = torch.triu(torch.ones(2048, 2048, device='npu'), 1).to(torch.int8)
row = torch.arange(T, device='npu').unsqueeze(1)
col = torch.arange(T, device='npu').unsqueeze(0)
req = torch.repeat_interleave(torch.arange(B, device='npu'), torch.full((B,), L, device='npu', dtype=torch.long))
starts = torch.tensor([0, L], device='npu', dtype=torch.long)
rloc = row - starts[req].unsqueeze(1)
cloc = col - starts[req].unsqueeze(0)
mask_bd_bool = (req.unsqueeze(1) != req.unsqueeze(0)) | (cloc > rloc)
mask_bd_i8 = mask_bd_bool.to(torch.int8)

def try_call(tag, *args, **kw):
    try:
        res = torch_npu.npu_fused_infer_attention_score(*args, **kw)
        out = res[0]
        print('%s -> OK out %s' % (tag, list(out.shape)), flush=True)
        return res[0]
    except Exception as e:
        print('%s -> FAIL: %s' % (tag, str(e)[:600]), flush=True)
        return None

oA = try_call('A_bool_bd_sparse0', q, k, v, num_heads=H, num_key_value_heads=H, input_layout='TND',
              actual_seq_lengths=ends, actual_seq_lengths_kv=ends, scale=scale,
              sparse_mode=0, atten_mask=mask_bd_bool, softmax_lse_flag=True)
oB = try_call('B_i8_2048_sparse3', q, k, v, num_heads=H, num_key_value_heads=H, input_layout='TND',
              actual_seq_lengths=ends, actual_seq_lengths_kv=ends, scale=scale,
              sparse_mode=3, atten_mask=m2048, softmax_lse_flag=True)
oC = try_call('C_i8_bd_sparse0', q, k, v, num_heads=H, num_key_value_heads=H, input_layout='TND',
              actual_seq_lengths=ends, actual_seq_lengths_kv=ends, scale=scale,
              sparse_mode=0, atten_mask=mask_bd_i8, softmax_lse_flag=True)

def ref_req(r, q, k, v, L):
    s = q[r*L:(r+1)*L].float() @ k[r*L:(r+1)*L].float().transpose(0, 1) * scale
    s = s.masked_fill(torch.triu(torch.ones(L, L, device='npu'), 1).bool().unsqueeze(0).unsqueeze(0), float('-inf'))
    p = torch.softmax(s, dim=-1)
    o = torch.einsum('hts,shd->thd', p, v[r*L:(r+1)*L].float())
    return o

# isolation test: TND no-mask (context pattern) with B=2 distinct contents
k2 = k.clone(); v2 = v.clone()
req0_out_nofill = None
resE = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=H, num_key_value_heads=H,
    input_layout='TND', actual_seq_lengths=ends, actual_seq_lengths_kv=ends, scale=scale,
    sparse_mode=0, atten_mask=None, softmax_lse_flag=True)
oE = resE[0]
iso0 = (oE[:L].float() - ref_req(0, q, k, v, L)).abs().max().item()
iso1 = (oE[L:].float() - ref_req(1, q, k, v, L)).abs().max().item()
print('ISOLATION TND-nomask B=2: req0 diff=%.4f req1 diff=%.4f' % (iso0, iso1), flush=True)

# per-request BSND loop reference (D)
outs = []
for r in range(B):
    qr = q[r*L:(r+1)*L].unsqueeze(0); kr = k[r*L:(r+1)*L].unsqueeze(0); vr = v[r*L:(r+1)*L].unsqueeze(0)
    am = torch.triu(torch.ones(L, L, device='npu', dtype=torch.bool), 1).unsqueeze(0).unsqueeze(0)
    rd = torch_npu.npu_fused_infer_attention_score(qr, kr, vr, num_heads=H, num_key_value_heads=H,
        scale=scale, input_layout='BSND', atten_mask=am, sparse_mode=0, softmax_lse_flag=True)
    outs.append(rd[0].squeeze(0))
oD = torch.cat(outs, dim=0)
print('D_loop -> OK out %s' % list(oD.shape), flush=True)

for tag, o in [('B_vs_loop', oB), ('C_vs_loop', oC), ('E_nofill_vs_loop_first', None)]:
    if o is not None:
        d = (o.float() - oD.float()).abs().max().item()
        print('CHECK %s diff=%.4f' % (tag, d), flush=True)
print('PROBE5H2_DONE', flush=True)


import torch, torch_npu, functools
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

def ref_req(r, q, k, v, L):
    s = torch.einsum('thd,shd->hts', q[r*L:(r+1)*L].float(), k[r*L:(r+1)*L].float()) * scale
    s = s.masked_fill(torch.triu(torch.ones(L, L, device='npu'), 1).bool().unsqueeze(0), float('-inf'))
    p = torch.softmax(s, dim=-1)
    return torch.einsum('hts,shd->thd', p, v[r*L:(r+1)*L].float())

oB = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=H, num_key_value_heads=H,
    input_layout='TND', actual_seq_lengths=ends, actual_seq_lengths_kv=ends, scale=scale,
    sparse_mode=3, atten_mask=m2048, softmax_lse_flag=True)[0]
d0 = (oB[:L].float() - ref_req(0, q, k, v, L)).abs().max().item()
d1 = (oB[L:].float() - ref_req(1, q, k, v, L)).abs().max().item()
print('B_vs_ref: req0=%.4f req1=%.4f' % (d0, d1), flush=True)

resE = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=H, num_key_value_heads=H,
    input_layout='TND', actual_seq_lengths=ends, actual_seq_lengths_kv=ends, scale=scale,
    sparse_mode=0, atten_mask=None, softmax_lse_flag=True)
oE = resE[0]
e0 = (oE[:L].float() - ref_req(0, q, k, v, L)).abs().max().item()
e1 = (oE[L:].float() - ref_req(1, q, k, v, L)).abs().max().item()
print('E_nofmask_isolation: req0=%.4f req1=%.4f' % (e0, e1), flush=True)

outs = []
for r in range(B):
    qr = q[r*L:(r+1)*L].unsqueeze(0); kr = k[r*L:(r+1)*L].unsqueeze(0); vr = v[r*L:(r+1)*L].unsqueeze(0)
    am = torch.triu(torch.ones(L, L, device='npu', dtype=torch.bool), 1).unsqueeze(0).unsqueeze(0)
    rd = torch_npu.npu_fused_infer_attention_score(qr, kr, vr, num_heads=H, num_key_value_heads=H,
        scale=scale, input_layout='BSND', atten_mask=am, sparse_mode=0, softmax_lse_flag=True)
    outs.append(rd[0].squeeze(0))
oD = torch.cat(outs, dim=0)
dB = (oB.float() - oD.float()).abs().max().item()
dE = (oE.float() - oD.float()).abs().max().item()
print('B_vs_loop=%.4f E_vs_loop=%.4f' % (dB, dE), flush=True)
print('PROBE5H3_DONE', flush=True)

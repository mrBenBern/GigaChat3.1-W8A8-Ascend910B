
import torch, torch_npu, functools
print = functools.partial(print, flush=True)
torch.npu.set_device(0)
torch.manual_seed(0)
H, Dq, Dv = 32, 192, 192
scale = Dq ** -0.5
z2048 = torch.zeros(2048, 2048, device='npu', dtype=torch.int8)
m2048 = torch.triu(torch.ones(2048, 2048, device='npu'), 1).to(torch.int8)

def ref(r, q, k, v, lens, causal):
    Lq = lens[0][r]; Lk = lens[1][r]
    qs = sum(lens[0][:r]); ks = sum(lens[1][:r])
    s = torch.einsum('thd,shd->hts', q[qs:qs+Lq].float(), k[ks:ks+Lk].float()) * scale
    if causal:
        Lq_, Lk_ = s.shape[-2], s.shape[-1]
        off = Lk_ - Lq_
        s = s.masked_fill(torch.triu(torch.ones(Lq_, Lk_, device='npu'), off + 1).bool().unsqueeze(0), float('-inf'))
    p = torch.softmax(s, dim=-1)
    return torch.einsum('hts,shd->thd', p, v[ks:ks+Lk].float())

def case(tag, lens_q, lens_kv, mask, sparse, causal_ref):
    T = sum(lens_q); S = sum(lens_kv)
    q = torch.randn(T, H, Dq, dtype=torch.bfloat16, device='npu')
    k = torch.randn(S, H, Dq, dtype=torch.bfloat16, device='npu')
    v = torch.randn(S, H, Dv, dtype=torch.bfloat16, device='npu')
    eq = []; acc = 0
    for x in lens_q: acc += x; eq.append(acc)
    ek = []; acc = 0
    for x in lens_kv: acc += x; ek.append(acc)
    o = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=H, num_key_value_heads=H,
        input_layout='TND', actual_seq_lengths=eq, actual_seq_lengths_kv=ek, scale=scale,
        sparse_mode=sparse, atten_mask=mask, softmax_lse_flag=True)[0]
    worst = 0.0
    for r in range(len(lens_q)):
        d = (o[sum(lens_q[:r]):sum(lens_q[:r])+lens_q[r]].float() - ref(r, q, k, v, [lens_q, lens_kv], causal_ref)).abs().max().item()
        worst = max(worst, d)
    print('%s worst=%.4f' % (tag, worst), flush=True)

case('H2_uneven_fresh_triu_vs_CAUSAL', [300, 740], [300, 740], m2048, 3, True)
case('F2_even_ctx_zeromask_vs_CAUSAL', [520, 520], [1024, 1024], z2048, 3, True)
case('F3_even_ctx_zeromask_vs_FULL', [520, 520], [1024, 1024], z2048, 3, False)

# L: per-request loop TND B=1 no-mask for uneven context
lens_q = [300, 700]; lens_kv = [640, 1408]
T = sum(lens_q); S = sum(lens_kv)
q = torch.randn(T, H, Dq, dtype=torch.bfloat16, device='npu')
k = torch.randn(S, H, Dq, dtype=torch.bfloat16, device='npu')
v = torch.randn(S, H, Dv, dtype=torch.bfloat16, device='npu')
worst = 0.0
for r in range(2):
    qs = sum(lens_q[:r]); ks = sum(lens_kv[:r])
    o = torch_npu.npu_fused_infer_attention_score(q[qs:qs+lens_q[r]], k[ks:ks+lens_kv[r]], v[ks:ks+lens_kv[r]],
        num_heads=H, num_key_value_heads=H, input_layout='TND', actual_seq_lengths=[lens_q[r]],
        actual_seq_lengths_kv=[lens_kv[r]], scale=scale, sparse_mode=0, atten_mask=None, softmax_lse_flag=True)[0]
    d = (o.float() - ref(r, q, k, v, [lens_q, lens_kv], False)).abs().max().item()
    worst = max(worst, d)
print('L_loop_ctx_uneven_vs_FULL worst=%.4f' % worst, flush=True)
print('PROBE5H5_DONE', flush=True)

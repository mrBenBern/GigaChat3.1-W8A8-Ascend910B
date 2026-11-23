
import torch, torch_npu, functools
print = functools.partial(print, flush=True)
torch.npu.set_device(0)
torch.manual_seed(0)
H, Dq, Dv = 32, 192, 192
scale = Dq ** -0.5
z2048 = torch.zeros(2048, 2048, device='npu', dtype=torch.int8)
m2048 = torch.triu(torch.ones(2048, 2048, device='npu'), 1).to(torch.int8)

def ref_full(r, q, k, v, lens):
    Lq = lens[0][r]; Lk = lens[1][r]
    qs = sum(lens[0][:r]); ks = sum(lens[1][:r])
    s = torch.einsum('thd,shd->hts', q[qs:qs+Lq].float(), k[ks:ks+Lk].float()) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum('hts,shd->thd', p, v[ks:ks+Lk].float())

def run_case(tag, lens_q, lens_kv, mask, sparse):
    T = sum(lens_q); S = sum(lens_kv)
    q = torch.randn(T, H, Dq, dtype=torch.bfloat16, device='npu')
    k = torch.randn(S, H, Dq, dtype=torch.bfloat16, device='npu')
    v = torch.randn(S, H, Dv, dtype=torch.bfloat16, device='npu')
    eq = []; acc = 0
    for x in lens_q: acc += x; eq.append(acc)
    ek = []; acc = 0
    for x in lens_kv: acc += x; ek.append(acc)
    res = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=H, num_key_value_heads=H,
        input_layout='TND', actual_seq_lengths=eq, actual_seq_lengths_kv=ek, scale=scale,
        sparse_mode=sparse, atten_mask=mask, softmax_lse_flag=True)
    o = res[0]
    worst = 0.0
    for r in range(len(lens_q)):
        d = (o[sum(lens_q[:r]):sum(lens_q[:r])+lens_q[r]].float() - ref_full(r, q, k, v, [lens_q, lens_kv])).abs().max().item()
        worst = max(worst, d)
    print('%s worst_diff=%.4f' % (tag, worst), flush=True)

run_case('F_even_ctx_zero_mask [520,520]/[1024,1024]', [520, 520], [1024, 1024], z2048, 3)
run_case('G_uneven_ctx_zero_mask [300,700]/[640,1408]', [300, 700], [640, 1408], z2048, 3)
run_case('H_uneven_fresh_triumask [300,740]/[300,740]', [300, 740], [300, 740], m2048, 3)
print('PROBE5H4_DONE', flush=True)

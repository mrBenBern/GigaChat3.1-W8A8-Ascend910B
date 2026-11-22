
import torch, torch_npu, functools
print = functools.partial(print, flush=True)
torch.npu.set_device(0)
torch.manual_seed(0)
H, Dq, Dv = 32, 192, 128
scale = Dq ** -0.5

def cpu_ref(q, k, v, ends_q, ends_k):
    outs = []
    lses = []
    qf = q.float(); kf = k.float(); vf = v.float()
    def rows(ends):
        prev = 0
        segs = []
        for e in ends:
            segs.append((prev, e))
            prev = e
        return segs
    qs = rows(ends_q); ks = rows(ends_k)
    for r in range(len(qs)):
        a, b = qs[r]
        c, d = ks[r]
        if d == c:
            outs.append(torch.zeros(b - a, H, Dv))
            lses.append(torch.full((H, b - a), float('-inf')))
            continue
        s = torch.einsum('thd,shd->hts', qf[a:b], kf[c:d]) * scale
        p = torch.softmax(s, dim=-1)
        o = torch.einsum('hts,shd->htd', p, vf[c:d])   # [H,Ts,Dv]
        lse = torch.logsumexp(s, dim=-1)               # [H,Ts]
        outs.append(o)
        lses.append(lse)
    out = torch.cat(outs, dim=1)    # [H,T,Dv]
    lse = torch.cat(lses, dim=1)    # [H,T]
    return out, lse

q = torch.randn(8, H, Dq, dtype=torch.bfloat16, device='npu')
k = torch.randn(12, H, Dq, dtype=torch.bfloat16, device='npu')
v = torch.randn(12, H, Dv, dtype=torch.bfloat16, device='npu')
res = torch_npu.npu_fused_infer_attention_score(
    q, k, v, num_heads=H, num_key_value_heads=H, input_layout='TND',
    actual_seq_lengths=[4, 8], actual_seq_lengths_kv=[4, 12],
    scale=scale, sparse_mode=0, atten_mask=None, softmax_lse_flag=True)
out, lse = res[0], res[1]
print('TND_PROBE out', list(out.shape), out.dtype, '| lse', list(lse.shape), lse.dtype)
ref_o, ref_l = cpu_ref(q, k, v, [4, 8], [4, 12])   # ref_o [H,T,Dv], ref_l [H,T]
diff_o = (out.float().transpose(0, 1) - ref_o.to('npu')).abs().max().item()
diff_l = (lse.reshape(8, H).transpose(0, 1).float() - ref_l.to('npu')).abs().max().item()
print('TND_PROBE diff_out=%.4f diff_lse=%.4f' % (diff_o, diff_l))

Sp, Sc = 16, 8
qp = torch.randn(Sc, H, Dq, dtype=torch.bfloat16, device='npu')
kp = torch.randn(Sp, H, Dq, dtype=torch.bfloat16, device='npu')
vp = torch.randn(Sp, H, Dv, dtype=torch.bfloat16, device='npu')
kc = torch.randn(Sc, H, Dq, dtype=torch.bfloat16, device='npu')
vc = torch.randn(Sc, H, Dv, dtype=torch.bfloat16, device='npu')
r1 = torch_npu.npu_fused_infer_attention_score(qp, kp, vp, num_heads=H, num_key_value_heads=H,
    input_layout='TND', actual_seq_lengths=[Sc], actual_seq_lengths_kv=[Sp],
    scale=scale, sparse_mode=0, atten_mask=None, softmax_lse_flag=True)
r2 = torch_npu.npu_fused_infer_attention_score(qp, kc, vc, num_heads=H, num_key_value_heads=H,
    input_layout='TND', actual_seq_lengths=[Sc], actual_seq_lengths_kv=[Sc],
    scale=scale, sparse_mode=0, atten_mask=None, softmax_lse_flag=True)
o1, l1 = r1[0], r1[1]
o2, l2 = r2[0], r2[1]
l1ht = l1.reshape(Sc, H).transpose(0, 1).float()   # [H,Sc]
l2ht = l2.reshape(Sc, H).transpose(0, 1).float()
new_l = torch.logaddexp(l1ht, l2ht)                # [H,Sc]
w1 = torch.exp(l1ht - new_l)
w2 = torch.exp(l2ht - new_l)
o1h = o1.float().transpose(0, 1)                   # [H,Sc,Dv]
o2h = o2.float().transpose(0, 1)
merged_h = w1.unsqueeze(-1) * o1h + w2.unsqueeze(-1) * o2h   # [H,Sc,Dv]
kf = torch.cat([kp, kc], dim=0)
vf = torch.cat([vp, vc], dim=0)
r3 = torch_npu.npu_fused_infer_attention_score(qp, kf, vf, num_heads=H, num_key_value_heads=H,
    input_layout='TND', actual_seq_lengths=[Sc], actual_seq_lengths_kv=[Sp + Sc],
    scale=scale, sparse_mode=0, atten_mask=None, softmax_lse_flag=True)
o3, l3 = r3[0], r3[1]
o3h = o3.float().transpose(0, 1)
diff = (merged_h - o3h).abs().max().item()
dlse = (new_l - l3.reshape(Sc, H).transpose(0, 1).float()).abs().max().item()
print('MERGE_CHECK merged_vs_full diff=%.4f lse_diff=%.4f' % (diff, dlse))
ok = diff < 0.05 and dlse < 0.05 and diff_o < 0.05
print('PROBE5G_%s' % ('OK' if ok else 'CHECK'), flush=True)

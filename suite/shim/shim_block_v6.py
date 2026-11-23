
_GIGA_TND_FRESH = True
_GIGA_MASK2048 = None

def _giga_shim(**kw):
    import torch, torch_npu, functools
    pr = functools.partial(print, flush=True)
    global _GIGA_MASK2048
    cnt = getattr(_giga_shim, 'cnt', 0)
    _giga_shim.cnt = cnt + 1
    if cnt < 60:
        try:
            def d(x):
                if x is None: return 'None'
                if torch.is_tensor(x): return 'T%s%s' % (list(x.shape), str(x.dtype))
                if isinstance(x, (list, tuple)): return 'L%d:%s' % (len(x), str(list(x)[:3])[:70])
                return str(x)[:40]
            pr('[SHIM%d] %s' % (cnt, {k: d(v) for k, v in sorted(kw.items())}), flush=True)
        except Exception as e:
            pr('[SHIM%d] logerr %s' % (cnt, e), flush=True)
    q = torch.cat([kw['q_nope'], kw['q_rope']], dim=-1).contiguous()
    k = torch.cat([kw['k_nope'], kw['k_rope']], dim=-1).contiguous()
    v = kw['value'].contiguous()
    T, H, D = q.shape
    S = k.shape[0]
    seqlen = kw.get('seqlen')
    is_ctx = seqlen is not None and getattr(seqlen, 'dim', None) is not None and seqlen.dim() == 2
    if _GIGA_MASK2048 is None or _GIGA_MASK2048.device != q.device:
        _GIGA_MASK2048 = torch.triu(torch.ones(2048, 2048, device=q.device), 1).to(torch.int8)

    def _lse_ht(lse):
        if lse is None:
            return None
        if lse.reshape(-1).numel() != H * T:
            return None
        if lse.dim() == 3:
            return lse.reshape(T, H).transpose(0, 1).contiguous().float()
        return lse.reshape(H, T).contiguous().float()

    if is_ctx:
        pre_out = kw.get('pre_out')
        prev_lse = kw.get('prev_lse')
        out_buf = kw.get('output')
        lse_buf = kw.get('softmax_lse')
        ql = [int(x) for x in seqlen[0].tolist()]
        cl = [int(x) for x in seqlen[1].tolist()]
        B = len(ql)
        pr('[CTXINFO] B=%d T=%d S=%d ql=%s cl=%s' % (B, T, S, ql[:12], cl[:12]), flush=True)
        if B == 1:
            res = torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=H, num_key_value_heads=kw.get('kv_head_num', H) or H,
                scale=kw.get('qk_scale', D ** -0.5), input_layout='TND',
                actual_seq_lengths=[T], actual_seq_lengths_kv=[S],
                sparse_mode=0, atten_mask=None, softmax_lse_flag=True)
            attn, lse = res[0], res[1]
            lse_ht = _lse_ht(lse)
            merged = False
            if pre_out is not None and prev_lse is not None and lse_ht is not None:
                prev = pre_out.transpose(0, 1).float()
                prev_l = prev_lse.float()
                if bool(torch.isfinite(prev_l).all()):
                    new_l = torch.logaddexp(prev_l, lse_ht)
                    w_prev = torch.exp(prev_l - new_l)
                    w_cur = torch.exp(lse_ht - new_l)
                    merged_t = w_prev.unsqueeze(-1) * prev + w_cur.unsqueeze(-1) * attn.transpose(0, 1).float()
                    if out_buf is not None:
                        out_buf.copy_(merged_t.transpose(0, 1).to(out_buf.dtype))
                    if lse_buf is not None:
                        lse_buf.copy_(new_l)
                    merged = True
            if not merged:
                if out_buf is not None:
                    out_buf.copy_(attn.reshape(out_buf.shape))
                if lse_buf is not None and lse_ht is not None:
                    lse_buf.copy_(lse_ht)
        else:
            acc_q = 0
            acc_k = 0
            for r in range(B):
                lq = ql[r]
                lk = cl[r]
                qs = acc_q
                ks = acc_k
                acc_q += lq
                acc_k += lk
                lq_eff = min(lq, max(0, T - qs))
                lk_eff = min(lk, max(0, S - ks))
                if lq != lq_eff or lk != lk_eff:
                    pr('[LOOPCLIP] r=%d lq=%d->%d lk=%d->%d T=%d S=%d qs=%d ks=%d ql=%s cl=%s' % (r, lq, lq_eff, lk, lk_eff, T, S, qs, ks, ql[:12], cl[:12]), flush=True)
                lq = lq_eff
                lk = lk_eff
                if lq == 0 or lk == 0:
                    continue
                eq = [lq]
                ek = [lk]
                res = torch_npu.npu_fused_infer_attention_score(
                    q[qs:qs + lq], k[ks:ks + lk], v[ks:ks + lk],
                    num_heads=H, num_key_value_heads=kw.get('kv_head_num', H) or H,
                    scale=kw.get('qk_scale', D ** -0.5), input_layout='TND',
                    actual_seq_lengths=eq, actual_seq_lengths_kv=ek,
                    sparse_mode=0, atten_mask=None, softmax_lse_flag=True)
                attn_r, lse_r = res[0], res[1]
                lse_ht_r = _lse_ht(lse_r)
                if pre_out is not None and prev_lse is not None and lse_ht_r is not None:
                    prev = pre_out.transpose(0, 1).float()[:, qs:qs + lq]
                    prev_l = prev_lse.float()[:, qs:qs + lq]
                    if bool(torch.isfinite(prev_l).all()):
                        new_l = torch.logaddexp(prev_l, lse_ht_r)
                        w_prev = torch.exp(prev_l - new_l)
                        w_cur = torch.exp(lse_ht_r - new_l)
                        merged_t = w_prev.unsqueeze(-1) * prev + w_cur.unsqueeze(-1) * attn_r.transpose(0, 1).float()
                        if out_buf is not None:
                            out_buf[qs:qs + lq].copy_(merged_t.transpose(0, 1).to(out_buf.dtype))
                        if lse_buf is not None:
                            lse_buf[:, qs:qs + lq].copy_(new_l)
                    else:
                        if out_buf is not None:
                            out_buf[qs:qs + lq].copy_(attn_r.reshape(out_buf[qs:qs + lq].shape))
                        if lse_buf is not None and lse_ht_r is not None:
                            lse_buf[:, qs:qs + lq].copy_(lse_ht_r)
                else:
                    if out_buf is not None:
                        out_buf[qs:qs + lq].copy_(attn_r.reshape(out_buf[qs:qs + lq].shape))
                    if lse_buf is not None and lse_ht_r is not None:
                        lse_buf[:, qs:qs + lq].copy_(lse_ht_r)
        cstat = getattr(_giga_shim, 'cstat', 0)
        if cstat < 8:
            _giga_shim.cstat = cstat + 1
            pr('[CTXSTAT] mode=%s B=%d T=%d S_ctx=%d' % ('single' if B == 1 else 'loop', B, T, S), flush=True)
        return kw.get('output')

    am = None
    branch = 'fresh'
    m = kw.get('mask')
    if m is not None and kw.get('mask_type', 'mask_type_triu') != 'no_mask':
        if m.shape[-2] >= T and m.shape[-1] >= S:
            am = (m <= -1e8)[..., :T, :S]
            branch = 'sliced'
        else:
            am = torch.triu(torch.ones(T, S, device=q.device, dtype=torch.bool), 1)
            branch = 'fresh_triu'
        while am.dim() < 4:
            am = am.unsqueeze(0)
    Bn = 1
    if seqlen is not None and getattr(seqlen, 'numel', None) is not None and seqlen.dim() == 1:
        try:
            Bn = int(seqlen.numel())
        except Exception:
            Bn = 1
    used = 'bsnd'
    res = None
    if _GIGA_TND_FRESH and Bn > 1 and S == T:
        try:
            qlf = [int(x) for x in seqlen.tolist()]
            ends = []
            acc = 0
            for x in qlf:
                acc += x
                ends.append(acc)
            res = torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=H, num_key_value_heads=kw.get('kv_head_num', H) or H,
                scale=kw.get('qk_scale', D ** -0.5), input_layout='TND',
                actual_seq_lengths=ends, actual_seq_lengths_kv=ends,
                sparse_mode=3, atten_mask=_GIGA_MASK2048, softmax_lse_flag=True)
            used = 'tnd_sparse3'
            branch = 'fresh_varlen'
            pr('[FRESHVAR] B=%d T=%d ends=%s' % (Bn, T, ends[:12]), flush=True)
        except Exception as e:
            pr('[SHIMFRESH] tnd_sparse3 failed: %s -> bsnd' % str(e)[:160], flush=True)
            res = None
    if res is None:
        res = torch_npu.npu_fused_infer_attention_score(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), num_heads=H,
            num_key_value_heads=kw.get('kv_head_num', H) or H,
            scale=kw.get('qk_scale', D ** -0.5), input_layout='BSND', atten_mask=am,
            sparse_mode=0, softmax_lse_flag=True)
        used = 'bsnd'
    attn, lse = (res[0], res[1]) if isinstance(res, (tuple, list)) else (res, None)
    if attn.dim() == 4:
        attn = attn.squeeze(0)
    lse_ht = _lse_ht(lse)
    out = kw.get('output')
    if out is not None:
        out.copy_(attn.reshape(out.shape))
    if kw.get('softmax_lse') is not None and lse_ht is not None:
        kw['softmax_lse'].copy_(lse_ht)
    fstat = getattr(_giga_shim, 'fstat', 0)
    if fstat < 12:
        _giga_shim.fstat = fstat + 1
        pr('[FRESHSTAT] %s used=%s T=%d S=%d B=%d lse=%s' % (
            branch, used, T, S, Bn, 'ok' if lse_ht is not None else 'none'), flush=True)
    return out


def _giga_shim(**kw):
    import torch, torch_npu, functools
    pr = functools.partial(print, flush=True)
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
    qb = q.unsqueeze(0); kb = k.unsqueeze(0); vb = v.unsqueeze(0)
    seqlen = kw.get('seqlen')
    is_ctx = seqlen is not None and getattr(seqlen, 'dim', None) is not None and seqlen.dim() == 2

    def _lse_ht(lse, used):
        if lse is None:
            return None
        if lse.reshape(-1).numel() != H * T:
            return None
        if used == 'tnd':
            return lse.reshape(T, H).transpose(0, 1).contiguous().float()
        return lse.reshape(H, T).contiguous().float()

    def _write(out_buf, lse_buf, attn_t, lse_ht):
        if out_buf is not None:
            out_buf.copy_(attn_t.reshape(out_buf.shape))
        if lse_buf is not None and lse_ht is not None:
            lse_buf.copy_(lse_ht)

    if is_ctx:
        pre_out = kw.get('pre_out')
        prev_lse = kw.get('prev_lse')
        out_buf = kw.get('output')
        lse_buf = kw.get('softmax_lse')
        ql = seqlen[0].tolist()
        cl = seqlen[1].tolist()
        used = None
        res = None
        try:
            ends_q = []; acc = 0
            for x in ql: acc += int(x); ends_q.append(acc)
            ends_k = []; acc = 0
            for x in cl: acc += int(x); ends_k.append(acc)
            res = torch_npu.npu_fused_infer_attention_score(
                q, k, v, num_heads=H, num_key_value_heads=kw.get('kv_head_num', H) or H,
                scale=kw.get('qk_scale', D ** -0.5), input_layout='TND',
                actual_seq_lengths=ends_q, actual_seq_lengths_kv=ends_k,
                sparse_mode=0, atten_mask=None, softmax_lse_flag=True)
            used = 'tnd'
        except Exception as e:
            pr('[SHIMCTX] tnd failed: %s -> bsnd' % str(e)[:100], flush=True)
        if res is None:
            res = torch_npu.npu_fused_infer_attention_score(
                qb, kb, vb, num_heads=H, num_key_value_heads=kw.get('kv_head_num', H) or H,
                scale=kw.get('qk_scale', D ** -0.5), input_layout='BSND', atten_mask=None,
                sparse_mode=0, softmax_lse_flag=True)
            used = 'bsnd'
        attn, lse = (res[0], res[1]) if isinstance(res, (tuple, list)) else (res, None)
        if attn.dim() == 4:
            attn = attn.squeeze(0)
        lse_ht = _lse_ht(lse, used)
        if lse_ht is not None and any(int(x) == 0 for x in cl):
            req_id = torch.repeat_interleave(torch.arange(len(ql)), torch.tensor(ql).to(q.device))
            zt = torch.tensor(cl, dtype=torch.int32).to(q.device)[req_id] == 0
            if bool(zt.any()):
                lse_ht[:, zt] = float('-inf')
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
            else:
                pr('[CTXSTAT] prev_lse NOT FINITE -> flat', flush=True)
        if not merged:
            _write(out_buf, lse_buf, attn, lse_ht)
        cstat = getattr(_giga_shim, 'cstat', 0)
        if cstat < 4:
            _giga_shim.cstat = cstat + 1
            try:
                fin = bool(torch.isfinite(lse_ht).all()) if lse_ht is not None else None
                pr('[CTXSTAT] merged=%s used=%s T=%d S_ctx=%d lse_finite=%s' % (merged, used, T, S, fin), flush=True)
            except Exception as e:
                pr('[CTXSTAT] err %s' % e, flush=True)
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
    res = torch_npu.npu_fused_infer_attention_score(
        qb, kb, vb, num_heads=H, num_key_value_heads=kw.get('kv_head_num', H) or H,
        scale=kw.get('qk_scale', D ** -0.5), input_layout='BSND', atten_mask=am,
        sparse_mode=0, softmax_lse_flag=True)
    attn, lse = (res[0], res[1]) if isinstance(res, (tuple, list)) else (res, None)
    attn = attn.squeeze(0)
    lse_ht = _lse_ht(lse, 'bsnd')
    out = kw.get('output')
    _write(out, kw.get('softmax_lse'), attn, lse_ht)
    if getattr(_giga_shim, 'mstat', 0) < 2 and am is not None:
        _giga_shim.mstat = getattr(_giga_shim, 'mstat', 0) + 1
        try:
            ref = torch.triu(torch.ones(T, S, device=am.device, dtype=torch.bool), 1)
            a2 = am if am.dim() == 2 else am[0, 0] if am.dim() == 4 else am.reshape(T, S)
            pr('[MASKSTAT] branch=%s T=%d S=%d sum=%d expected=%d row0=%d/%d rowLast=%d equal_ref=%s' % (
                branch, T, S, int(a2.sum()), int(ref.sum()), int(a2[0].sum()), S - 1, int(a2[-1].sum()),
                bool(torch.equal(a2, ref))), flush=True)
        except Exception as e:
            pr('[MASKSTAT] err %s' % e, flush=True)
    fstat = getattr(_giga_shim, 'fstat', 0)
    if fstat < 2:
        _giga_shim.fstat = fstat + 1
        pr('[FRESHSTAT] %s T=%d S=%d lse=%s' % (branch, T, S, 'ok' if lse_ht is not None else 'none'), flush=True)
    return out

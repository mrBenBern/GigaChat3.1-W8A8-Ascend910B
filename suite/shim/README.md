
# GigaChat3.1 W8A8 on Ascend 910B — FIA shim (final, v10)

AtbRingMLAGetWorkspaceSize crashes on real prefill (vllm-ascend 0.11.0/0.13.0, CANN 8.3/8.5, torch_npu 2.7.1/2.8.0.post5).
Fix = direct patch of installed vllm_ascend/attention/mla_v1.py (71-line diff):

1. `_giga_shim(**kw)` replaces both `torch_npu.atb.npu_ring_mla(` call sites:
   decompress cat(q_nope,q_rope)->[T,H,192], cat(k_nope,k_rope)->[S,H,192], value=[S,H,192]
   -> torch_npu.npu_fused_infer_attention_score(BSND [1,T,H,D] / [1,S,H,D],
      num_heads/num_key_value_heads kwargs, scale=qk_scale, atten_mask bool, sparse_mode=0)
2. mask: ring mask is 512x512 -> rebuild bool triu [T,S] when 512 < T; bool semantics (True=masked)
3. `_v_up_proj` _C_ascend batch_matmul_transpose fast-path disabled (PyPI wheel has no binary) -> torch.bmm

Apply:
    cp vllm_ascend/attention/mla_v1.py mla_v1.py.bak
    patch vllm_ascend/attention/mla_v1.py < giga31_shim.patch
Serve:
    vllm serve /tmp/models/giga31_w8a8 --port 8000 --max-model-len 4096 \
      --max-num-batched-tokens 4096 --enforce-eager --quantization ascend

## Status (2026-11-22, task 941568)
- FIRST LIVE TOKENS of GigaChat3.1-10B-A1.8B (W8A8) on Ascend 910B: 2026-11-22 ~04:50 UTC+8
- Serving OK via OpenAI API; decode ~10 tps, prefill ~700 tps (eager, single 910B 64GB)
- Chat template adds ~1034 role_description tokens to every request (model design, RAW user prompt = 6 tokens)
- vLLM seed=0 -> deterministic samples

## Mask experiments (prefill, seed=0)
- bool (True=masked): generation runs, output coherent-looking but continues foreign context
- float bf16 min/0: REJECTED by FIA (561002) — this build accepts bool mask only
- bool inverted (True=visible): generation runs, different output, still not question-focused

## Open quality hypotheses (next session)
1. kv-cache int8 (kv_cache_dtype=auto) vs decompressed path mismatch
2. qk_scale YaRN mscale handling
3. W8A8 quantization quality itself (A/B vs bf16 on same stack)
4. FIA actual_seq_lengths semantics for BSND

Evidence: evidence/5c/ (smoke answers, serve log keylines)

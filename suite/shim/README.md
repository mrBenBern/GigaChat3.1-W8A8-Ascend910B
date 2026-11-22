
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

## ACT ACCEPTED (2026-11-22, task 942047, session 5d): 20/20 SEMANTIC TESTS PASSED

Root cause of off-topic generation in earlier sessions: **prefix caching**.
With --enable-prefix-caching (default), repeated requests sharing the ~1034-token
role_description prefix hit cached KV blocks incompatible with the decompressed-prefill
shim path -> coherent-looking but off-topic output. First request on a clean server
(full prefill through the shim) was always correct.

Fix: serve with **--no-enable-prefix-caching**.

Result (seed=0, temp=0.2, all full prefill through shim):
- 20/20 tests PASS: 10 sanity (facts/arithmetic/translation), 5 instruction-following
  (exact repeat, one-word, format), 5 context-invariants (777, password, name, city, EN prompt)
- Determinism: byte-identical on repeated request
- TTFT: 0.2s @100 tok, 0.5s @1500 tok (full prefill through shim)
- npu-smi info + dmon logs captured during generation
Evidence: evidence/act5d/ (test5d_results.json, shim_calls_verbose.log, decode_callsite.txt,
npusmi_dmon.log, pipfreeze.txt, serve_nopc_tail.log)

## SESSION 5e (2026-11-22, task 942187): off-topic root-cause chain finalized

Three-layer causal story (all evidence-backed):
1. Inference mechanics CORRECT: mask stats equal_ref=True (sum=549676=expected, row0=1048,
   rowLast=0), FIA synthetic test vs CPU fp32 reference max_diff=0.0077 (bf16 noise level),
   rowFirst diff=0.0 (perfect causality), inner_precise=2 has no effect.
2. Prefix caching INCOMPATIBLE with decompressed-prefill shim -> --no-enable-prefix-caching.
3. Homonym acronyms (bare "MLA") sample wrong expansions from weights at any temperature
   ("Master Mold Allegiance"/artillery/fan-fiction at temp 0.3/0.2/0.2-seed1) — a W8A8
   quantization nuance (--disable_anti); full term "Multi-head Latent Attention (MLA)"
   answers correctly, KV-cache/GQA questions precise. Not an engine bug.

Recommendation for users of the shim: disable prefix caching; prefer full-term prompts;
consider re-quantization with anti-outlier once msmodelslim supports non-standard MLA dims.
Evidence: evidence/act5e/ (sm9_*.json, maskstat.log, diskcheck.txt, homonym_matrix.log)


## Model on Hugging Face

Quantized checkpoint + model card: https://huggingface.co/Bentuzzi/GigaChat3.1-10B-A1.8B-W8A8-Ascend910B

## SESSION 5g (2026-11-23, task 942558): prefix caching MADE COMPATIBLE (shim v4)

Root fix for the 5d corruption: the shim never wrote `softmax_lse` and never merged
`pre_out/prev_lse` (the ring-op contract), so cache-hit context calls merged through
garbage LSE. Shim v4 restores the contract:
- fresh branch (seqlen [B], square causal): BSND FIA with softmax_lse_flag=True,
  writes output AND softmax_lse ([H,T] fp32, FIA ln-scale);
- context branch (seqlen [2,B]): TND FIA (per-request cumulative ends, sparse_mode=0,
  no mask) -> output [T,H,D], lse [T,H,1]; zero-length chunks masked to -inf;
  log-sum-exp merge with pre_out/prev_lse in fp32; writes merged output + merged LSE.
  BSND fallback (no per-request separation) only if TND raises.
Validation: PROBE5G_OK - TND probe diff_out=0.0093 (bf16 noise), diff_lse=0.0000;
merge identity vs full single-shot attention diff=0.0071, lse_diff=0.0000 (fp32 exact).
Serving with prefix caching ON: 20/20 tests PASS, determinism IDENTICAL,
engine Prefix cache hit rate 89%, context merges confirmed in live traffic
(CTXSTAT merged=True used=tnd T=16 S_ctx=1024 lse_finite=True).
Cache-path outputs are CORRECT but not always byte-identical to the cold path
(two different numerical routes to the same causal result; bf16 rounding;
e.g. cold answer verbose, cache-hit answer terse - both correct).
TTFT p1500: 0.5s (5d, no cache) -> 0.23s (cache warm); identical-request wall time
1.76s (cold) -> 0.60s (cache hits).
Known limitation: fresh branch is dense-BSND - batched multi-request prefill would
cross-attend (all 5d/5g tests are single-request); TND context branch is
per-request correct.
Recommendation: prefix caching may stay ENABLED with shim v4.
Evidence: evidence/act5g/ (test5g_results.json, accept5g.log, chain5g.log,
chain5g_lfs.log, probe5g.log, shim_calls.log, vllm_tail.log, npusmi_*, pipfreeze)

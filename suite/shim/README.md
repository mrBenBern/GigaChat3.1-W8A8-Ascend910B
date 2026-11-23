
# GigaChat3.1 W8A8 on Ascend 910B — FIA shim (final, v10)

AtbRingMLAGetWorkspaceSize crashes on real prefill (vllm-ascend 0.11.0/0.13.0, CANN 8.3/8.5, torch_npu 2.7.1/2.8.0.post5; on 0.13.0 the crash already fires in profile_run; the shim itself is verified on 0.11.0 only). Consolidated crash record: evidence/crash/README.md.
Fix = shim installed into vllm_ascend/attention/mla_v1.py (current block: v6; the v1 diff below is kept for reference):

1. `_giga_shim(**kw)` replaces both `torch_npu.atb.npu_ring_mla(` call sites:
   decompress cat(q_nope,q_rope)->[T,H,192], cat(k_nope,k_rope)->[S,H,192], value=[S,H,192]
   -> torch_npu.npu_fused_infer_attention_score(BSND [1,T,H,D] / [1,S,H,D],
      num_heads/num_key_value_heads kwargs, scale=qk_scale, atten_mask bool, sparse_mode=0)
2. mask: ring mask is 512x512 -> rebuild bool triu [T,S] when 512 < T; bool semantics (True=masked)
3. `_v_up_proj` _C_ascend batch_matmul_transpose fast-path disabled (PyPI wheel has no binary) -> torch.bmm

Apply (current shim v6):
    python apply_shim.py       # backs up mla_v1.py, prepends shim_block_v6.py, replaces both call sites
Legacy v1 (the 71-line diff described below, single-request only):
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

Fix at the time (5d): serve with **--no-enable-prefix-caching**.
*Superseded 2026-11-23: shim v4 implements the ring-merge contract, so prefix caching is
compatible for single-request serving (see SESSION 5g); for concurrent traffic the flag is
still recommended (SESSION 5h).*

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
2. Prefix caching INCOMPATIBLE with decompressed-prefill shim (as of shim v1–v3) -> --no-enable-prefix-caching.
   *(fixed in shim v4, see SESSION 5g)*
3. Homonym acronyms (bare "MLA") sample wrong expansions from weights at any temperature
   ("Master Mold Allegiance"/artillery/fan-fiction at temp 0.3/0.2/0.2-seed1) — a W8A8
   quantization nuance (--disable_anti); full term "Multi-head Latent Attention (MLA)"
   answers correctly, KV-cache/GQA questions precise. Not an engine bug.

Recommendation as of 5e (first clause superseded by shim v4, see 5g/5h): disable prefix caching; prefer full-term prompts;
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
Recommendation: prefix caching may stay ENABLED with shim v4 (single-request serving; for concurrency see SESSION 5h).
Evidence: evidence/act5g/ (test5g_results.json, accept5g.log, chain5g.log,
chain5g_lfs.log, probe5g.log, shim_calls.log, vllm_tail.log, npusmi_*, pipfreeze)

## SESSION 5h (2026-11-23, task 943225): concurrency smoke — shim v6 (fresh-varlen + per-request context loop)

Question left open after 5g ("is the dense fresh branch at batch>1 a crash or
silent corruption?") answered empirically:

- Phase A (v4 behavior, PC ON, 8 parallel mixed prompts): MIXED8 = 2/8 — NO crash,
  SILENT CORRUPTION (cross-attention between different requests merged into one
  prefill step). SAME8 = 8/8 (identical prompts cannot detect cross-attention).
- Phase B (v5 attempt, TND varlen fresh): 182x FIA rejection
  "When Layout is TND, sparseMode=0, not support attentionMask" -> silent fallback
  to dense BSND -> corruption remained. Lesson: silent operator fallbacks hide
  contract violations; assert, don't fall back.

Isolated probes (probe5h2-6.py, seed=0, H=32 D=192):
- TND + sparse_mode=3 + triu int8 2048x2048 mask = CORRECT fresh-varlen kernel
  (max diff 0.0039-0.0082 vs per-request reference, even and uneven lengths);
- TND without mask does NOT isolate requests (diff ~4.0) — for context attention
  only a per-request loop of TND no-mask calls is correct (diff 0.0022);
- npu_paged_cache_load packs output rows request-major ([0,0,2,2] / [0,2,2]).

Shim v6 (shim_block_v6.py, supersedes v4/v5): 4 branches — fresh B=1 dense BSND;
fresh B>1 TND+sparse3 varlen; context B=1 single TND call; context B>1 per-request
TND loop with fp32 log-sum-exp merge into pre_out/prev_lse.

Concurrent run history (40-task 8-way burst, semantic asserts):
v6 21/40 (offset bug: acc not advanced on skip) -> v6.1 0/40 crash
"queryT(6) must equal last element of actualSeqenceLengthQ(31)" (double
accumulation -> slices overrun; also documents the 0.11.0 ring contract:
row0=query_lens of THIS step, q carries only scheduled tokens) ->
v6.2 instrumented (CTXINFO/LOOPCLIP/FRESHVAR telemetry, slice guards) 31/40 ->
v6.2 fixed, PC ON 13/40 -> PC OFF (--no-enable-prefix-caching) 38/40 in 11.8s.

Verdict:
- fresh-varlen (TND+sparse3) is correct under 8-way concurrency (208 FRESHVAR
  calls, zero exceptions, probes exact);
- with prefix caching ON the context-merge path still corrupts under batching
  (13/40) — the per-request loop + LSE merge is numerically correct in probes
  but production-grade chunked-context batching needs upstream work
  (0.11.0 calls _compute_prefill_context per chunk with full-tensor pre_out);
- RECOMMENDATION for concurrent serving on this stack: --no-enable-prefix-caching
  (38/40; remaining 2 BAD are template-echo answers - <|pretext|> / <think> -
  suspected native decode path at B>1, not the shim; single-request acceptance
  from 5d/5g remains the canonical gate).
For "first publicly documented W8A8 inference on Ascend 910B" nothing changes:
single-request 20/20 + determinism + PC 89% hit rate (5g) is the claim basis;
concurrency is documented as best-effort with explicit numbers.

Note: npu-smi dmon/info files are empty this session (collector did not flush);
hardware-execution evidence for the checkpoint is in act5d/act5g runs.
Evidence: evidence/act5h/ (5 run JSONs incl. phaseA MIXED8/SAME8, phase logs,
chain5h.log, chain5h_lfs.log, patch5h.log, patch6.log, shim_telemetry.log,
vllm tails x3, pipfreeze); suite/shim/{shim_block_v6.py, probe5h2..probe5h6.py}.

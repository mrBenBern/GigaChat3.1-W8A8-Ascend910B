# GigaChat3.1-10B-A1.8B — W8A8 inference on Ascend 910B

**First publicly documented W8A8 inference of GigaChat3.1-10B-A1.8B (MoE, MIT) on Huawei Ascend 910B** — an attention shim patch for vllm-ascend, a reproducible pipeline, raw evidence, and a full inference act.

> Hugging Face model: [Bentuzzi/GigaChat3.1-10B-A1.8B-W8A8-Ascend910B](https://huggingface.co/Bentuzzi/GigaChat3.1-10B-A1.8B-W8A8-Ascend910B) · Mirror: [OpenI MutTechDev/onion-openi](https://openi.pcl.ac.cn/MutTechDev/onion-openi)
> Based on [ai-sage/GigaChat3.1-10B-A1.8B-bf16](https://modelscope.cn/models/ai-sage/GigaChat3.1-10B-A1.8B-bf16) (MIT, © Sber)

**RU TL;DR:** первый публично документированный запуск GigaChat3.1-10B-A1.8B (W8A8, 11 ГБ) на Ascend 910B через открытый стек vLLM + vllm-ascend. Крах родного MLA-оператора обойдён шимом на двух точках вызова (v6), качество закрыто 20/20 контрольными тестами плюс 38/40 при 8 параллельных запросах, детерминизм байт-в-байт, полный акт инференса и все сырые evidence — в этом репозитории.

## Results (2026-11-22/23, OpenI tasks 942047 / 942558 / 943225)

| Metric | Value |
|---|---|
| Semantic tests | **20/20 PASS** (10 sanity · 5 instruction-following · 5 context-invariants) |
| Determinism | byte-identical on repeats (seed=0) |
| TTFT | 0.2s @1k tokens · 0.5s @1.5k tokens (full prefill through shim) |
| Full QA round-trip | 0.3–1.5s |
| Attention operator | verified vs CPU-fp32 reference: max_diff 0.0077 (bf16 noise), causality perfect |
| Mask | formally verified: sum/rows/`equal_ref=True` against causal reference |
| NPU evidence | pip freeze + full engine serve logs archived; npu-smi capture FAILED in 5d/5g (files empty — honest note, recapture planned; see docs/ACT-INFERENCE.md) |
| Prefix caching (shim v4, 2026-11-23) | **compatible**: 20/20 with caching ON, engine hit rate 89%, identical request 1.76s → 0.60s |
| Concurrency (shim v6, 2026-11-23) | 8 parallel requests with different prompts: **38/40** with `--no-enable-prefix-caching`; with caching ON the batched context merge still corrupts (13/40) |

## The problem

Native vllm-ascend MLA prefill operator crashes on this model: `npu_ring_mla` fails with `AtbRingMLAGetWorkspaceSize` and reports no reason even at `ATB_LOG_LEVEL=DEBUG`. Reproduced live on vllm-ascend **0.11.0** (torch_npu 2.7.1/CANN 8.3 and 2.8.0.post5/CANN 8.5 — `profile_run` passes, the first real prefill dies, 1049–1066 tokens) and **0.13.0** (torch_npu 2.8.0.post2/CANN 8.5 — the crash already fires inside `profile_run`). GigaChat3.1 has **32 attention heads** with qk_nope 128 + qk_rope 64 = 192 and **v_head_dim 192**: Q=K=V=192 is legal per CANN documentation, and in our synthetic harness the ring operator executed real arithmetic only with 128 heads. Consolidated record + head-count matrix: [`evidence/crash/README.md`](evidence/crash/README.md).

## The fix — a shim on the two `npu_ring_mla` call-sites

The description below is the original 71-line patch (shim v1). The current shim (**v6**, `suite/shim/shim_block_v6.py`) keeps the same call-site replacement and adds the ring-merge contract (fp32 LSE + log-sum-exp merge, v4) and batched fresh-varlen prefill (v6).

`suite/shim/giga31_shim.patch` (the legacy v1 diff) patches the installed `vllm_ascend/attention/mla_v1.py`:

1. `_giga_shim(**kw)` replaces both `npu_ring_mla` call-sites: decompress `cat(q_nope,q_rope)→[T,32,192]`, `cat(k_nope,k_rope)→[S,32,192]` → `npu_fused_infer_attention_score` (BSND `[1,T,H,192]` **without transpose**, `num_heads`/`num_key_value_heads` kwargs, `scale=qk_scale`, bool causal mask, `sparse_mode=0`)
2. Ring mask is only **512×512** → rebuild causal bool mask at **T×S** when 512 < T
3. `_v_up_proj` `_C_ascend` fast-path disabled (PyPI wheel has no binary) → `torch.bmm`

Decode runs the native absorb path — mathematically equivalent (linear W_UV commutes with the attention-weighted sum).

## Two root causes we hit on the way (so you don't have to)

1. **Prefix caching broke the shim until v4** (fixed 2026-11-23): on cache hits the shim skipped the ring-operator merge contract (no `softmax_lse`), so repeated requests sharing the ~1034-token role_description prefix produced coherent-looking but off-topic output. Since shim v4 the contract is implemented — 20/20 with prefix caching ON, hit rate 89%. For **concurrent** traffic keep `--no-enable-prefix-caching` until the batched context merge is fixed (shim v6: 38/40 at 8-way with the flag, 13/40 with caching ON).
2. **Bare homonym acronyms** (e.g. "MLA") may sample wrong expansions at higher temperatures — a W8A8-without-anti-outlier nuance. Use full terms ("Multi-head Latent Attention (MLA)") or re-quantize with anti-outlier once tooling supports non-standard MLA dims.

## Quickstart

```bash
# 1) restore checkpoint (42 LFS chunks, per-chunk sha256) — see openi-suite/RECOVER.md
# 2) stack
pip install torch==2.8.0 vllm==0.11.0 torch_npu==2.8.0.post5
pip install --no-deps vllm-ascend==0.11.0
pip install transformers==4.57.4
# 3) apply the current shim (v6): backs up mla_v1.py, prepends the shim block, replaces both npu_ring_mla call-sites
#    (the two operations ran in sessions 5g/5h; the wrapper itself was tested on a mock target only)
python suite/shim/apply_shim.py
# legacy: suite/shim/giga31_shim.patch is the original 71-line shim v1 — single-request only, needs --no-enable-prefix-caching
# 4) serve — single-request / low concurrency (prefix caching stays ON)
vllm serve /tmp/models/giga31_w8a8 --port 8000 --max-model-len 4096 \
  --max-num-batched-tokens 4096 --enforce-eager --quantization ascend
# 4b) serve — concurrent traffic (batched context merge not fixed yet)
vllm serve /tmp/models/giga31_w8a8 --port 8000 --max-model-len 4096 \
  --max-num-batched-tokens 4096 --enforce-eager --quantization ascend --no-enable-prefix-caching
```

Checkpoint integrity: tar sha256 `f5551738e672010b861077540f39ba8b9cae144601679a3c2530322ae827a28b`, 42×256MB chunks each verified against LFS oids.

## Repository layout

| Path | Contents |
|---|---|
| `suite/shim/` | shim blocks (v1 patch, v4, v6), `apply_shim.py`, probes 5g/5h, status README |
| `openi-suite/` | session scripts g0–g13b, LFS manifest + `RECOVER.md` |
| `evidence/act5d` … `act5h` | raw evidence per session: test JSONs, SHIM call dumps, mask stats, merge validation, concurrency runs + telemetry, npu-smi, pip freeze |
| `evidence/crash/` | consolidated record of the native-operator crash (stacks, head-count matrix) |
| `docs/ACT-INFERENCE.md` | полный акт инференса (RU) |
| `docs/ECONOMICS-CUSTOMER.md` | customer economics one-pager |

## Stack pins

torch 2.8.0 · torch_npu 2.8.0.post5 · vllm 0.11.0 · vllm-ascend 0.11.0 (`--no-deps`) · transformers 4.57.4 · CANN 8.5.0 · Ascend 910B 64GB · eager mode, single card

## Limitations

- eager mode (graph mode pending), decode ~10 tps single stream (window-averaged 8–9.3 t/s in evidence serve logs; dedicated benchmark pending)
- prefix caching compatible since shim v4 (2026-11-23): cache-hit answers correct — semantically equal to cold, numerically distinct ~0.007 (two bf16 routes); see `evidence/act5g/`
- concurrency: with prefix caching ON the batched context merge still corrupts answers (13/40 in an 8-way burst); for concurrent serving use `--no-enable-prefix-caching` (38/40, shim v6); see `evidence/act5h/`
- the shim is verified on vllm-ascend 0.11.0 only; 0.13.0 has identical call-sites but is untested
- single card, max-model-len 4096
- quantization ran with `--disable_anti` (see homonym note)

## License

MIT — original model © Sber; W8A8 quantization, patch and evidence © 2026 MutTechDev.

## Changelog

- 2026-11-23 — **shim v4: prefix caching made compatible** (ring-merge contract restored: fp32 LSE from FIA + log-sum-exp merge on cached-context calls; merge vs single-shot diff 0.0071 = bf16 noise, LSE diff 0.0000). 20/20 with prefix caching ON, engine cache hit rate 89%. Exact live FIA dims: Q=K=V=192 (v_head_dim=192). Evidence: evidence/act5g/ + suite/shim/shim_block_v4.py.
- 2026-11-23 — network note for contributors in CN: direct TLS to github.com may fail from some CN networks (`Empty reply from server`); with a SOCKS5 endpoint use `git -c http.proxy=socks5h://HOST:PORT clone/push …`. OpenI (openi.pcl.ac.cn) is reachable directly from CN.
- 2026-11-23 — **shim v6: batched prefill smoke** (fresh-varlen TND `sparse_mode=3` + per-request context loop with fp32 log-sum-exp merge): 8 parallel requests with different prompts — **38/40** with `--no-enable-prefix-caching`; with caching ON the context merge still corrupts under batching (13/40). Evidence: `evidence/act5h/` + `suite/shim/shim_block_v6.py`, `probe5h2–6.py`.
- 2026-11-24 — docs consistency pass: front page updated to the shim v6 state (results, prefix caching, two serve commands, quickstart installs the current shim via `suite/shim/apply_shim.py`), `evidence/crash/README.md` added (consolidated record of the native-operator crash — reproduced live on 0.11.0 and 0.13.0; raw tracebacks were not captured), `docs/ACT-INFERENCE.md` synced with the 5g/5h addenda.

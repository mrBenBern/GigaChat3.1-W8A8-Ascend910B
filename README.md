# GigaChat3.1-10B-A1.8B — W8A8 inference on Ascend 910B

**First publicly documented W8A8 inference of GigaChat3.1-10B-A1.8B (MoE, MIT) on Huawei Ascend 910B** — an attention shim patch for vllm-ascend, a reproducible pipeline, raw evidence, and a full inference act.

> Hugging Face model: [Bentuzzi/GigaChat3.1-10B-A1.8B-W8A8-Ascend910B](https://huggingface.co/Bentuzzi/GigaChat3.1-10B-A1.8B-W8A8-Ascend910B) · Mirror: [OpenI MutTechDev/onion-openi](https://openi.pcl.ac.cn/MutTechDev/onion-openi)
> Based on [ai-sage/GigaChat3.1-10B-A1.8B-bf16](https://modelscope.cn/models/ai-sage/GigaChat3.1-10B-A1.8B-bf16) (MIT, © Sber)

**RU TL;DR:** первый публично документированный запуск GigaChat3.1-10B-A1.8B (W8A8, 11 ГБ) на Ascend 910B через открытый стек vLLM + vllm-ascend. Крах родного MLA-оператора обойдён патчем на 71 строку, качество закрыто 20/20 контрольными тестами, детерминизм байт-в-байт, полный акт инференса и все сырые evidence — в этом репозитории.

## Results (2026-11-22, OpenI task 942047)

| Metric | Value |
|---|---|
| Semantic tests | **20/20 PASS** (10 sanity · 5 instruction-following · 5 context-invariants) |
| Determinism | byte-identical on repeats (seed=0) |
| TTFT | 0.2s @1k tokens · 0.5s @1.5k tokens (full prefill through shim) |
| Full QA round-trip | 0.3–1.5s |
| Attention operator | verified vs CPU-fp32 reference: max_diff 0.0077 (bf16 noise), causality perfect |
| Mask | formally verified: sum/rows/`equal_ref=True` against causal reference |
| NPU evidence | npu-smi info + dmon captured during generation |

## The problem

Native vllm-ascend MLA prefill operator crashes on this model: `AtbRingMLAGetWorkspaceSize` fails on the first real prefill (1049 tokens). Reproduced across **vllm-ascend 0.11.0 / 0.13.0 × torch_npu 2.7.1 / 2.8.0.post5 × CANN 8.3 / 8.5**. GigaChat3.1 uses non-standard MLA dims (qk_nope 128 + qk_rope 64 = 192, **v_head_dim 192**) — Q=K=V=192 is legal per CANN documentation, but the ring operator doesn't accept it.

## The fix — a 71-line patch

`openi-suite` → `suite/shim/giga31_shim.patch` patches the installed `vllm_ascend/attention/mla_v1.py`:

1. `_giga_shim(**kw)` replaces both `npu_ring_mla` call-sites: decompress `cat(q_nope,q_rope)→[T,32,192]`, `cat(k_nope,k_rope)→[S,32,192]` → `npu_fused_infer_attention_score` (BSND `[1,T,H,192]` **without transpose**, `num_heads`/`num_key_value_heads` kwargs, `scale=qk_scale`, bool causal mask, `sparse_mode=0`)
2. Ring mask is only **512×512** → rebuild causal bool mask at **T×S** when 512 < T
3. `_v_up_proj` `_C_ascend` fast-path disabled (PyPI wheel has no binary) → `torch.bmm`

Decode runs the native absorb path — mathematically equivalent (linear W_UV commutes with the attention-weighted sum).

## Two root causes we hit on the way (so you don't have to)

1. **Prefix caching breaks the shim**: repeated requests sharing the ~1034-token role_description prefix read cached KV blocks incompatible with the decompressed-prefill path → coherent-looking but off-topic output. Serve with **`--no-enable-prefix-caching`**.
2. **Bare homonym acronyms** (e.g. "MLA") may sample wrong expansions at higher temperatures — a W8A8-without-anti-outlier nuance. Use full terms ("Multi-head Latent Attention (MLA)") or re-quantize with anti-outlier once tooling supports non-standard MLA dims.

## Quickstart

```bash
# 1) restore checkpoint (42 LFS chunks, per-chunk sha256) — see openi-suite/RECOVER.md
# 2) stack
pip install torch==2.8.0 vllm==0.11.0 torch_npu==2.8.0.post5
pip install --no-deps vllm-ascend==0.11.0
pip install transformers==4.57.4
# 3) patch
cp vllm_ascend/attention/mla_v1.py vllm_ascend/attention/mla_v1.py.bak
patch vllm_ascend/attention/mla_v1.py < suite/shim/giga31_shim.patch
# 4) serve
vllm serve /tmp/models/giga31_w8a8 --port 8000 --max-model-len 4096 \
  --max-num-batched-tokens 4096 --enforce-eager --quantization ascend \
  --no-enable-prefix-caching
```

Checkpoint integrity: tar sha256 `f5551738e672010b861077540f39ba8b9cae144601679a3c2530322ae827a28b`, 42×256MB chunks each verified against LFS oids.

## Repository layout

| Path | Contents |
|---|---|
| `suite/shim/` | the patch + how-to + status README |
| `openi-suite/` | session scripts g0–g13b, LFS manifest + `RECOVER.md` |
| `evidence/5c`, `evidence/act5d`, `evidence/act5e` | raw evidence: smoke JSONs, 40 verbose SHIM call dumps, decode call-site, npu-smi, pip freeze, mask stats, homonym matrix |
| `docs/ACT-INFERENCE.md` | полный акт инференса (RU) |
| `docs/ECONOMICS-CUSTOMER.md` | customer economics one-pager |

## Stack pins

torch 2.8.0 · torch_npu 2.8.0.post5 · vllm 0.11.0 · vllm-ascend 0.11.0 (`--no-deps`) · transformers 4.57.4 · CANN 8.5.0 · Ascend 910B 64GB · eager mode, single card

## Limitations

- eager mode (graph mode pending), decode ~10–14 tps single stream
- prefix caching must stay off (see above)
- single card, max-model-len 4096
- quantization ran with `--disable_anti` (see homonym note)

## License

MIT — original model © Sber; W8A8 quantization, patch and evidence © 2026 MutTechDev.
# Native `npu_ring_mla` crash — consolidated record

Model: GigaChat3.1-10B-A1.8B, W8A8 checkpoint (MoE; MLA with **32 attention heads / 32 KV heads**, qk_nope 128 + qk_rope 64 = 192, **v_head_dim 192**, kv_lora_rank 512).
Hardware: 1× Ascend 910B 64GB, eager mode.

## Symptom (verbatim)

`torch_npu.atb.npu_ring_mla(...)` fails with **`AtbRingMLAGetWorkspaceSize failed`** (call-site `vllm_ascend/attention/mla_v1.py:1057` on vllm-ascend 0.11.0).

- On 0.11.0 the checkpoint loads, `/v1/models` answers and `profile_run` passes; the first **real** prefill dies (1049 and 1066 tokens in two sessions).
- On 0.13.0 the same crash already fires inside `profile_run`.
- Decode-only paths are not affected.
- ATB reports no internal reason, even with `ATB_LOG_LEVEL=DEBUG` + `ASCEND_GLOBAL_LOG_LEVEL=2` (`OpParamMaker.cpp:454` without details); the `ASCEND_LAUNCH_BLOCKING=1` log was unusable (polluted by nanobind noise from the dying process).

## Where it was reproduced (lab records)

| vllm-ascend | vLLM | torch_npu | CANN | result | session |
|---|---|---|---|---|---|
| 0.13.0 | 0.13.0 | 2.8.0.post2 | 8.5.0 (NNAL 8.3.RC1 → 8.5.0) | crash inside `profile_run`; the NNAL upgrade did not help | 2026-11-20, OpenI task 939529 |
| 0.11.0 | 0.11.0 | 2.7.1 | 8.3 env | crash, same signature | 2026-11-21, OpenI task 940514 |
| 0.11.0 | 0.11.0 | 2.8.0.post5 | 8.5.0 | crash on the first real prefill (1066 / 1049 tokens) | 2026-11-21 task 940514; 2026-11-22 task 941568 |

## Head-configuration matrix (synthetic harness, 2026-11-21)

| num_heads | v_head_dim | result |
|---|---|---|
| 128 | 192 | real arithmetic (the operator executes) |
| 128 | 128 (canonical DeepSeek-V3) | workspace FAIL in our harness — DeepSeek itself runs on vllm-ascend in production, so this row reflects the harness, not the operator |
| 32 / 64 | both tested values | silently returns zeros |

Reading: the ring operator effectively supports only **128 heads**; the documented head set {32, 64, 128} applies to the FIA path, not to RingMLA. Q=K=V=192 (this model's live triple) is legal per the CANN FIA documentation.

## What is not archived

Raw tracebacks of the three crash runs were **not captured**: the containers were ephemeral and the session records (journal entries above) preserve the error line, the call-site and the stack pins only. This file is the consolidated record. The workaround shim and all of its evidence (mask stats, FIA probes, merge validation, 20-test results, concurrency runs) are in `evidence/5c`, `evidence/act5d` … `evidence/act5h` and `suite/shim/`.

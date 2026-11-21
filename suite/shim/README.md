# GigaChat3.1 W8A8 on Ascend 910B — FIA shim

AtbRingMLAGetWorkspaceSize crashes on real prefill (0.11.0/0.13.0, CANN 8.3/8.5).
Fix = direct patch of installed vllm_ascend/attention/mla_v1.py:
1. _giga_shim(**kw): decompress cat(q_nope,q_rope)/cat(k_nope,k_rope) -> FIA BSND [1,T,H,192], num_heads/num_key_value_heads kwargs, scale=qk_scale
2. mask rebuilt T x S triu-bool when ring mask (512) < T
3. _v_up_proj _C_ascend fast-path disabled (PyPI build has no binary) -> torch.bmm

Apply: cp vllm_ascend/attention/mla_v1.py mla_v1.py.bak && patch mla_v1.py < giga31_shim.patch
Serve: vllm serve /tmp/models/giga31_w8a8 --port 8000 --max-model-len 4096 --max-num-batched-tokens 4096 --enforce-eager --quantization ascend
Evidence: first live tokens 2026-11-22 04:50 UTC+8, evidence/5c/

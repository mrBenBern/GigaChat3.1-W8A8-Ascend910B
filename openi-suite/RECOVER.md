# RECOVER.md — восстановление W8A8-чекпойнта GigaChat3.1 из LFS

**Артефакт**: первый в мире W8A8-чекпойнт GigaChat3.1-10B-A1.8B для Ascend 910B (формат ModelSlim/vllm-ascend: quant_model_description.json + w8a8_dynamic safetensors ×3 + index).
**Тар**: `giga31_w8a8.tar` = 11,294,667,776 байт, **SHA256 = f5551738e672010b861077540f39ba8b9cae144601679a3c2530322ae827a28b**.
**Хранение**: 42 LFS-объекта репо MutTechDev/onion-openi (манифест: `lfs-manifest-giga31w8a8.json`, порядок = порядок в списке).

## Восстановление (нужен токен с доступом к репо)

1. Для каждой записи манифеста по порядку:
   `POST /{repo}.git/info/lfs/objects/batch` (Basic auth, Content-Type application/vnd.git-lfs+json):
   `{"operation":"download","transfers":["basic"],"objects":[{"oid":OID,"size":SIZE}],"hash_algo":"sha256"}`
   → взять `objects[0].actions.download.href` (+ header.Authorization) → GET → файл.
   Проверить sha256 файла = oid.
2. Склеить по порядку: `cat giga31c_00 giga31c_01 … giga31c_41 > giga31_w8a8.tar`
3. Проверить sha256 склейки = `f5551738e672010b861077540f39ba8b9cae144601679a3c2530322ae827a28b`
4. `tar -xf giga31_w8a8.tar` → каталог `giga31_w8a8/` → `vllm serve ./giga31_w8a8 --quantization ascend` (стек: vllm 0.11.0 + vllm-ascend 0.11.0 + torch 2.8.0 + torch_npu 2.8.0.post5 + CANN 8.5.0; chat_template.jinja и tokenizer-файлы положить рядом из bf16-репо).

## Статус проверки (21–22.11)
- Загрузка 42/42 (LFS batch+PUT+verify), верифицирован с хоста вне контейнера: chunk_41 скачан, 244,393,984 байт, SHA256 совпал.
- Известное: PUT больших тел (≥4ГБ) рвётся TLS-ом платформы — поэтому чанки 256 МБ.

## Ограничения (честно)
- Serve на 910B через vllm-ascend 0.11/0.13 блокирован ограничением ATB-оператора npu_ring_mla (только 128 голов; у GigaChat 32) — см. atb-mla-вердикт. Инференс-фиксы: FIA-патч prefill / MindIE / апстрим.
- Качество не валидировано: anti-outlier отключён, калибровка синтетическая RU 384 сэмпла, quant_mtp=none.

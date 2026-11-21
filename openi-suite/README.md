# openi-suite — скелет заезда OpenI 910B (GigaChat W8A8 → serve)

_Инструмент > воспоминание: весь порядок заезда — здесь, в репо. Следующий контейнер клонирует этот репо и выполняет шаги по порядку._

## Порядок заезда (проверен 21.11, задача 940514)

| Шаг | Скрипт | Что делает | Время |
|---|---|---|---|
| 0 | `g0_setup.py` | env-факты, `/tmp/lab_env.sh` (CANN 8.5.0 + nnal), `pip install modelscope`, config.json ДО скачивания (dims!), bg-скачивание GigaChat3.1 bf16 (ModelScope ~8 МБ/с, 23 ГБ ≈ 45 мин), проверка CI-репо | 1 мин + фон |
| 1 | `g1c_pip2.py` | **ПОСЛЕДОВАТЕЛЬНЫЙ** pip-каскад: torch==2.8.0 → torch_npu==2.8.0.post5 → vllm==0.11.0 → vllm-ascend==0.11.0 (один вызов = ResolutionImpossible!) | 6–8 мин |
| 2 | `g3_slim_calib.py` | клон msit (ветка TR5) + touch __init__.py ×4 + RU-калибровка 384+64 + install.sh bg | ~4 мин |
| 3 | `g4_fixes.py` | compat-guard set_initialized_submodules + pad_token `</s>` в tokenizer_config модели | 10 с |
| 4 | `g5_smoke.py` | смоук кванта layer_count 1 (~2 мин; валиден только для пайплайна, serve на нём не поднять) | 2 мин |
| 5 | `g6_fullquant.py` | полный W8A8: 26 слоёв, `--disable_anti` (anti-outlier несовместим с dims GigaChat!), `--quant_mtp none`; 40 мин на .post5, 109 мин на 2.7.1 | 40–110 мин |
| 6 | `g8_serve.py` | копия chat_template.jinja + tokenizer файлов в каталог кванта, serve 0.11.0 (`--enforce-eager --quantization ascend`) | 3 мин |
| 7 | `g10b_bench2.py` | RU-смоук + замеры tok/s | 5 мин |

## ГРАБЛИ (все кровью, 19–21.11)

0. **МИГРАЦИЯ ПЛАТФОРМЫ (21.11)**: старая вкладка Cloudbrain помечена «Offline soon», новая кнопка «Create Computing Task» → `/cloudbrains/create` (новое поколение). Старые ai_task (наш API) прибиваются при миграции — задокументированы случайные STOPPED на 38 мин и на 2:52 (задачи 941263/941038). Митигация: короткая цепочка (~1ч45), LFS-части сохраняются независимо; при обрыве — новый таск, докачка недостающих LFS-частей.

1. **pip строго каскадом** (шаг 1) — единый вызов даёт ResolutionImpossible.
2. **jupyter_exec: WS-канал умирает на вызовах > 3–4 мин** — слипы в коде ≤ 90 с при `sh(t=150)`; долгие ожидания — повторные короткие poll-вызовы; nohup до разрыва выживает.
3. **pkill -f с паттерном из собственного cmdline = self-kill** — паттерн должен не совпадать с командой, или без pkill.
4. **sh timeout > sleep внутри** (sleep 115 при t=110 рвал вызов).
5. База Jupyter **без** суффикса `?token=`; сессия `onion-lab` переиспользуется.
6. create задачи **только с `--no-stop`**; delete после stop (STOPPING держит слот ~1 мин).
7. env CANN не персистентен — `source /tmp/lab_env.sh` в каждой sh-команде.
8. Скачивания: ModelScope первичен (hf-mirror душит 429/503); config.json читать ДО (v_head 192, q_lora null — отличия от канона V3).
9. Токен OpenI: из сейфа хоста, в контейнер — инъекцией в момент запуска (временный файл без BOM), **не печатать**.
10. Известный крах serve: первый реальный prefill → `AtbRingMLAGetWorkspaceSize failed` (см. `atb-mla-вердикт` — репро-матрица: оператор держит только 128 голов). Профиль: загрузка+profile_run+`/v1/models` проходят.

## Артефакты/выгрузка
- Чекпойнт: `/tmp/models/giga31_w8a8` (11 ГБ, формат ModelSlim: description.json + w8a8_dynamic safetensors ×3 + index).
- **Канал №1 (проверен живьём 21.11): git-LFS batch API** — токен-Basic, без сессий: `POST /{repo}.git/info/lfs/objects/batch` (Content-Type: application/vnd.git-lfs+json, {operation,objects:[{oid:sha256,size}]}) → PUT по href → verify. Round-trip подтверждён. Скрипт: `g12_lfs_save.py` (tar → чанки 4 ГБ → batch/PUT/verify). ОСТОРОЖНО: в PowerShell тело LFS передавать только через файл (-d @file) — кавычки режутся!
- Канал №2 (проверен 21.11): **репо-датасет + v2-upload**. Создание: `POST /api/v1/datasets/{o}/{r}/create` (токен, JSON {title,description,category:"其他",task:"其他"} — Category/Task обязательны; новый-style `POST /api/v1/dataset/create` на токене отвечает uid:0 — платформенный баг, создавать им нельзя). Загрузка: v2-слой SDK — `from openi.uploader import upload_file; upload_file(file='x.zip', repo_id='MutTechDev/onion-openi')` — файл ОБЯЗАН быть zip; чанки 4 ГБ паковать zip-ом каждый. Логин в контейнере: `openi login --token …` (инъекция из сейфа).
- Канал №3: TRAIN-задача output (персистентен, у DEBUG — нет) — не тестирован.
- Логи serve (`/tmp/vllm_w8a8.log`) — коммитить в репо при крахе (материал для issue).

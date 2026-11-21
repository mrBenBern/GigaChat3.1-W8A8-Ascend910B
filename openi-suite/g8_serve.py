# -*- coding: utf-8 -*-
# G8: serve W8A8 Э0 — chat template copy + vllm serve (night-1 combo) + RU smoke
import subprocess, json

def sh(cmd, n=2000, t=110):
    print('$ ' + cmd[:110])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t, errors='replace')
    out = (r.stdout or '') + (('\n[ERR] ' + r.stderr[-600:]) if r.stderr.strip() else '')
    print(out[:n])
    print('-' * 50)

sh('ls /tmp/models/giga31_w8a8/ | head -8; cp -f /tmp/models/giga31/chat_template.jinja /tmp/models/giga31_w8a8/ 2>/dev/null && echo TEMPLATE_COPIED; cp -f /tmp/models/giga31/tokenizer_config.json /tmp/models/giga31_w8a8/ && cp -f /tmp/models/giga31/tokenizer.json /tmp/models/giga31_w8a8/ 2>/dev/null; echo TOK_COPIED')

sh('pkill -f "bin/vllm serve"; sleep 3; echo cleaned')
sh('nohup bash -c "source /tmp/lab_env.sh && vllm serve /tmp/models/giga31_w8a8 --port 8000 --max-model-len 4096 --enforce-eager --quantization ascend" > /tmp/vllm_w8a8.log 2>&1 < /dev/null & echo SERVE_BG')
sh('sleep 115; tail -4 /tmp/vllm_w8a8.log | head -c 900; echo; curl -s -m 5 http://127.0.0.1:8000/v1/models | head -c 120')

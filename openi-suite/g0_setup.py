# -*- coding: utf-8 -*-
# G0: env facts + lab_env.sh + bg download GigaChat bf16 (ModelScope) + config dims + CI repo check
import subprocess, json

def sh(cmd, n=1800, t=120):
    print('$ ' + cmd[:110])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t, errors='replace')
    out = (r.stdout or '') + (('\n[ERR] ' + r.stderr[-600:]) if r.stderr.strip() else '')
    print(out[:n])
    print('-' * 50)

sh('npu-smi info 2>&1 | head -8; nproc; free -g | head -2; df -h /tmp | tail -1')

sh('printf "export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/8.5.0.alpha001\\nsource /usr/local/Ascend/ascend-toolkit/set_env.sh\\n[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh\\n" > /tmp/lab_env.sh; cat /tmp/lab_env.sh')

sh('pip install -q modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -1; modelscope --version 2>&1 | head -1', 800, 300)

sh('mkdir -p /tmp/models/giga31 && wget -q -O /tmp/models/giga31/config.json "https://modelscope.cn/models/ai-sage/GigaChat3.1-10B-A1.8B-bf16/resolve/master/config.json" && python3 -c "import json;c=json.load(open(\'/tmp/models/giga31/config.json\'));a={k:v for k,v in c.items() if any(s in k.lower() for s in [\'head\',\'lora\',\'rope\',\'hidden\',\'dim\',\'experts\',\'moe\',\'mtp\',\'arch\'])};print(json.dumps(a,indent=1,ensure_ascii=False)[:1500])"')

sh('nohup bash -c "source /tmp/lab_env.sh && modelscope download --model ai-sage/GigaChat3.1-10B-A1.8B-bf16 --local_dir /tmp/models/giga31" > /tmp/dl_giga_ms.log 2>&1 < /dev/null & echo GIGA_MS_BG')
sh('sleep 30; du -sh /tmp/models/giga31 2>/dev/null; tail -1 /tmp/dl_giga_ms.log | head -c 200')

sh('curl -s -m 20 "https://hf-mirror.com/api/models/vllm-ascend/DeepSeek-V3-W8A8" | head -c 400; echo; curl -s -m 20 "https://hf-mirror.com/api/models?author=vllm-ascend" | python3 -c "import sys,json;d=json.load(sys.stdin);print([m[\'modelId\'] for m in d][:20])" 2>&1 | head -c 600')

# -*- coding: utf-8 -*-
# G1c: sequential pip cascade (night-1 order)
import subprocess

def sh(cmd, n=1200, t=110):
    print('$ ' + cmd[:110])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t, errors='replace')
    out = (r.stdout or '') + (('\n[ERR] ' + r.stderr[-500:]) if r.stderr.strip() else '')
    print(out[:n])
    print('-' * 50)

sh('nohup bash -c "source /tmp/lab_env.sh && pip install --no-cache-dir torch==2.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && pip install --no-cache-dir torch_npu==2.8.0.post5 -i https://pypi.tuna.tsinghua.edu.cn/simple && pip install --no-cache-dir vllm==0.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && pip install --no-cache-dir vllm-ascend==0.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && echo PIP_ALL_DONE" > /tmp/pip11b.log 2>&1 < /dev/null & echo PIP_CASCADE_BG')
sh('sleep 100; tail -2 /tmp/pip11b.log | head -c 400')

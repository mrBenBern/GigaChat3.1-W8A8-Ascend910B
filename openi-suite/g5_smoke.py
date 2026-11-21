# -*- coding: utf-8 -*-
# G5: smoke quant layer_count=1 (bg) + poll
import subprocess

def sh(cmd, n=1800, t=110):
    print('$ ' + cmd[:110])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t, errors='replace')
    out = (r.stdout or '') + (('\n[ERR] ' + r.stderr[-600:]) if r.stderr.strip() else '')
    print(out[:n])
    print('-' * 50)

sh('rm -rf /tmp/models/giga31_smoke; nohup bash -c "source /tmp/lab_env.sh && cd /tmp/msit/msmodelslim && PYTHONPATH=/tmp/msit/msmodelslim python3 example/DeepSeek/quant_deepseek_w8a8.py --model_path /tmp/models/giga31 --save_path /tmp/models/giga31_smoke --layer_count 1 --batch_size 4 --calib_dataset /tmp/calib_prompt.json --anti_dataset /tmp/anti_prompt.json --quant_mtp none --from_bf16 --disable_anti" > /tmp/quant_smoke.log 2>&1 < /dev/null & echo SMOKE_BG')
sh('sleep 95; tail -6 /tmp/quant_smoke.log | head -c 1400')

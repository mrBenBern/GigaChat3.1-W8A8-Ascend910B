# -*- coding: utf-8 -*-
# G6: FULL W8A8 quant bg (night-2 verbatim + disable_anti)
import subprocess

def sh(cmd, n=1600, t=110):
    print('$ ' + cmd[:110])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t, errors='replace')
    out = (r.stdout or '') + (('\n[ERR] ' + r.stderr[-500:]) if r.stderr.strip() else '')
    print(out[:n])
    print('-' * 50)

sh('rm -rf /tmp/models/giga31_w8a8; nohup bash -c "source /tmp/lab_env.sh && cd /tmp/msit/msmodelslim && PYTHONPATH=/tmp/msit/msmodelslim python3 example/DeepSeek/quant_deepseek_w8a8.py --model_path /tmp/models/giga31 --save_path /tmp/models/giga31_w8a8 --batch_size 4 --calib_dataset /tmp/calib_prompt.json --anti_dataset /tmp/anti_prompt.json --quant_mtp none --from_bf16 --disable_anti" > /tmp/quant_full.log 2>&1 < /dev/null & echo FULL_QUANT_BG')
sh('sleep 90; tail -3 /tmp/quant_full.log | head -c 700')

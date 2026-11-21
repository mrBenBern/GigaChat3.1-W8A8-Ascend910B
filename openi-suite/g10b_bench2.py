# -*- coding: utf-8 -*-
# G10v2: bench — all request bodies prepared as files, generous timeouts
import subprocess, json

M = '/tmp/models/giga31_w8a8'
prompts = [
    "Объясни в двух абзацах, чем HCCL отличается от NCCL и почему это важно при переезде с NVIDIA на Ascend.",
    "Напиши функцию на Python, которая проверяет палиндром, и два теста к ней.",
    "Составь 5-пунктовый план ввода китайского AI-сервера в эксплуатацию в российской компании.",
]
for i, p in enumerate(prompts):
    body = {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": 350, "temperature": 0.3, "top_p": 0.95}
    json.dump(body, open('/tmp/bq%d.json' % i, 'w'), ensure_ascii=False)

# smoke (cold start, generous)
r = subprocess.run('curl -s -m 220 -H "Content-Type: application/json" -d @/tmp/bq0.json -o /tmp/bresp0.json http://127.0.0.1:8000/v1/chat/completions && echo CURL_OK', shell=True, capture_output=True, text=True, timeout=240, errors='replace')
print('smoke curl:', r.stdout[-30:], r.stderr[:120])
try:
    j = json.load(open('/tmp/bresp0.json'))
    print('CONTENT:', j['choices'][0]['message']['content'][:420])
    print('USAGE:', j['usage'])
except Exception as e:
    print('smoke parse FAIL:', str(e)[:80], open('/tmp/bresp0.json').read()[:300])

# sequential P2, P3 with timing
for i in (1, 2):
    r = subprocess.run('T0=$(date +%s.%N); curl -s -m 200 -H "Content-Type: application/json" -d @/tmp/bq%d.json -o /tmp/bresp%d.json http://127.0.0.1:8000/v1/chat/completions; T1=$(date +%s.%N); echo "WALL=$(echo "$T1 - $T0" | bc)"' % (i, i), shell=True, capture_output=True, text=True, timeout=230, errors='replace')
    wall = None
    for line in r.stdout.splitlines():
        if line.startswith('WALL='):
            wall = float(line.split('=')[1])
    try:
        j = json.load(open('/tmp/bresp%d.json' % i))
        ct = j['usage']['completion_tokens']
        print('P%d | wall %.1fs | %d tok | %.1f tok/s' % (i + 1, wall or 0, ct, ct / wall if wall else 0))
        print(j['choices'][0]['message']['content'][:300])
    except Exception as e:
        print('P%d FAIL:' % (i + 1), str(e)[:70], open('/tmp/bresp%d.json' % i).read()[:200])
    print('-' * 40)

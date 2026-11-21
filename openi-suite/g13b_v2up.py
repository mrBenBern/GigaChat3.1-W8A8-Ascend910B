# -*- coding: utf-8 -*-
# G13b: v2-compat upload_file test (zip)
import subprocess

def sh(cmd, n=1600, t=150):
    print('$ ' + cmd[:110])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t, errors='replace')
    out = (r.stdout or '') + (('\n[ERR] ' + r.stderr[-500:]) if r.stderr.strip() else '')
    print(out[:n])
    print('-' * 50)

code = r'''
import zipfile, functools
print = functools.partial(print, flush=True)
with zipfile.ZipFile('/tmp/dstest.zip', 'w') as z:
    z.writestr('onion-ds-test.txt', 'onion lab artifact channel test\n')
try:
    from openi.uploader import upload_file
    url = upload_file(file='/tmp/dstest.zip', repo_id='MutTechDev/onion-openi')
    print('UPLOAD_OK:', url)
except Exception as e:
    print('UPLOAD_FAIL:', str(e)[:300])
'''
open('/tmp/v2up.py', 'w').write(code)
sh('python3 /tmp/v2up.py 2>&1 | tail -6')

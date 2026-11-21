# -*- coding: utf-8 -*-
# G12: LFS save checkpoint (tar -> 4GB chunks -> LFS batch upload -> verify). Token injected at exec time.
import subprocess

TOKEN = '__TOKEN__'

def sh(cmd, n=1600, t=150):
    print('$ ' + cmd[:110])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t, errors='replace')
    out = (r.stdout or '') + (('\n[ERR] ' + r.stderr[-400:]) if r.stderr.strip() else '')
    print(out[:n])
    print('-' * 50)

import functools
print = functools.partial(print, flush=True)

sh('grep -c "Save successfully" /tmp/quant_full.log; du -sh /tmp/models/giga31_w8a8 2>/dev/null')

sh('nohup bash -c "tar -cf /tmp/giga31_w8a8.tar -C /tmp/models giga31_w8a8 && split -b 4G -d /tmp/giga31_w8a8.tar /tmp/giga31_w8a8.tar.part && echo TAR_SPLIT_DONE" > /tmp/tar.log 2>&1 < /dev/null & echo TAR_BG')
sh('sleep 110; grep -c TAR_SPLIT_DONE /tmp/tar.log; ls /tmp/giga31_w8a8.tar.part* 2>/dev/null | head -5')

code = r'''
import hashlib, json, glob, urllib.request, base64, functools
print = functools.partial(print, flush=True)
TOKEN = 'LFS_TOKEN_PLACEHOLDER'
REPO = 'MutTechDev/onion-openi'
def basic():
    return 'Basic ' + base64.b64encode(f'MutTechDev:{TOKEN}'.encode()).decode()
def lfs_batch(op, oid, size):
    body = json.dumps({'operation': op, 'transfers': ['basic'], 'objects': [{'oid': oid, 'size': size}], 'hash_algo': 'sha256'}).encode()
    req = urllib.request.Request(f'https://openi.pcl.ac.cn/{REPO}.git/info/lfs/objects/batch', data=body, method='POST',
                                 headers={'Authorization': basic(), 'Content-Type': 'application/vnd.git-lfs+json', 'Accept': 'application/vnd.git-lfs+json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())
parts = sorted(glob.glob('/tmp/giga31_w8a8.tar.part*'))
print('parts:', len(parts))
for p in parts:
    h = hashlib.sha256(); n = 0
    with open(p, 'rb') as f:
        while True:
            b = f.read(1 << 22)
            if not b: break
            h.update(b); n += len(b)
    oid = h.hexdigest()
    obj = lfs_batch('upload', oid, n)['objects'][0]
    if 'actions' in obj and 'upload' in obj.get('actions', {}):
        href = obj['actions']['upload']['href']
        print(p.split('/')[-1], oid[:12], n, 'uploading...')
        req = urllib.request.Request(href, data=open(p, 'rb').read(), method='PUT',
                                     headers={'Content-Type': 'application/octet-stream', 'Authorization': basic()})
        with urllib.request.urlopen(req, timeout=3600) as r:
            print('PUT', r.status)
        vhref = obj['actions'].get('verify', {}).get('href')
        if vhref:
            vreq = urllib.request.Request(vhref, data=json.dumps({'oid': oid, 'size': n}).encode(), method='POST',
                                          headers={'Content-Type': 'application/vnd.git-lfs+json', 'Authorization': basic()})
            with urllib.request.urlopen(vreq, timeout=60) as r:
                print('VERIFY', r.status)
    elif obj.get('uploaded'):
        print(p.split('/')[-1], oid[:12], 'already uploaded')
    else:
        print(p.split('/')[-1], 'UNEXPECTED:', str(obj)[:200])
print('LFS_UPLOAD_DONE')
'''
code = code.replace('LFS_TOKEN_PLACEHOLDER', TOKEN)
open('/tmp/lfs_up.py', 'w').write(code)
print('lfs_up.py written, token embedded:', 'LFS_TOKEN_PLACEHOLDER' not in code)

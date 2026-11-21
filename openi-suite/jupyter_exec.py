# -*- coding: utf-8 -*-
"""jupyter_exec.py — исполняет python-код в kernel отладочной задачи OpenI (ModelArts JupyterLab).
Использование: python jupyter_exec.py <base_url> <code_file.py> [timeout_sec]
base_url: https://host:port/notebook_..._task0  (token= можно опустить — прокси пустой токен пропускает)
Печатает stdout/stderr ячеек. Ядро переиспользуется между вызовами (поиск сессии 'onion-lab').
"""
import json, ssl, sys, time, uuid, urllib.request, urllib.error, urllib.parse, http.cookiejar, io
import websocket

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
BASE = sys.argv[1].rstrip('/')
CODE_FILE = sys.argv[2]
TIMEOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 300
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
JAR = http.cookiejar.CookieJar()
OP = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx), urllib.request.HTTPCookieProcessor(JAR))
OP.open(urllib.request.Request(BASE + '/lab'), timeout=60).read()

def xsrf():
    for c in JAR:
        if c.name in ('_xsrf', 'XSRF-TOKEN'):
            return c.value
    return ''

def api(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    hdr = {'Content-Type': 'application/json', 'X-XSRFToken': xsrf(), 'X-CSRFToken': xsrf()}
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=hdr)
    with OP.open(r, timeout=60) as resp:
        return json.loads(resp.read().decode('utf-8', 'replace'))

SESSION_NAME = 'onion-lab'
sessions = api('GET', '/api/sessions')
sessions = sessions.get('sessions') if isinstance(sessions, dict) else sessions
sessions = sessions or []
sess = next((s for s in sessions if s.get('name') == SESSION_NAME), None)
if not sess:
    sess = api('POST', '/api/sessions', {'kernel': {'name': 'python3'}, 'name': SESSION_NAME,
                                         'path': 'onion-lab.ipynb', 'type': 'notebook'})
kid = sess['kernel']['id']
sid = sess['id']
print('kernel:', kid)

code = open(CODE_FILE, encoding='utf-8').read()
ws_url = BASE.replace('https://', 'wss://') + '/api/kernels/%s/channels?session_id=%s' % (kid, sid)
ws = websocket.create_connection(ws_url, timeout=TIMEOUT, sslopt={'cert_reqs': ssl.CERT_NONE},
                                 origin=BASE, header=['Cookie: ' + '; '.join('%s=%s' % (c.name, c.value) for c in JAR)])
msg_id = uuid.uuid4().hex
req = {'header': {'msg_id': msg_id, 'msg_type': 'execute_request', 'username': 'onion',
                  'session': sid, 'version': '5.3', 'date': time.strftime('%Y-%m-%dT%H:%M:%SZ')},
       'parent_header': {}, 'metadata': {}, 'content': {'code': code, 'silent': False,
       'store_history': True, 'user_expressions': {}, 'allow_stdin': False}, 'buffers': None}
ws.send(json.dumps(req))

deadline = time.time() + TIMEOUT
buf = []
while time.time() < deadline:
    try:
        raw = ws.recv()
    except websocket.WebSocketTimeoutException:
        print('!! recv timeout'); break
    m = json.loads(raw)
    p = m.get('parent_header') or {}
    if p.get('msg_id') != msg_id:
        continue
    t = m.get('header', {}).get('msg_type')
    c = m.get('content') or {}
    if t == 'stream':
        buf.append(c.get('text', ''))
    elif t == 'execute_result':
        buf.append(c.get('data', {}).get('text/plain', ''))
    elif t == 'error':
        buf.append('\n'.join(c.get('traceback', [])))
    elif t == 'status' and c.get('execution_state') == 'idle':
        break
ws.close()
out = ''.join(buf)
print(out if out else '(no output)')

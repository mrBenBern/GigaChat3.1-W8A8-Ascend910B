# -*- coding: utf-8 -*-
# G4v2: quant script fixes (atomic) + tokenizer pad_token + slim install poll
import subprocess

def sh(cmd, n=1600, t=110):
    print('$ ' + cmd[:110])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t, errors='replace')
    out = (r.stdout or '') + (('\n[ERR] ' + r.stderr[-500:]) if r.stderr.strip() else '')
    print(out[:n])
    print('-' * 50)

code = r'''
import re
p = '/tmp/msit/msmodelslim/example/DeepSeek/quant_deepseek_w8a8.py'
src = open(p, encoding='utf-8').read()
changed = []
needle = "patch(\"transformers.modeling_utils.set_initialized_submodules\", new=set_initialized_submodules).start()"
FIX = ("import transformers.modeling_utils as _tmu\n"
       "if hasattr(_tmu, 'set_initialized_submodules'):\n"
       "    patch(\"transformers.modeling_utils.set_initialized_submodules\", new=set_initialized_submodules).start()\n"
       "else:\n"
       "    print('[compat] set_initialized_submodules absent; skipping patch')")
if needle in src and '[compat]' not in src:
    src = src.replace(needle, FIX)
    changed.append('compat-guard')
elif '[compat]' in src:
    changed.append('compat-guard already')
else:
    changed.append('compat-guard NEEDLE NOT FOUND')
if 'PAD_TOKEN_FIX' not in src:
    m = list(re.finditer(r'(tokenizer\s*=\s*AutoTokenizer\.from_pretrained\([^\n]*\))', src))
    if m:
        last = m[-1]
        repl = last.group(1) + '\n    if tokenizer.pad_token is None:\n        tokenizer.pad_token = tokenizer.eos_token  # PAD_TOKEN_FIX'
        src = src[:last.start()] + repl + src[last.end():]
        changed.append('pad-fallback')
    else:
        changed.append('pad-fallback site NOT FOUND')
else:
    changed.append('pad-fallback already')
open(p, 'w', encoding='utf-8').write(src)
print('FIXES:', changed)
'''
open('/tmp/fix_quant.py', 'w').write(code)
sh('python3 /tmp/fix_quant.py; grep -c "PAD_TOKEN_FIX" /tmp/msit/msmodelslim/example/DeepSeek/quant_deepseek_w8a8.py; grep -c "\\[compat\\]" /tmp/msit/msmodelslim/example/DeepSeek/quant_deepseek_w8a8.py')

sh('''python3 -c "
import json
tc_p = '/tmp/models/giga31/tokenizer_config.json'
tc = json.load(open(tc_p))
tc['pad_token'] = '</s>'
json.dump(tc, open(tc_p, 'w'), ensure_ascii=False, indent=1)
print('pad_token set to', tc['pad_token'])
"''')

sh('grep -c SLIM_INSTALL_DONE /tmp/slim_install.log 2>/dev/null; tail -2 /tmp/slim_install.log | head -c 400')

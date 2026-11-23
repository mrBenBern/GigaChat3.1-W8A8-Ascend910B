#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Install the GigaChat3.1 FIA shim into an installed vllm-ascend (block v6 by default).

Two operations are combined here; each of them was executed verbatim in our runs:
  * fresh install (session 5g, patch5g.py): prepend the shim block to
    vllm_ascend/attention/mla_v1.py, replace both `torch_npu.atb.npu_ring_mla(`
    call-sites with `_giga_shim(`, disable the `_C_ascend` fast-path that is
    absent from PyPI wheels;
  * block swap over an already installed shim (session 5h): replace the old
    block -- from `_GIGA_TND_FRESH` up to the shim's `    return out` -- with
    the new block.
This wrapper itself was tested on a mock target only (call-site count, syntax,
fresh == swap idempotency); it was not re-run on an NPU. The shim blocks were:
v4 -- evidence/act5g, v6 -- evidence/act5h.

Usage:
    python apply_shim.py                                   # v6, auto-detect mla_v1.py
    python apply_shim.py --file /path/mla_v1.py --dry-run
    python apply_shim.py --block shim_block_v4.py
"""
import argparse
import ast
import os
import re
import shutil
import sys

CALL = 'torch_npu.atb.npu_ring_mla('
CALL_NEW = '_giga_shim('
FASTPATH = 'and hasattr(torch.ops._C_ascend, "batch_matmul_transpose")'
RETURN_OUT = re.compile(r'^    return out\r?\n', re.M)


def detect_target():
    try:
        import importlib.util
        spec = importlib.util.find_spec('vllm_ascend.attention.mla_v1')
        return spec.origin if spec and spec.origin else None
    except Exception:
        return None


def build(src, block):
    """Return (new_source, mode) or exit with a message."""
    if 'def _giga_shim(' not in block:
        sys.exit('block does not define _giga_shim')
    if '_giga_shim' in src:
        # swap over an installed shim (as executed in session 5h)
        start = src.index('_GIGA_TND_FRESH') if '_GIGA_TND_FRESH' in src else src.index('def _giga_shim(')
        m = RETURN_OUT.search(src, start)
        if not m:
            sys.exit('end of the installed shim block not found ("    return out")')
        new, mode = block.rstrip('\n') + '\n' + src[m.end():], 'swap'
    else:
        n = src.count(CALL)
        if n != 2:
            sys.exit('expected 2 npu_ring_mla call-sites, found %d -- wrong vllm-ascend version?' % n)
        new = block.rstrip('\n') + '\n\n' + src.replace(CALL, CALL_NEW)
        if FASTPATH in src:
            new = new.replace(FASTPATH, 'and False')
        mode = 'fresh'
    if CALL in new:
        sys.exit('a npu_ring_mla call-site is left unpatched')
    return new, mode


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', default=os.path.join(here, 'shim_block_v6.py'))
    ap.add_argument('--file', dest='target', default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    target = a.target or detect_target()
    if not target or not os.path.isfile(target):
        sys.exit('mla_v1.py not found -- pass --file /path/to/vllm_ascend/attention/mla_v1.py')
    with open(target, encoding='utf-8') as f:
        src = f.read()
    with open(a.block, encoding='utf-8') as f:
        block = f.read()
    new, mode = build(src, block)
    ast.parse(new)
    print('%s mode=%s block=%s _giga_shim calls=%d target=%s + SYNTAX OK' % (
        'DRY-RUN' if a.dry_run else 'PATCHED', mode, os.path.basename(a.block),
        new.count(CALL_NEW), target))
    if not a.dry_run:
        bak = target + '.bak'
        if not os.path.exists(bak):
            shutil.copy2(target, bak)
        with open(target, 'w', encoding='utf-8') as f:
            f.write(new)


if __name__ == '__main__':
    main()

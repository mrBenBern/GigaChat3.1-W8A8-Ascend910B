
import ast
F = '/usr/local/python3.11.13/lib/python3.11/site-packages/vllm_ascend/attention/mla_v1.py'
src = open(F).read()
n = src.count('torch_npu.atb.npu_ring_mla(')
print('callsites:', n)
assert n == 2
shim = open('/tmp/shim_block_v4.py').read()
new = shim + '\n' + src.replace('torch_npu.atb.npu_ring_mla(', '_giga_shim(')
new = new.replace('and hasattr(torch.ops._C_ascend, "batch_matmul_transpose")', 'and False')
ast.parse(new)
open(F, 'w').write(new)
print('PATCHED5G + SYNTAX OK')

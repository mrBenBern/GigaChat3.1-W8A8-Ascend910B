
import torch, torch_npu, functools
print = functools.partial(print, flush=True)
torch.npu.set_device(0)
num_blocks, block_size, H, D = 6, 2, 1, 512
cache_kv_c = torch.zeros(num_blocks, block_size, H, D, dtype=torch.bfloat16, device='npu')
cache_k_pe = torch.zeros(num_blocks, block_size, H, 64, dtype=torch.bfloat16, device='npu')
for b in range(num_blocks):
    cache_kv_c[b] = float(b + 1)
# two requests: req0 -> blocks [0,1], req1 -> blocks [2,3]; full 2-row chunks
block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device='npu')
ctx_len = torch.tensor([2, 2], dtype=torch.int32, device='npu')
starts = torch.tensor([0, 0], dtype=torch.int32, device='npu')
out_k = torch.empty(4, H, D, dtype=torch.bfloat16, device='npu')
out_pe = torch.empty(4, H, 64, dtype=torch.bfloat16, device='npu')
torch_npu.atb.npu_paged_cache_load(cache_kv_c, cache_k_pe, block_table, ctx_len,
                                   seq_starts=starts, key=out_k, value=out_pe)
vals = out_k[:, 0, 0].float().tolist()
print('LOAD_ORDER rows->block_ids:', [int(v - 1) for v in vals], flush=True)
# case2: uneven ctx lens [1, 2], starts [1, 0] (req0 loads only its row1)
block_table2 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device='npu')
ctx_len2 = torch.tensor([1, 2], dtype=torch.int32, device='npu')
starts2 = torch.tensor([1, 0], dtype=torch.int32, device='npu')
out2 = torch.empty(3, H, D, dtype=torch.bfloat16, device='npu')
out2p = torch.empty(3, H, 64, dtype=torch.bfloat16, device='npu')
torch_npu.atb.npu_paged_cache_load(cache_kv_c, cache_k_pe, block_table2, ctx_len2,
                                   seq_starts=starts2, key=out2, value=out2p)
print('LOAD_ORDER2 rows->block_ids:', [int(v - 1) for v in out2[:, 0, 0].float().tolist()], flush=True)
print('PROBE5H6_DONE', flush=True)

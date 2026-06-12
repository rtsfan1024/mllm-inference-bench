# encoding=utf-8

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 创建一个大矩阵乘法任务模拟Linear层，使用FP16以激活Tensor Core的可能性
device = torch.device("cuda:0")
a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
b = torch.randn(4096, 4096, device=device, dtype=torch.float16)

# 预热GPU(Warm-up)
torch.matmul(a, b)

print("开始抓取cuBLAS算子...")

# 使用Profiler抓取GPU底层活动
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        c = torch.matmul(a, b)

# 过滤出包含GEMM的底层算子
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))




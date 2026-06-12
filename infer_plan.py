# encoding=utf-8

import torch
import tensorrt as trt

# 加载引擎
logger = trt.Logger(trt.Logger.ERROR)

with open("gnn.plan", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# 准备数据：模拟5个节点、特征16维
x = torch.randn(5, 16).cuda()
adj = torch.randn(5, 5).cuda()

# 预分配输出空间
output = torch.empty(5, 8).cuda()

# 设置维度 & 绑定地址，告诉TRT现在的动态形状是多少
context.set_input_shape("x", x.shape)
context.set_input_shape("adj", adj.shape)

# 告诉TRT每个名字对应的显存地址在哪里
context.set_tensor_address("x", int(x.data_ptr()))
context.set_tensor_address("adj", int(adj.data_ptr()))
context.set_tensor_address("output", int(output.data_ptr()))

# 执行推理 (使用TRT v3接口)
context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)

# 验证结果
print("Shape:", output.shape)
print("前5行数据:\n", output[:5])




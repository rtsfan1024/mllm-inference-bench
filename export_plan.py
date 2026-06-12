# encoding=utf-8

import tensorrt as trt

# 创建Logger
logger = trt.Logger(trt.Logger.WARNING)
# 创建Builder
builder = trt.Builder(logger)
# 创建Network(显式Batch模式)
# network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network()


# 创建ONNX Parser
parser = trt.OnnxParser(network, logger)

# 读取ONNX文件(需提前确定gnn.onnx和.data在同一路径下)
with open("gnn.onnx", "rb") as f:
    parser.parse(f.read())

# 创建构建配置和优化配置 (Profile)
config = builder.create_builder_config()
profile = builder.create_optimization_profile()

# 设定动态维度的范围 (Min, Opt, Max)，假设特征维度固定为 16
profile.set_shape("x", (1, 16), (100, 16), (1024, 16))
profile.set_shape("adj", (1, 1), (100, 100), (1024, 1024))

# 将Profile添加到Config中
config.add_optimization_profile(profile)
# 设置最大显存(比如1GB)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# 构建(Build)
plan = builder.build_serialized_network(network, config)

# 保存.plan文件
with open("gnn.plan", "wb") as f:
    f.write(plan)




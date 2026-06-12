# encoding=utf-8

import torch
import torch.nn as nn
import torch.onnx
from torch.export import Dim

# 一个普通神经网络 (MLP)
dim_mlp_batch = Dim("mlp_batch", min=1)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 2)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

model_mlp = SimpleMLP()
dummy_input_mlp = torch.randn(1, 10)

print("\n转换 MLP 模型...")

torch.onnx.export(
    model_mlp,
    dummy_input_mlp,
    "mlp.onnx",
    dynamo=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_shapes=({0: dim_mlp_batch},)
)

print("mlp.onnx导出成功！")

# 强化学习模型
dim_rl_batch = Dim("rl_batch", min=1)

class DQNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(4, 16, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 6 * 6, 4)

    def forward(self, state):
        x = self.cnn(state)
        x = self.flatten(x)
        return self.fc(x)

model_rl = DQNPolicy()
dummy_input_rl = torch.randn(1, 4, 8, 8)

print("\n转换 RL 策略网络...")

torch.onnx.export(
    model_rl,
    dummy_input_rl,
    "rl_policy.onnx",
    dynamo=True,
    input_names=["state"],
    output_names=["action"],
    dynamic_shapes=({0: dim_rl_batch},)
)

print("rl_policy.onnx导出成功！")

# 图神经网络 (GNN)
class SimpleGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return output

model_gnn = SimpleGCN(16, 8)

# 输入数据
dummy_node_features = torch.randn(5, 16)
dummy_adjacency = torch.randn(5, 5)
dummy_input_gnn = (dummy_node_features, dummy_adjacency)

dim_n = Dim("n", min=1)     # 输入节点数
dim_out = Dim("out", min=1) # 输出节点数

print("\n转换 GNN 模型...")

torch.onnx.export(
    model_gnn,
    dummy_input_gnn,
    "gnn.onnx",
    dynamo=True,
    input_names=["x", "adj"],
    output_names=["output"],
    dynamic_shapes=(
        {0: dim_n},
        {0: dim_out, 1: dim_n}
    )
)

print("gnn.onnx导出成功！")




# 多模态模型评估体系与推理优化

## 项目概述

本项目聚焦多模态大模型的**评估体系**与**推理优化**两大核心主题，涵盖从评估指标计算、模型格式转换、推理引擎部署到底层性能分析的完整链路。

---

## 目录结构

```
.
├── 评估体系
│   ├── clip_score.py              # CLIP 图文相似度评分（图文对齐质量）
│   ├── FID.py                     # FID（Fréchet Inception Distance）图像生成质量评估
│   ├── CIDEr_and_SPICE.py         # CIDEr / SPICE 图像描述质量评估
│   └── clip-vit-base-patch32/     # CLIP 模型权重（本地离线推理用）
│
├── 推理优化
│   ├── on_vllm.py                 # vLLM 推理：单请求基准测试
│   ├── off_vllm_benchmark.py      # 原生 Transformers 推理：并发吞吐量对比
│   ├── client_benchmark.py        # 异步并发客户端压测（模拟多用户）
│   ├── pagedattention_demo.py     # PagedAttention 显存管理原理演示
│   └── measure_qwen.py            # Qwen2.5-VL 推理速度测量
│
├── 模型格式转换（ONNX / TensorRT）
│   ├── export_onnx.py             # PyTorch → ONNX 导出（MLP / RL / GNN）
│   ├── export_plan.py             # ONNX → TensorRT .plan 引擎构建
│   ├── infer_plan.py              # TensorRT .plan 引擎推理
│   └── builder_nvidia.py          # TensorRT BERT Builder（FP16/INT8/Plugin/稀疏化）
│
├── 底层性能分析
│   ├── cuBLAS_demo.py             # cuBLAS GEMM 算子 Profiling
│   ├── python_gil.py              # Python GIL 对多线程性能的影响
│   ├── python_versus_cpp.py       # Python vs C++（NumPy）性能对比
│   ├── gil_test.cpp               # C++ 多线程并行（无 GIL 限制）
│   └── gil_test.java              # Java 多线程并行对比
│
├── 模型权重
│   ├── Qwen2.5-0.5B-Instruct/    # Qwen2.5 0.5B 模型（~954MB）
│   └── Qwen2.5-VL-7B-Instruct/   # Qwen2.5 VL 7B 多模态模型（~16GB）
│
├── 测试图片
│   ├── real1.jpg / real2.jpg      # FID 真实图片样本
│   └── fake1.png / fake2.png      # FID 生成图片样本
│
└── benchmark_throughput.py        # vLLM 吞吐量基准测试（已迁移至 vllm bench CLI）
```

---

## 核心内容

### 1. 多模态评估体系

| 评估指标 | 用途 | 文件 |
|---------|------|------|
| **CLIP Score** | 衡量图文对齐质量（图文匹配度） | `clip_score.py` |
| **FID** | 衡量图像生成质量（真实 vs 生成分布距离） | `FID.py` |
| **CIDEr** | 衡量图像描述的关键词共识度 | `CIDEr_and_SPICE.py` |
| **SPICE** | 衡量图像描述的语义场景图谱（需 Java 8） | `CIDEr_and_SPICE.py` |

### 2. 推理引擎对比

- **原生 Transformers**：`off_vllm_benchmark.py` — 基线推理，支持 batch 推理但无调度优化
- **vLLM**：`on_vllm.py` — 基于 PagedAttention 的高效推理引擎
- **异步压测**：`client_benchmark.py` — 模拟 20 并发用户，测量系统吞吐量（TPS）

### 3. 模型格式转换链路

```
PyTorch → ONNX (export_onnx.py) → TensorRT .plan (export_plan.py) → 推理 (infer_plan.py)
```

- 支持 MLP / DQN / GNN 等多种网络结构的动态形状导出
- TensorRT Builder 支持 FP16 混合精度、INT8 量化（PTQ/QAT）、稀疏化（A100+）

### 4. 底层优化原理

- **cuBLAS**：通过 `torch.profiler` 抓取 GPU 底层 GEMM 算子
- **GIL 问题**：Python 多线程在 CPU 密集型任务下的性能瓶颈
- **编译优化**：NumPy（C++ 后端）相比 Python 原生循环的加速比

---

## 环境依赖

```bash
pip install torch torchvision transformers accelerate qwen-vl-utils
pip install vllm onnx onnxruntime
pip install torchmetrics torch-fidelity    # FID 计算
pip install pycocoevalcap                   # CIDEr / SPICE 计算
pip install tensorrt                        # TensorRT（需匹配 CUDA 版本）
pip install aiohttp                         # 异步压测
```

---

## 快速开始

```bash
# CLIP 图文相似度评估
python clip_score.py

# FID 图像质量评估
python FID.py

# vLLM 推理（需 GPU + 模型权重）
python on_vllm.py

# ONNX 导出
python export_onnx.py

# PagedAttention 原理演示
python pagedattention_demo.py
```

---

## 未上传文件说明

以下文件因体积较大或属于生成产物，未纳入版本管理，需自行下载或由脚本生成。

### 模型权重

| 目录 | 大小 | 用途 | 下载地址 |
|------|------|------|---------|
| `clip-vit-base-patch32/` | ~581MB | CLIP 评估时的本地模型，用于 `clip_score.py` 图文相似度计算 | [HuggingFace openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| `Qwen2.5-0.5B-Instruct/` | ~954MB | 轻量级文本模型，用于 `measure_qwen.py` 推理速度测量 | [HuggingFace Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| `Qwen2.5-VL-7B-Instruct/` | ~16GB | 多模态视觉语言模型，用于 vLLM / Transformers 推理对比和客户端压测 | [HuggingFace Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

> 下载方式（以 CLIP 为例）：
> ```bash
> # 方式一：huggingface-cli
> huggingface-cli download openai/clip-vit-base-patch32 --local-dir clip-vit-base-patch32
>
> # 方式二：Python 代码自动下载（脚本中已内置）
> # 取消 clip_score.py 中 from_pretrained 的注释即可联网加载
> ```

### 测试图片

| 文件 | 大小 | 用途 | 来源 |
|------|------|------|------|
| `real1.jpg` | ~2.2MB | FID 评估的真实图片样本（`FID.py`） | 可替换为任意真实图片 |
| `real2.jpg` | ~607KB | FID 评估的真实图片样本（`FID.py`） | 可替换为任意真实图片 |
| `fake1.png` | ~7.3MB | FID 评估的生成图片样本（`FID.py`） | 可替换为模型生成图片 |
| `fake2.png` | ~5.7MB | FID 评估的生成图片样本（`FID.py`） | 可替换为模型生成图片 |

> 真实图片与生成图片各准备 2 张以上即可运行 `FID.py`，不限定具体来源。

### 脚本生成产物

以下文件由对应脚本运行时自动生成，无需手动下载：

| 文件 | 生成方式 | 说明 |
|------|---------|------|
| `mlp.onnx` / `mlp.onnx.data` | `python export_onnx.py` | MLP 模型的 ONNX 导出 |
| `rl_policy.onnx` / `rl_policy.onnx.data` | `python export_onnx.py` | DQN 策略网络的 ONNX 导出 |
| `gnn.onnx` / `gnn.onnx.data` | `python export_onnx.py` | 图神经网络的 ONNX 导出 |
| `gnn.plan` | `python export_plan.py` | TensorRT 优化引擎（需 NVIDIA GPU） |
| `GilTest.class` | `javac gil_test.java` | Java 多线程对比测试的编译产物 |

---

## 注意事项

- 模型权重文件较大（Qwen2.5-VL-7B 约 16GB），需单独下载
- SPICE 评估需要本地 Java 8 环境
- TensorRT 相关脚本需要 NVIDIA GPU + 对应版本的 TensorRT 和 CUDA
- `builder_nvidia.py` 来自 NVIDIA 官方示例，包含完整的 BERT TRT 构建流程

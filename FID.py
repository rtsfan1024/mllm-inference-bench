# encoding=utf-8

import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance  #需pip install torchmetrics torch-fidelity

# 初始化FID计算器 (会下载Inception模型)
fid = FrechetInceptionDistance(feature=64) # 此处用64维特征，实际常用2048

# 定义预处理 (Resize -> Tensor -> uint8)
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

def load_batch(path1, path2):
    # 读取两张图片
    img1 = preprocess(Image.open(path1).convert("RGB"))
    img2 = preprocess(Image.open(path2).convert("RGB"))

    # 转为uint8[0,255]格式
    img1 = (img1 * 255).to(dtype=torch.uint8)
    img2 = (img2 * 255).to(dtype=torch.uint8)

    # 堆叠成一个Batch:[2,3,299,299]
    return torch.stack([img1, img2])

# 加载数据
real_batch = load_batch("real1.jpg", "real2.jpg")
fake_batch = load_batch("fake1.png", "fake2.png")

# 计算指标
fid.update(real_batch, real=True)
fid.update(fake_batch, real=False)

result = fid.compute()

print(f"FID Score: {result.item():.4f}")




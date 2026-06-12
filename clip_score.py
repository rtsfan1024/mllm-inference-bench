# encoding=utf-8

import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 加载模型（联网加载）
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 加载模型 (本地加载)
local_model_path = "clip-vit-base-patch32"
model = CLIPModel.from_pretrained(local_model_path)
# processor = CLIPProcessor.from_pretrained(local_model_path, use_fast=True)
processor = CLIPProcessor.from_pretrained(local_model_path)

# 准备图片和文本
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
# text = ["a photo of a cat", "a photo of a dog"] # 对比
text = ["一只猫在图片中", "一条狗在图片中"] # 对比

# 处理输入
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

# 计算相似度
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 图-文相似度
probs = logits_per_image.softmax(dim=1)      # 转为概率

# 结果应该是cat的分数远高于dog
print("CLIP Scores:", logits_per_image.detach().numpy())
print("Probability:", probs.detach().numpy())





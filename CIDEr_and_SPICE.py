# encoding=utf-8
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# 1. 准备标准数据集（真值 / 人类标注答案）
ref = {
    'img1': ['a cat sitting on the mat', 'a furry cat on a rug'],
    'img2': ['a man playing guitar']
}
# 2. 假设这是待评测多模态大模型的预测结果
gene = {
    'img1': ['a cat on a mat'],
    'img2': ['a person playing music']
}

# 3. 计算 CIDEr（聚焦关键词共识度）
cider_scorer = Cider()
score_cider, _ = cider_scorer.compute_score(ref, gene)
print(f"CIDEr Score: {score_cider}")  # 输出示例: 1.8413

# 4. 计算 SPICE（聚焦语义场景图谱，需本地 Java 8 环境支持）
# spice_scorer = Spice()
# score_spice, _ = spice_scorer.compute_score(ref, gene)
# print(f"SPICE Score: {score_spice}")
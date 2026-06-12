# encoding=utf-8

"""
pip install transformers accelerate qwen-vl-utils
"""

import time
import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/root/Qwen2.5-VL-7B-Instruct"

# 模拟一张测试的384*384的噪点图
def create_dummy_image():
    return Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))

def main():
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 准备数据
    image = create_dummy_image()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail and write a long poem about it."},
            ],
        }
    ]

    # 预处理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # 推理计时
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    end_time = time.time()

    # 统计结果
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    new_tokens = len(generated_ids_trimmed[0])
    duration = end_time - start_time
    tps = new_tokens / duration

    print(f"总耗时: {duration:.2f} s")
    print(f"Token数量: {new_tokens}")
    print(f"速度: {tps:.2f} tokens/s")

if __name__ == "__main__":
    main()




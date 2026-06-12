# encoding=utf-8

import time
import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/hy-tmp/Qwen2.5-VL-7B-Instruct"
# 并发数(模拟20人同时访问)
BATCH_SIZE = 20

def create_dummy_image():
    return Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))

def main():
    # 加载模型
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 准备X份数据(因为模拟多个用户，X=默认的人数)
    image = create_dummy_image()
    messages_list = []

    # 构造Batch
    for _ in range(BATCH_SIZE):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        messages_list.append(messages)

    # 预处理(Padding会消耗显存)
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_list
    ]
    image_inputs, video_inputs = process_vision_info(messages_list)

    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,  # 必须Padding才能batch推理
        return_tensors="pt",
    ).to("cuda")

    start_time = time.time()

    # 推理
    generated_ids = model.generate(**inputs, max_new_tokens=200)

    end_time = time.time()
    duration = end_time - start_time

    # 统计TPS
    total_new_tokens = 0
    for i in range(BATCH_SIZE):
        in_len = len(inputs.input_ids[i])
        out_len = len(generated_ids[i])
        total_new_tokens += (out_len - in_len)

    tps = total_new_tokens / duration

    print(f"总耗时: {duration:.2f} s")
    print(f"总吞吐量 (TPS): {tps:.2f} tokens/s")

if __name__ == "__main__":
    main()




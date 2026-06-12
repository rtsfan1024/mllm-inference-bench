# encoding=utf-8

"""
pip install vllm
"""

import time
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_PATH = "/hy-tmp/Qwen2.5-VL-7B-Instruct"

# 模拟一张测试的384*384的噪点图
def create_dummy_image():
    return Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))

def main():
    # 初始化vLLM
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # 准备数据
    image = create_dummy_image()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail and write a long poem about it."},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = {
        "prompt": prompt_text,
        "multi_modal_data": {
            "image": image
        },
    }

    # 采样参数
    sampling_params = SamplingParams(temperature=0.1, max_tokens=200, ignore_eos=True)

    # 推理计时
    start_time = time.time()
    outputs = llm.generate([inputs], sampling_params)
    end_time = time.time()

    # 统计结果
    new_tokens = len(outputs[0].outputs[0].token_ids)
    duration = end_time - start_time
    tps = new_tokens / duration

    print(f"总耗时: {duration:.2f} s")
    print(f"Token数量: {new_tokens}")
    print(f"速度: {tps:.2f} tokens/s")

if __name__ == "__main__":
    main()




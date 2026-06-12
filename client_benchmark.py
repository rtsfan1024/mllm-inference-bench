# encoding=utf-8

import time
import asyncio
import aiohttp
import numpy as np
from PIL import Image
import base64
from io import BytesIO

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "/root/Qwen2.5-VL-7B-Instruct"
CONCURRENCY = 20  # 模拟20人并发

# 生成一张测试图并转成Base64
def create_dummy_image():
    img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

async def send_request(session):
    image_b64 = create_dummy_image()
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": "Describe this image."}  # 简短的Prompt
                ]
            }
        ],
        "max_tokens": 200,  # 限制输出长度以便测速
        "temperature": 0.1
    }

    async with session.post(API_URL, json=payload) as resp:
        if resp.status == 200:
            result = await resp.json()
            # 获取生成的token数量
            usage = result.get('usage', {}).get('completion_tokens', 0)
            return usage
        else:
            print(f"Error: {resp.status}")
            return 0

async def main():
    async with aiohttp.ClientSession() as session:
        # 同时发起20个请求
        tasks = [send_request(session) for _ in range(CONCURRENCY)]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

    total_tokens = sum(results)
    total_time = end_time - start_time
    # 系统吞吐量 = 所有生成的Token总和 / 总耗时
    tps = total_tokens / total_time

    print(f"总耗时: {total_time:.2f} s")
    print(f"总生成Token: {total_tokens}")
    print(f"系统吞吐量(TPS): {tps:.2f} tokens/s")

if __name__ == "__main__":
    asyncio.run(main())




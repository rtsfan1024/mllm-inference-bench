# encoding=utf-8

import time
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 加载本地模型
model_path = "Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# 首次预热
dummy = tokenizer(["Warmup"], return_tensors="pt").to(model.device)
model.generate(**dummy, max_new_tokens=2) # 随便跑2个字，唤醒CUDA

# 准备输入
prompt = "简单介绍一下什么是深度学习"

# 使用Qwen的聊天模版格式化输入
messages = [
    {
        "role": "user",
        "content": prompt
    }
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 初始化流式传输器(测试TTFT)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)

# 启动生成(必须在子线程运行，否则主线程无法同时计时)
thread = Thread(target=model.generate, kwargs=generation_kwargs)

start_time = time.perf_counter() # 计时开始
thread.start()

# 循环接收Token并计时
first_token_time = None
token_count = 0

for new_text in streamer:
    current_time = time.perf_counter()

    # 捕获第一个Token(TTFT)
    if first_token_time is None:
        first_token_time = current_time
        ttft = (first_token_time - start_time) * 1000 # 换算成毫秒

    token_count += 1
    print(new_text, end="", flush=True) # 打印生成的字

end_time = time.perf_counter()

# 计算TPOT：TPOT = (结束时间 - 首字时间) / (生成的 Token 数 - 1)；意思是除去第一个字，后面平均每个字花了多久
if token_count > 1:
    generation_duration = end_time - first_token_time
    tpot = (generation_duration / (token_count - 1)) * 1000
else:
    tpot = 0

print(f"\n\n生成长度: {token_count} tokens")
print(f"TTFT (首字延迟): {ttft:.2f} ms")
print(f"TPOT (生成耗时): {tpot:.2f} ms/token")




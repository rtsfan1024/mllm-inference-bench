# encoding=utf-8

import time
import threading

# 定义一个纯粹CPU密集型计算任务：计算1亿次
def cpu_task():
    count = 0
    for i in range(10**8):
        count += 1

# 单线程串行跑两次
start = time.time()
cpu_task()
cpu_task()
serial_time = time.time() - start

print(f"单线程串行耗时: {serial_time:.4f} 秒")

# 多线程并行跑（理论上应该时间减半）
start = time.time()
t1 = threading.Thread(target=cpu_task)
t2 = threading.Thread(target=cpu_task)
t1.start(); t2.start()
t1.join(); t2.join()
thread_time = time.time() - start

print(f"多线程并行耗时: {thread_time:.4f} 秒")




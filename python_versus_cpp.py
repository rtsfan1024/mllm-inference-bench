# encoding=utf-8

import time
import numpy as np

# 计算1到1亿的累加和
N = 10**7 # 一千万次

# Python原生写法(模拟PyTorch的纯Python逻辑)
start = time.time()
py_sum = 0
for i in range(N):
    py_sum += i
py_time = time.time() - start

print(f"Python原生循环耗时: {py_time:.4f} 秒")

# NumPy写法(底层是C/C++，模拟ONNX/TensorRT)
start = time.time()
np_sum = np.sum(np.arange(N))
c_time = time.time() - start
print(f"C++/Optimized耗时: {c_time:.4f} 秒")

print(f"\n结论：编译后的代码比Python快了 {py_time / c_time:.1f} 倍！")




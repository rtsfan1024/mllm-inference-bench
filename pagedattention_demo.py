# encoding=utf-8

import time

class PhysicalMemory:
    def __init__(self, total_blocks=12):
        # 模拟物理显存，一共有12个Block，其中None代表空闲，字符串代表被占用
        self.blocks = [None] * total_blocks

    def show_state(self):
        # 可视化打印当前物理显存的状态
        state = []
        for i, b in enumerate(self.blocks):
            val = b if b else "   "  # 空闲块显示空格
            state.append(f"|{i:2}: {val}|")
        print("物理显存状态: " + "".join(state))

    def allocate_contiguous(self, size, req_id):
        # 传统模式：尝试分配连续空间，必须找到一段连续的None
        count = 0
        start_index = -1
        for i, b in enumerate(self.blocks):
            if b is None:
                if count == 0: start_index = i
                count += 1
                if count == size:
                    # 找到了！分配
                    for j in range(start_index, start_index + size):
                        self.blocks[j] = req_id
                    return True
            else:
                count = 0
                start_index = -1
        return False  # 没找到连续空间

    def allocate_paged(self, size, req_id):
        # PagedAttention模式：分配离散空间（只要空闲总数够就行）
        free_indices = [i for i, b in enumerate(self.blocks) if b is None]

        if len(free_indices) < size:
            return None  # 表示物理空间满了

        # 拿出前size个空闲块的索引
        allocated_indices = free_indices[:size]

        # 在物理显存里标记占用
        for idx in allocated_indices:
            self.blocks[idx] = req_id

        return allocated_indices  # 返回页表（逻辑 -> 物理的映射）

print("1. 初始化显存(12个Block)")
gpu = PhysicalMemory(12)
gpu.show_state()
time.sleep(1)

print("\n2. 模拟请求A,B,C (传统连续分配)")
gpu.allocate_contiguous(3, " A ")
gpu.allocate_contiguous(3, " B ")
gpu.allocate_contiguous(3, " C ")
gpu.show_state()
time.sleep(1)

print("\n3. 请求B完成并释放")
for i in range(3, 6): gpu.blocks[i] = None
gpu.show_state()
time.sleep(2)

print("\n4. 出现一个大请求D(需要5个块)")
success = gpu.allocate_contiguous(5, " D ")
if not success:
    print("OOM (Out Of Memory)！")
time.sleep(2)

print("\n[系统] 切换到PagedAttention模式...")
page_table = gpu.allocate_paged(5, " D ")

if page_table:
    print(f"生成页表 (Page Table): 逻辑块 [0,1,2,3,4] -> 物理块 {page_table}")
    gpu.show_state()

time.sleep(3)
print("\n5. Prompt共享 (Parallel Sampling)")

# E只需要申请1个新块作为它的新生成内容
new_block_for_E = gpu.allocate_paged(1, " E ")
# E的页表 = D的页表 + E的新块
page_table_E = page_table + new_block_for_E

print(f"Request E的页表: {page_table_E}")
gpu.show_state()


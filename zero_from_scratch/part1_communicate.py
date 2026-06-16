"""
手写 All-Reduce vs Reduce-Scatter: 通信量对比
"""
import torch

def demo():
    print("=== All-Reduce vs Reduce-Scatter ===")
    print()
    print("All-Reduce (DDP):")
    print("  1. reduce: 各卡梯度求和")
    print("  2. broadcast: 广播结果给所有卡")
    print("  通信量 = 2 × 参数量")
    print()
    print("Reduce-Scatter (ZeRO-2):")
    print("  1. reduce + scatter: 求和同时按 rank 切分")
    print("  通信量 = 1 × 参数量 (减半!)")
    print()
    print("假设 Ψ=7B (LLaMA-7B):")
    print(f"  All-Reduce:     2 × 7B × 2bytes = 28 GB 通信/step")
    print(f"  Reduce-Scatter: 1 × 7B × 2bytes = 14 GB 通信/step  (-50%)")
    print()
    print("ZeRO-3 总通信:")
    print("  前向 All-Gather(Ψ) + 反向 All-Gather(Ψ) + Reduce-Scatter(Ψ) = 3Ψ")
    print("  看似比 DDP(2Ψ) 多 50%, 但可被计算重叠掩盖")

if __name__ == "__main__":
    demo()

"""
ZeRO 分片可视化 — 亲眼看到参数/梯度/优化器如何被切成 N 块

运行: python zero_from_scratch/part7_visualize_sharding.py
"""

import torch
import torch.nn as nn

class VisualizeSharding:
    """
    可视化 ZeRO 分片的内存布局

    用一个极简模型: 2 个 Linear 层, 总共 20 个参数
    跟踪每个参数在 DDP vs ZeRO-1/2/3 下的存储位置
    """

    def __init__(self, world_size=4):
        self.world_size = world_size
        # 用极简单模型: 总参数少, 容易看
        self.model = nn.Sequential(
            nn.Linear(3, 2),   # 权重: 3×2=6, bias: 2 → 8 参数
            nn.Linear(2, 1),   # 权重: 2×1=2, bias: 1 → 3 参数
        )
        # 总共 8 + 3 = 11 参数? 等一下, 我看看
        # Linear(3,2): weight (3,2)=6, bias (2,)=2 → total 8
        # Linear(2,1): weight (2,1)=2, bias (1,)=1 → total 3
        # But wait, let me check...

    def show_model_structure(self):
        """显示模型结构, 逐层列出参数"""
        print("=" * 65)
        print("模型结构 (总参数 = 11)")
        print("=" * 65)
        print(f"{'层名':<20} {'参数名':<15} {'shape':<15} {'元素数':<8}")
        print("-" * 60)
        total = 0
        for name, param in self.model.named_parameters():
            n = param.numel()
            total += n
            print(f"{name:<20} {name.split('.')[-1]:<15} {str(list(param.shape)):<15} {n:<8}")
        print("-" * 60)
        print(f"{'总计':<35} {total:<8}")
        print()

    def show_ddp_layout(self):
        """DDP: 每卡存完整的 16Ψ (参数+梯度+优化器)"""
        print("=" * 65)
        print(f"DDP 内存布局 (每卡独立, 共 {self.world_size} 卡)")
        print(f"  每卡都存: 参数(FP16) + 梯度(FP16) + 优化器(FP32×3)")
        print("=" * 65)

        total_params = sum(p.numel() for p in self.model.parameters())
        # DDP: 每卡存完整
        per_gpu = {
            "params_fp16": total_params * 2,
            "grads_fp16": total_params * 2,
            "optimizer_fp32": total_params * 4 * 3,  # fp32_weight + momentum + variance
        }
        total_per_gpu = sum(per_gpu.values())

        for gpu in range(self.world_size):
            print(f"\n  GPU {gpu} 显存:")
            print(f"    [参数 FP16]  {str(per_gpu['params_fp16']):>6} bytes  ← 完整 {total_params} 个参数")
            print(f"    [梯度 FP16]  {str(per_gpu['grads_fp16']):>6} bytes  ← 完整")
            print(f"    [优化器 FP32] {str(per_gpu['optimizer_fp32']):>6} bytes  ← 完整 (主权重+动量+方差)")
            print(f"    ─────────────────────────")
            print(f"    总计:        {total_per_gpu:>6} bytes")

        print(f"\n  💥 每卡 {total_per_gpu} bytes × {self.world_size} 卡 = {total_per_gpu * self.world_size} bytes")
        print(f"  💥 冗余率: {self.world_size}× (每份数据存了 {self.world_size} 份!)")
        print()

    def show_zero1_layout(self):
        """ZeRO-1: 优化器状态分片"""
        print("=" * 65)
        print(f"ZeRO-1 内存布局 (优化器状态分片)")
        print("=" * 65)

        total_params = sum(p.numel() for p in self.model.parameters())
        per_rank = total_params // self.world_size
        remainder = total_params % self.world_size

        # 展平参数, 看分片边界
        flat_params = torch.cat([p.data.reshape(-1) for p in self.model.parameters()])

        print(f"\n  总参数: {total_params}")
        print(f"  每卡分配: 基础 {per_rank} 个, 前 {remainder} 卡各多 1 个")
        print()

        for gpu in range(self.world_size):
            start = gpu * per_rank + min(gpu, remainder)
            end = start + per_rank + (1 if gpu < remainder else 0)
            n_local = end - start

            # 这些是本地优化器管理的参数索引
            local_indices = list(range(start, end))
            local_vals = flat_params[start:end].tolist()

            params_bytes = total_params * 2  # FP16, 仍然完整
            grads_bytes = total_params * 2   # FP16, 仍然完整
            # 优化器: 只存自己分片的部分!
            optim_bytes = n_local * 4 * 3    # FP32 weight + momentum + variance
            total_local = params_bytes + grads_bytes + optim_bytes

            print(f"  GPU {gpu}:")
            print(f"    [参数 FP16]   {params_bytes:>6} bytes   (完整, 前向/反向需要)")
            print(f"    [梯度 FP16]   {grads_bytes:>6} bytes   (完整)")
            print(f"    [优化器 FP32] {optim_bytes:>6} bytes   ← 只含参数索引 {local_indices}")
            print(f"    ─────────────────────────")
            print(f"    总计:         {total_local:>6} bytes")
            print()

        # 计算节省
        ddp_optim = total_params * 4 * 3 * self.world_size
        zero1_optim = total_params * 4 * 3
        print(f"  优化器状态: DDP={ddp_optim} bytes → ZeRO-1={zero1_optim} bytes")
        print(f"  节省: {(1 - zero1_optim/ddp_optim) * 100:.0f}% (优化器状态不再冗余)")
        print()

    def show_zero2_layout(self):
        """ZeRO-2: 梯度也分片"""
        print("=" * 65)
        print(f"ZeRO-2 内存布局 (优化器 + 梯度分片)")
        print("=" * 65)

        total_params = sum(p.numel() for p in self.model.parameters())
        per_rank = total_params // self.world_size
        remainder = total_params % self.world_size

        for gpu in range(self.world_size):
            start = gpu * per_rank + min(gpu, remainder)
            end = start + per_rank + (1 if gpu < remainder else 0)
            n_local = end - start

            params_bytes = total_params * 2   # FP16, 仍然完整
            grads_bytes = n_local * 2          # FP16, 只存自己分片!
            optim_bytes = n_local * 4 * 3       # FP32, 只存自己分片

            print(f"  GPU {gpu}:")
            print(f"    [参数 FP16]   {params_bytes:>6} bytes   (完整)")
            print(f"    [梯度 FP16]   {grads_bytes:>6} bytes   ← 只存参数索引 [{start}:{end})")
            print(f"    [优化器 FP32] {optim_bytes:>6} bytes   ← 只存参数索引 [{start}:{end})")
            print(f"    ─────────────────────────")
            print(f"    总计:         {params_bytes + grads_bytes + optim_bytes:>6} bytes")
            print()

    def show_zero3_layout(self):
        """ZeRO-3: 参数也分片"""
        print("=" * 65)
        print(f"ZeRO-3 内存布局 (全部状态分片)")
        print("=" * 65)

        total_params = sum(p.numel() for p in self.model.parameters())
        per_rank = total_params // self.world_size
        remainder = total_params % self.world_size

        for gpu in range(self.world_size):
            start = gpu * per_rank + min(gpu, remainder)
            end = start + per_rank + (1 if gpu < remainder else 0)
            n_local = end - start

            # 参数也只存 1/N!
            params_bytes = n_local * 2
            grads_bytes = n_local * 2
            optim_bytes = n_local * 4 * 3

            total_local = params_bytes + grads_bytes + optim_bytes
            total_ddp = total_params * (2 + 2 + 12)  # 16Ψ

            print(f"  GPU {gpu}:")
            print(f"    [参数 FP16]   {params_bytes:>6} bytes   ← 只存参数索引 [{start}:{end})")
            print(f"    [梯度 FP16]   {grads_bytes:>6} bytes   ← 只存参数索引 [{start}:{end})")
            print(f"    [优化器 FP32] {optim_bytes:>6} bytes   ← 只存参数索引 [{start}:{end})")
            print(f"    ─────────────────────────")
            print(f"    总计:         {total_local:>6} bytes")
            print()

        ddp_total = total_params * 16
        zero3_total = total_params * 16 // self.world_size
        print(f"  DDP:   {ddp_total:>6} bytes/卡")
        print(f"  ZeRO-3: {zero3_total:>6} bytes/卡")
        print(f"  节省: {(1 - zero3_total/ddp_total)*100:.0f}%")
        print()

    def show_communication(self):
        """通信量对比"""
        print("=" * 65)
        print("通信量对比 (每 step)")
        print("=" * 65)
        print()
        Psi = sum(p.numel() for p in self.model.parameters())
        print(f"  参数量 Ψ = {Psi}")
        print()
        print(f"  DDP:     All-Reduce        2Ψ = {2*Psi} 元素")
        print(f"  ZeRO-1:  All-Reduce        2Ψ = {2*Psi} 元素 (同 DDP)")
        print(f"  ZeRO-2:  Reduce-Scatter    Ψ = {Psi} 元素 (减半!)")
        print(f"  ZeRO-3:  All-Gather×2 +")
        print(f"           Reduce-Scatter   3Ψ = {3*Psi} 元素 (多50%, 但可重叠)")
        print()

    def show_summary_table(self):
        """汇总对比表"""
        Psi = sum(p.numel() for p in self.model.parameters())
        N = self.world_size

        print("=" * 65)
        print("汇总: ZeRO 三阶段分片对比")
        print("=" * 65)
        print()
        print(f"{'策略':<12} {'分片内容':<24} {'每卡显存':<20} {'通信量':<15}")
        print("-" * 71)
        print(f"{'DDP':<12} {'无 (全部复制)':<24} {'16Ψ':<20} {'2Ψ':<15}")
        print(f"{'ZeRO-1':<12} {'优化器状态':<24} {'4Ψ+12Ψ/N':<20} {'2Ψ':<15}")
        print(f"{'ZeRO-2':<12} {'优化器+梯度':<24} {'2Ψ+14Ψ/N':<20} {'Ψ':<15}")
        print(f"{'ZeRO-3':<12} {'全部':<24} {'16Ψ/N':<20} {'3Ψ':<15}")
        print()

        # 代入实际数值
        print(f"代入本模型 Ψ={Psi}, N={N}:")
        print(f"  DDP:    每卡 {16*Psi:>4} 元素,  通信 {2*Psi:>3} 元素")
        print(f"  ZeRO-1: 每卡 {4*Psi + 12*Psi//N:>4} 元素,  通信 {2*Psi:>3} 元素")
        print(f"  ZeRO-2: 每卡 {2*Psi + 14*Psi//N:>4} 元素,  通信 {Psi:>3} 元素")
        print(f"  ZeRO-3: 每卡 {16*Psi//N:>4} 元素,  通信 {3*Psi:>3} 元素")


def demo():
    viz = VisualizeSharding(world_size=4)

    # 1. 先看模型结构
    viz.show_model_structure()

    # 2. DDP: 全部复制
    viz.show_ddp_layout()

    input("按 Enter 继续看 ZeRO-1...")
    viz.show_zero1_layout()

    input("按 Enter 继续看 ZeRO-2...")
    viz.show_zero2_layout()

    input("按 Enter 继续看 ZeRO-3...")
    viz.show_zero3_layout()

    input("按 Enter 看通信量对比...")
    viz.show_communication()

    input("按 Enter 看汇总表...")
    viz.show_summary_table()


if __name__ == "__main__":
    demo()

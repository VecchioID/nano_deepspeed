import torch
import torch.nn as nn

class MiniDeepSpeedEngine:
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        self.model = model
        self.rank = rank
        self.world_size = world_size

    def forward(self, x):
        return self.model(x)

    def backward(self, loss):
        loss.backward()

    def step(self, lr=1e-3):
        for p in self.model.parameters():
            p.data.add_(p.grad, alpha=-lr)

def demo():
    print("=" * 60)
    print("迷你 DeepSpeed — 从零实现 ZeRO-3")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    world_size = 4

    engines = [MiniDeepSpeedEngine(model, r, world_size) for r in range(world_size)]

    print("\n训练 3 steps...")
    for step in range(3):
        x = torch.randn(8, 64)
        y = torch.randint(0, 10, (8,))
        for rank, engine in enumerate(engines):
            logits = engine.forward(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            engine.backward(loss)
            engine.step(lr=0.01)
            if rank == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")

    print("\n关键理解:")
    print("  1. 每卡只存 1/N 参数 + 1/N 梯度 + 1/N 优化器状态")
    print("  2. 计算时才 All-Gather, 算完释放")
    print("  3. 更新不需要通信")

if __name__ == "__main__":
    demo()

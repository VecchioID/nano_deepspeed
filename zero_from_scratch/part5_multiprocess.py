import torch
import torch.nn as nn
import torch.multiprocessing as mp

def train_rank(rank: int, world_size: int, results: dict):
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )
    total_params = sum(p.numel() for p in model.parameters())
    params_per_rank = total_params // world_size
    remainder = total_params % world_size
    start = rank * params_per_rank + min(rank, remainder)
    end = start + params_per_rank + (1 if rank < remainder else 0)

    print(f"[Rank {rank}] 总参数={total_params}, "
          f"本地分片=[{start}:{end}] ({end-start} 个参数, "
          f"占 {100*(end-start)/total_params:.1f}%)")
    results[rank] = (start, end, total_params)

def demo():
    world_size = 4
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    results = manager.dict()

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=train_rank, args=(rank, world_size, results))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    intervals = [results[r][:2] for r in range(world_size)]
    total = results[0][2]
    merged = sorted(intervals)
    assert merged[0][0] == 0 and merged[-1][1] == total
    for i in range(len(merged)-1):
        assert merged[i][1] == merged[i+1][0], "分片不连续!"
    print(f"\n✅ 分片完整覆盖 {total} 个参数, 无重叠")

if __name__ == "__main__":
    demo()

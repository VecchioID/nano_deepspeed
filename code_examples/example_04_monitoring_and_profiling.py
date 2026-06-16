"""
example_04_monitoring.py
DeepSpeed 性能监控与分析

演示目的:
    - Flops Profiler 使用
    - 通信分析
    - TensorBoard 集成
    - 显存跟踪
"""

import time
import torch
import deepspeed
from deepspeed.profiling.flops_profiler import FlopsProfiler


def monitor_basics(model_engine):
    """基本性能监控"""

    # GPU 显存
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory: {allocated:.1f}GB allocated, "
          f"{reserved:.1f}GB reserved")

    # DeepSpeed 引擎状态
    engine_info = {
        "stage": model_engine.zero_optimization_stage(),
        "dp_size": model_engine.data_parallel_size,
        "dp_rank": model_engine.data_parallel_rank,
        "global_batch": model_engine.train_batch_size(),
        "micro_batch": model_engine.train_micro_batch_size_per_gpu(),
    }
    print(f"Engine info: {engine_info}")


def profile_step(model_engine, batch):
    """使用 FlopsProfiler 分析单个 step"""

    profiler = FlopsProfiler(model_engine)

    # Warmup
    for _ in range(5):
        outputs = model_engine(**batch)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

    # Profile
    profiler.start_profile()
    outputs = model_engine(**batch)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()
    profiler.stop_profile()

    # 打印分析结果
    profiler.print_model_profile(profile_step=5)
    profiler.end_profile()

    return profiler


def communication_analysis(model_engine):
    """分析通信开销"""
    comm_stats = {}

    if hasattr(model_engine, 'gradient_allreduce'):
        allreduce_time = model_engine.gradient_allreduce.comm_time
        comm_stats["allreduce_time"] = allreduce_time

    # ZeRO-3 通信统计
    if model_engine.zero_optimization_stage() == 3:
        zero = model_engine.optimizer
        if hasattr(zero, 'all_gather_time'):
            comm_stats["allgather_time"] = zero.all_gather_time
        if hasattr(zero, 'reduce_scatter_time'):
            comm_stats["reduce_scatter_time"] = zero.reduce_scatter_time

    total = sum(comm_stats.values())
    if total > 0:
        for k, v in comm_stats.items():
            print(f"  {k}: {v:.3f}s ({v/total*100:.1f}%)")
        print(f"  Total comm time: {total:.3f}s")


def throughput_benchmark(model_engine, data_loader, num_steps=50):
    """吞吐量基准测试"""
    model_engine.eval()

    # Warmup
    for batch in data_loader:
        model_engine(**batch)
        break

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    step_times = []
    for step, batch in enumerate(data_loader):
        if step >= num_steps:
            break

        torch.cuda.synchronize()
        step_start = time.time()

        with torch.no_grad():
            outputs = model_engine(**batch)

        torch.cuda.synchronize()
        step_times.append(time.time() - step_start)

    total_time = time.time() - start
    avg_step_time = sum(step_times) / len(step_times)
    tokens_per_step = (
        model_engine.train_micro_batch_size_per_gpu()
        * batch["input_ids"].shape[1]
    )

    print(f"\nThroughput Results:")
    print(f"  Average step time: {avg_step_time*1000:.1f}ms")
    print(f"  Tokens per second: {tokens_per_step / avg_step_time:.0f}")
    print(f"  Samples per second: {1.0 / avg_step_time:.1f}")
    print(f"  Total benchmark time: {total_time:.1f}s")

    return {
        "avg_step_time_ms": avg_step_time * 1000,
        "tokens_per_second": tokens_per_step / avg_step_time,
        "samples_per_second": 1.0 / avg_step_time,
    }


def main():
    print("DeepSpeed Monitoring Examples")
    print("=" * 50)

    # 基准测试
    print("\n1. Throughput Benchmark:")
    print("   (需要实际的 DeepSpeed engine, 请参考 example_01)")

    print("\n2. Flops Profiler:")
    print("   deepspeed.profiling.flops_profiler.FlopsProfiler")

    print("\n3. Communication Analysis:")
    print("   - NCCL_DEBUG=INFO 环境变量")
    print("   - DeepSpeed engine 内置统计")

    print("\n4. Memory Tracking:")
    print("   - torch.cuda.memory_allocated()")
    print("   - torch.cuda.memory_reserved()")
    print("   - nvidia-smi dmon")

    print("\n5. TensorBoard Integration:")
    print("   'tensorboard': {'enabled': true, 'output_path': './logs'}")


if __name__ == "__main__":
    main()

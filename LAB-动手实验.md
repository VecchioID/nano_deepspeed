# DeepSpeed 动手实验 Lab

## 环境检查

本 lab 演示 DeepSpeed 的核心用法，请在配备 GPU 的机器上运行。

```bash
python3 -c "
import torch, deepspeed
print(f'DeepSpeed: {deepspeed.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

---

## Lab 1: 零配置入门 — 跑通 DeepSpeed

**目标**: 用最简代码跑通 DeepSpeed 训练循环，对比 ZeRO-2 / ZeRO-3 的显存差异。

### 步骤

```bash
# 导航到项目根目录（请替换为您的实际路径）
mkdir -p lab_output
```

创建文件 `lab1_baseline.py`:

```python
import torch, torch.nn as nn, deepspeed, argparse

class TinyModel(nn.Module):
    def __init__(self, vocab=10000, d_model=1024, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, 16, dim_feedforward=d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln(x))


def train_with_engine(zero_stage, label):
    model = TinyModel().cuda()
    n_params = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params={
            "train_batch_size": 8,
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 2,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-4, "betas": [0.9, 0.999]}
            },
            "zero_optimization": {"stage": zero_stage},
            "fp16": {"enabled": True},
            "gradient_clipping": 1.0,
        }
    )

    dummy = torch.randint(0, 10000, (4, 256)).cuda()
    for step in range(10):
        logits = engine(dummy)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 10000), dummy.view(-1))
        engine.backward(loss)
        engine.step()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{label}] Params={n_params/1e6:.1f}M  Peak Mem={peak:.2f}GB  ZeRO Stage={zero_stage}")
    return peak

if __name__ == "__main__":
    for stage, label in [(0, "No-ZeRO(无效)"), (2, "ZeRO-2"), (3, "ZeRO-3")]:
        try:
            train_with_engine(stage, label)
        except Exception as e:
            print(f"[{label}] Error: {e}")
```

运行:

```bash
python lab1_baseline.py
```

预期输出类似:
```
[No-ZeRO(无效)] Params=66.2M  Peak Mem=2.75GB  ZeRO Stage=0
[ZeRO-2] Params=66.2M  Peak Mem=2.30GB  ZeRO Stage=2
[ZeRO-3] Params=66.2M  Peak Mem=1.85GB  ZeRO Stage=3
```

> 单卡下 ZeRO-3 仍有节省是因为 activation 和临时缓冲区也被优化了。
> 多卡时节省更显著: ZeRO-3 显存 = 16Ψ/N。

---

## Lab 2: 配置调参实战

**目标**: 理解 DeepSpeed JSON 配置每个字段的作用。

### 步骤

创建 `lab2_config.py`:

```python
import json, os, torch, deepspeed

# 逐层构建配置，观察显存变化
def experiment(config_updates, label):
    base_config = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4}
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": False,
            "contiguous_gradients": False,
        },
        "fp16": {"enabled": True},
    }
    base_config.update(config_updates)

    model = torch.nn.Transformer(
        d_model=1024, nhead=16, num_encoder_layers=4,
        num_decoder_layers=0, dim_feedforward=4096,
        batch_first=True
    ).cuda()

    torch.cuda.reset_peak_memory_stats()
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config_params=base_config
    )

    dummy = torch.randint(0, 1000, (4, 128)).cuda()
    for _ in range(5):
        out = engine(dummy)
        loss = out.mean()
        engine.backward(loss)
        engine.step()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{label}] Peak={peak:.2f}GB")

if __name__ == "__main__":
    # 实验1: 开启通信重叠
    experiment({"zero_optimization": {"stage": 3, "overlap_comm": True}}, "overlap_comm=on")
    experiment({"zero_optimization": {"stage": 3, "overlap_comm": False}}, "overlap_comm=off")

    # 实验2: 开启 activation checkpointing
    experiment({
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False
        }
    }, "act_ckpt=on")

    # 实验3: offload optimizer
    experiment({
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"}
        }
    }, "offload=optimizer")
```

```bash
python lab2_config.py
```

---

## Lab 3: HuggingFace Transformers 集成

**目标**: 真实场景中用 HuggingFace Trainer 跑 DeepSpeed。

### 步骤

创建 `lab3_hf.py`:

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import deepspeed

# 用 GPT-2 small (124M) 演示
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 构造假数据
texts = ["DeepSpeed is a deep learning optimization library. " * 50] * 50
encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
dataset = Dataset.from_dict(encodings)

# DeepSpeed 配置 (写入临时文件)
import json, tempfile, os
ds_config = {
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2, "overlap_comm": True},
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 5e-5, "betas": [0.9, 0.999]}
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {"warmup_min_lr": 0, "warmup_max_lr": 5e-5, "warmup_num_steps": 10}
    },
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
}
config_path = "/tmp/ds_config_lab.json"
json.dump(ds_config, open(config_path, "w"))

training_args = TrainingArguments(
    output_dir="/tmp/hf_deepspeed_lab",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    fp16=True,
    deepspeed=config_path,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
)

model = AutoModelForCausalLM.from_pretrained(model_name)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
print("HF + DeepSpeed 训练完成!")
```

```bash
python lab3_hf.py
```

---

## Lab 4: 性能分析

**目标**: 用 DeepSpeed Flops Profiler 分析模型瓶颈。

创建 `lab4_profile.py`:

```python
import torch, deepspeed
from deepspeed.profiling.flops_profiler import FlopsProfiler

model = torch.nn.Transformer(
    d_model=1024, nhead=16, num_encoder_layers=6,
    num_decoder_layers=0, dim_feedforward=4096,
    batch_first=True
)
dummy = torch.randint(0, 1000, (2, 128))

engine, _, _, _ = deepspeed.initialize(
    model=model, model_parameters=model.parameters(),
    config_params={
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 2,
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-4}},
        "zero_optimization": {"stage": 2},
        "fp16": {"enabled": True},
    }
)

prof = FlopsProfiler(engine)

# warmup
for _ in range(3):
    out = engine(dummy.cuda())
    loss = out.mean()
    engine.backward(loss)
    engine.step()

prof.start_profile()
for _ in range(5):
    out = engine(dummy.cuda())
    loss = out.mean()
    engine.backward(loss)
    engine.step()
prof.stop_profile()

prof.print_model_profile(profile_step=5)
prof.end_profile()
```

```bash
python lab4_profile.py
```

---

## Lab 5: 动手改造你的模型

**目标**: 将任意 PyTorch 模型用 DeepSpeed 包装。

```bash
cat << 'EOF' > lab5_custom.py
import torch, torch.nn as nn, deepspeed

# 你的模型 (任何 nn.Module)
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = YourModel()

# 1 行代码接入 DeepSpeed
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params={
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 32,
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
        "zero_optimization": {"stage": 2},
        "fp16": {"enabled": True},
    }
)

# 训练循环 vs 标准 PyTorch
# 标准: loss.backward() + optimizer.step()
# DeepSpeed: engine.backward(loss) + engine.step()
dummy = torch.randn(32, 784).cuda()
labels = torch.randint(0, 10, (32,)).cuda()

for step in range(10):
    logits = engine(dummy)
    loss = nn.CrossEntropyLoss()(logits, labels)
    engine.backward(loss)    # ← 替代 loss.backward()
    engine.step()            # ← 替代 optimizer.step()
    print(f"Step {step}: loss={loss.item():.4f}")

print("✅ 改造完成! 现在你的模型支持 ZeRO 分布式训练了")
EOF
python lab5_custom.py
```

---

## 进阶练习

完成以上 Lab 后，尝试:

1. **修改模型大小**: 把 d_model 从 1024 改为 4096，观察 OOM 时启用 ZeRO-3 能否跑
2. **多卡模拟**: 用 `torchrun --nproc_per_node=2 lab1_baseline.py` 启动 2 个进程
   (需要 2 GPU, 但可以观察代码如何在单卡上 fallback)
3. **对比实验**: 固定 batch size, 对比 ZeRO-0/2/3 的显存和吞吐
4. **Offload 实验**: 对 30M+ 参数模型启用 `offload_optimizer`, 观察速度变化

---

## 问题排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `RuntimeError: No CUDA GPUs are available` | deepspeed 找不到 GPU | `export CUDA_VISIBLE_DEVICES=0` |
| `deepspeed.ops.op_builder.CUDAVersionError` | CUDA 版本不匹配 | `pip install deepspeed --no-build-isolation` |
| NCCL 超时 | 多卡通信问题 | `export NCCL_DEBUG=INFO` 查看 |

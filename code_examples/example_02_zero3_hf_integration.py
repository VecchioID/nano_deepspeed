"""
example_02_zero3_config.py
ZeRO-3 配置与 HuggingFace 集成示例

用法:
    deepspeed --num_gpus=4 example_02_zero3_config.py

演示目的:
    - ZeRO-3 完整配置
    - HuggingFace Transformers 集成
    - Activation Checkpointing
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset


def get_dummy_dataset(tokenizer, num_samples=100, max_length=512):
    """生成模拟的预训练数据"""
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 100
        for _ in range(num_samples)
    ]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    dataset = Dataset.from_dict(encodings)
    return dataset


def main():
    # ZeRO-3 完整配置文件 (也支持 JSON 文件传入)
    ds_config = {
        "train_batch_size": 64,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 8,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 100,
                "total_num_steps": 1000,
            },
        },
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e8,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
        },
        "gradient_clipping": 1.0,
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
        },
    }

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./zero3_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=100,
        num_train_epochs=1,
        fp16=True,
        deepspeed=ds_config,  # 直接传 dict 或 JSON 文件路径
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    # 加载 tokenizer 和模型 (ZeRO-3 下模型自动分片)
    model_name = "gpt2"  # 小模型用于演示
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 准备数据
    dataset = get_dummy_dataset(tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # 创建 Trainer (内部自动集成 DeepSpeed)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    trainer.save_model("./final_model")


if __name__ == "__main__":
    main()

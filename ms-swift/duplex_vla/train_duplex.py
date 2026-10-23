#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工 VLA 训练入口 (B 方案: 自定义 id 级 Dataset + transformers Trainer).

与现有代码零冲突: 不动 mcvla/、swift、任何现有模型目录; 全部产物在 --output.

用法 (GPU 机 swift 环境, 8 卡):
  torchrun --nproc_per_node 8 train_duplex.py \
      --model HF_MODEL/JarvisVLA-Qwen2-VL-7B-duplex \
      --dataset dataset/duplex_vla/duplex_turns_train.jsonl \
      --output <ckpt目录> --max-steps 362   # 92,781/(8*8*4)≈362 step/epoch

前置: make_duplex_base.py 已生成扩词表副本.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import Qwen2VLForConditionalGeneration, Trainer, TrainingArguments

HERE = os.path.dirname(os.path.abspath(__file__))
DS_JSONL_DEFAULT = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                    "chengzhixu/dataset/duplex_vla/duplex_turns_train.jsonl")
FRAMES_DEFAULT = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                  "chengzhixu/dataset/mcvla_sft/frames")


class DuplexTrainer(Trainer):
    """标准 CE + 预留自定义 loss 落点.

    act_span/txt_span 从 batch pop 出来 (模型 forward 不吃), 供后续
    自定义 compute_loss (双通道加权/act 受限 CE 等) 直接使用.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        act_span = inputs.pop("act_span", None)
        txt_span = inputs.pop("txt_span", None)
        outputs = model(**inputs)
        loss = outputs.loss  # 标准_shifted_ CE, 仅 labels 非 -100 位置 (目标段)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="扩词表副本目录 (make_duplex_base 产物)")
    ap.add_argument("--dataset", default=DS_JSONL_DEFAULT)
    ap.add_argument("--frames-root", default=FRAMES_DEFAULT)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-steps", type=int, default=362,
                    help="92,781 / (8卡x8x4=256) ≈ 362 = 1 epoch (streaming 必须)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per-device-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    ap.add_argument("--shuffle-buffer", type=int, default=4096)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="冒烟: 只跑 2 step")
    ap.add_argument("--report-to", default="none", choices=["none", "wandb"],
                    help="指标上报后端 (default: none)")
    ap.add_argument("--run-name", default=None,
                    help="wandb run 名 (默认: wandb 自动生成)")
    ap.add_argument("--wandb-project", default="mcvla-duplex",
                    help="wandb project 名")
    args = ap.parse_args()

    # wandb 需走公司代理 (在 Trainer/wandb import 前设置)
    if args.report_to == "wandb":
        os.environ.setdefault("http_proxy", "http://star-proxy.oa.com:3128")
        os.environ.setdefault("https_proxy", "http://star-proxy.oa.com:3128")
        os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    from duplex_dataset import DuplexStreamingDataset
    from duplex_collator import duplex_collate

    ds = DuplexStreamingDataset(args.dataset, args.frames_root, args.model,
                                shuffle_buffer=args.shuffle_buffer, seed=args.seed)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    # 冻结视觉塔 (patch embed + blocks + merger/aligner), 等效 freeze_vit+freeze_aligner
    # 注意: 该版 transformers 中视觉塔路径是 model.model.visual (旧版有 .visual 属性转发)
    visual = getattr(model, "visual", None) or model.model.visual
    for p in visual.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    if not args.smoke:
        # use_reentrant=False: 冻结视觉塔时 reentrant 模式可能断梯度 (多模态训练标准配置)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    targs = TrainingArguments(
        output_dir=args.output,
        max_steps=2 if args.smoke else args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.run_name,
        remove_unused_columns=False,          # 自定义 Dataset 必须关
        dataloader_num_workers=args.workers,
        # 各 rank 样本长度不一 (padding 后 max_len 不同), accelerate 默认
        # dispatch_batches=True 分发同形 batch 会崩, 必须各进程自取
        accelerator_config={"dispatch_batches": False},
        deepspeed=os.path.join(HERE, "ds_zero2.json"),
        adam_beta1=0.9,
        adam_beta2=0.95,        # 与 ds_zero2.json 的 optimizer betas 对齐
        seed=args.seed,
        max_grad_norm=1.0,
    )

    trainer = DuplexTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=duplex_collate,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output, "final"))
    print("训练完成: %s" % args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU 前向冒烟: 副本模型 + collator + labels -> loss (不做反向/GPU 特性验证)."""
import sys

sys.path.insert(0, "/apdcephfs/private_chengzhixu/ms-swift/duplex_vla")
import torch

from duplex_dataset import DuplexMapDataset
from duplex_collator import duplex_collate

MODEL = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
         "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B-duplex")

print("加载数据 (2条)...", flush=True)
ds = DuplexMapDataset(
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/duplex_vla/duplex_turns_train.jsonl",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames",
    MODEL, max_samples=200)
# 取一条普通轮 + 一条 QA 轮
items = [ds[0], ds[3]]
for i, it in enumerate(items):
    if it["txt_span"][1] - it["txt_span"][0] > 2:
        items = [it, items[0]]
        break
batch = duplex_collate(items)
print("batch: input_ids %s, labels有效 %d, pixel_values %s" % (
    tuple(batch["input_ids"].shape),
    (batch["labels"] != -100).sum().item(),
    tuple(batch["pixel_values"].shape)), flush=True)

print("加载模型 (CPU bf16)...", flush=True)
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16)
model.eval()

print("前向...", flush=True)
with torch.no_grad():
    fwd = {k: batch[k] for k in ("input_ids", "labels", "attention_mask",
                                 "mm_token_type_ids", "pixel_values",
                                 "image_grid_thw")}
    out = model(**fwd)
print("loss = %.4f  logits %s" % (out.loss.item(), tuple(out.logits.shape)))
assert torch.isfinite(out.loss), "loss 非有限值!"
print()
print("CPU 冒烟通过: 副本模型+collator+labels 前向链路正常, loss 有限 ✓")
print("(GPU 反向/deepspeed 冒烟待 stage2 训练结束后跑 train_duplex.py --smoke)")

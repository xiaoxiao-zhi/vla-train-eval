#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工训练 collator: batch padding + 视觉张量拼接.

输入: DuplexStreamingDataset/MapDataset 产出的样本列表, 每条:
  {input_ids, labels, pixel_values, image_grid_thw, act_span, txt_span, n_images}
输出 (喂 Qwen2VLForConditionalGeneration):
  input_ids       (B, L_max)  right pad, pad_id=151643 (<|endoftext|>)
  labels          (B, L_max)  pad -100 (右 padding 不影响 span 位置索引)
  attention_mask  (B, L_max)
  pixel_values    (Σpatches, C)  各样本 cat
  image_grid_thw  (Σframes, 3)
  act_span/txt_span: list[(lo,hi)]  保留原位 (供自定义 loss 用)
"""
import torch

PAD_ID = 151643  # <|endoftext|>, Qwen 系标准 pad


def duplex_collate(batch):
    B = len(batch)
    L = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    mm_types = torch.zeros((B, L), dtype=torch.long)   # pad 位置 0 (文本), 被 attn mask 屏蔽
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
        mm_types[i, :n] = torch.tensor(b["mm_token_type_ids"], dtype=torch.long)
        attn[i, :n] = 1
    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
    grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "mm_token_type_ids": mm_types,
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
        "act_span": [b["act_span"] for b in batch],
        "txt_span": [b["txt_span"] for b in batch],
    }

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练拟合检查: 抽 N 条训练样本, 相同输入下模型 greedy 生成 vs 训练标签对照.

用法 (GPU 机):
  python check_trainfit.py --ckpt <checkpoint> --n 10
"""
import argparse
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from duplex_dataset import DuplexIndexedDataset  # noqa: E402
from duplex_collator import duplex_collate  # noqa: E402

DS_JSONL = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
            "chengzhixu/dataset/duplex_vla/duplex_turns_train.jsonl")
FRAMES = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
          "chengzhixu/dataset/mcvla_sft/frames")
BASE_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
            "chengzhixu/HF_MODEL/JarvisVLA-s2-duplex")   # tokenizer 来源, 与训练一致


def rd(text):
    """保留字文本 -> 可读编号序列."""
    ns = re.findall(r'reserved_special_token_(\d+)', text or '')
    return ' '.join(ns) if ns else (text or '')[:50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/duplex_v1/checkpoint-362')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--stride', type=int, default=977)
    args = ap.parse_args()

    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model.eval()

    # 注意: checkpoint-362 保存的 tokenizer 有问题(image_pad->0),
    # dataset/renderer 用基座目录的好 tokenizer; 模型权重仍用 checkpoint
    base_dir = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
                'chengzhixu/HF_MODEL/JarvisVLA-s2-duplex')
    tok = AutoTokenizer.from_pretrained(base_dir)
    ds = DuplexIndexedDataset(DS_JSONL, FRAMES, base_dir)
    n_total = len(ds)
    idxs = list(range(0, n_total, args.stride))[:args.n]
    print('数据 %d 条, 检查 %d 条' % (n_total, len(idxs)), flush=True)

    exact = 0
    for i in idxs:
        # 与训练完全相同的输入路径: DuplexIndexedDataset -> duplex_collate -> model(**batch)
        batch = duplex_collate([ds[i]])
        batch.pop('act_span', None); batch.pop('txt_span', None)
        labels = batch.pop('labels')[0]
        fwd_in = {k: (v.to(model.device) if torch.is_tensor(v) else v)
                  for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**fwd_in).logits[0]        # [L, V], 与训练同一 forward
        pos = (labels != -100).nonzero().squeeze(-1)
        pred_ids = logits[pos - 1].argmax(-1).cpu()   # shift 对齐: logits[p-1] 预测 labels[p]
        gold_ids = labels[pos]
        match = bool((pred_ids == gold_ids).all())
        exact += match
        pred_text = tok.decode(pred_ids, skip_special_tokens=False)
        gold_text = tok.decode(gold_ids, skip_special_tokens=False)
        n_ok = int((pred_ids == gold_ids).sum())
        print('\n===== 样本 %d  token命中 %d/%d %s =====' % (
            i, n_ok, len(gold_ids), '[完全一致]' if match else ''))
        print('GOLD: %s' % rd(gold_text))
        print('PRED: %s' % rd(pred_text))

    print('\n总体: %d/%d 条完全一致 (%.0f%%)' % (exact, len(idxs), 100*exact/len(idxs)))


if __name__ == '__main__':
    main()

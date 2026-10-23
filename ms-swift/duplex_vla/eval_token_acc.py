#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线评估 duplex checkpoint 的 token acc (action 段 / text 段分开统计).

用法 (GPU 机):
  python eval_token_acc.py --ckpt checkpoint/duplex_v1/checkpoint-362 --n 100
"""
import argparse
import json
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/duplex_v1/checkpoint-362')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--stride', type=int, default=50, help='抽样的样本间隔')
    args = ap.parse_args()

    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map='cuda:4')
    model.eval()

    ds = DuplexIndexedDataset(DS_JSONL, FRAMES, args.ckpt, shuffle_buffer=0, seed=0)
    n_total = len(ds)
    idxs = list(range(0, n_total, args.stride))[:args.n]
    print('数据 %d 条, 评估 %d 条 (stride=%d)' % (n_total, len(idxs), args.stride), flush=True)

    n_act = n_act_ok = n_txt = n_txt_ok = 0
    with torch.no_grad():
        for k, i in enumerate(idxs):
            batch = duplex_collate([ds[i]])
            act_span = batch.pop('act_span', None)
            txt_span = batch.pop('txt_span', None)
            batch = {k2: v.to(model.device) if torch.is_tensor(v) else v
                     for k2, v in batch.items()}
            logits = model(**batch).logits[0]          # [L, V]
            labels = batch['labels'][0]                 # [L]
            pred = logits.argmax(-1)
            # shifted 对齐: 位置 p 预测 labels[p]
            mask = labels != -100
            L = labels.shape[0]
            pos = mask.nonzero().squeeze(-1)
            ok_all = (pred[pos] == labels[pos])
            # act/txt 段划分
            if act_span is not None:
                a0, a1 = act_span[0]
                in_act = (pos >= a0) & (pos < a1)
                n_act += int(in_act.sum()); n_act_ok += int(ok_all[in_act].sum())
            if txt_span is not None:
                t0, t1 = txt_span[0]
                in_txt = (pos >= t0) & (pos < t1)
                n_txt += int(in_txt.sum()); n_txt_ok += int(ok_all[in_txt].sum())
            if (k+1) % 20 == 0:
                print('  %d/%d  act_acc=%.3f txt_acc=%.3f' % (
                    k+1, len(idxs),
                    n_act_ok/max(n_act,1), n_txt_ok/max(n_txt,1)), flush=True)

    print()
    print('===== token acc (checkpoint-362, teacher forcing) =====')
    print('action 段: %d/%d = %.4f' % (n_act_ok, n_act, n_act_ok/max(n_act,1)))
    print('text   段: %d/%d = %.4f' % (n_txt_ok, n_txt, n_txt_ok/max(n_txt,1)))
    print('总体:     %d/%d = %.4f' % (n_act_ok+n_txt_ok, n_act+n_txt,
          (n_act_ok+n_txt_ok)/max(n_act+n_txt,1)))


if __name__ == '__main__':
    main()

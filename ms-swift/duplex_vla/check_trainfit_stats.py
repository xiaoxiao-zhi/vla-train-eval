#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练拟合分通道统计: 100 条样本, 动作段/文本段分开统计 token 差异.

用法 (GPU 机): python check_trainfit_stats.py --n 100
"""
import argparse
import os
import re
import sys
from collections import Counter

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
            "chengzhixu/HF_MODEL/JarvisVLA-s2-duplex")

NAME = {176:'BEGIN',178:'BEGIN',179:'END',180:'hotbar.0',181:'hotbar.1',182:'hotbar.2',
        183:'hotbar.3',184:'hotbar.4',185:'hotbar.5',186:'hotbar.6',187:'hotbar.7',
        188:'hotbar.8',189:'hotbar.9',191:'forward',192:'back',193:'left',194:'right',
        196:'sprint0',197:'sprint',198:'sneak',199:'use?',203:'atk0',204:'attack',
        205:'jmp0',206:'jump',207:'camY?',208:'camY'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/duplex_v1/checkpoint-362')
    ap.add_argument('--n', type=int, default=100)
    args = ap.parse_args()

    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model.eval()
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    ds = DuplexIndexedDataset(DS_JSONL, FRAMES, BASE_DIR)
    stride = max(1, len(ds) // args.n)
    idxs = list(range(0, len(ds), stride))[:args.n]
    print('数据 %d 条, 检查 %d 条 (stride=%d)' % (len(ds), len(idxs), stride), flush=True)

    # 分通道累计
    act = {'n': 0, 'ok': 0, 'errs': Counter()}      # errs: (gold,pred) 对
    txt = {'n': 0, 'ok': 0, 'perfect': 0, 'samples': []}
    for k, i in enumerate(idxs):
        batch = duplex_collate([ds[i]])
        act_span = batch.pop('act_span')[0]
        txt_span = batch.pop('txt_span')[0]
        labels = batch.pop('labels')[0]
        fwd = {kk: (v.to(model.device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
        with torch.no_grad():
            logits = model(**fwd).logits[0]
        pos = (labels != -100).nonzero().squeeze(-1)
        pred = logits[pos - 1].argmax(-1).cpu()
        gold = labels[pos]
        ok = (pred == gold)
        for (lo, hi) in (act_span if isinstance(act_span[0], (list, tuple)) else [act_span]):
            m = (pos >= lo) & (pos < hi)
            act['n'] += int(m.sum()); act['ok'] += int(ok[m].sum())
            for p_, g_ in zip(pred[m].tolist(), gold[m].tolist()):
                if p_ != g_:
                    act['errs'][(g_, p_)] += 1
        for (lo, hi) in (txt_span if isinstance(txt_span[0], (list, tuple)) else [txt_span]):
            m = (pos >= lo) & (pos < hi)
            txt['n'] += int(m.sum()); txt['ok'] += int(ok[m].sum())
            if bool(ok[m].all()):
                txt['perfect'] += 1
            elif len(txt['samples']) < 3:
                txt['samples'].append((
                    tok.decode(gold[m], skip_special_tokens=False)[:60],
                    tok.decode(pred[m], skip_special_tokens=False)[:60]))
        if (k+1) % 25 == 0:
            print('  %d/%d  act %.3f txt %.3f' % (k+1, len(idxs),
                act['ok']/max(act['n'],1), txt['ok']/max(txt['n'],1)), flush=True)

    print('\n========== 动作段 ==========')
    print('token 命中: %d/%d = %.4f' % (act['ok'], act['n'], act['ok']/max(act['n'],1)))
    print('错误对 top10 (gold -> pred):')
    for (g, p), c in act['errs'].most_common(10):
        print('  %s(%s) -> %s(%s) : %d 次' % (
            g, NAME.get(g, '?'), p, NAME.get(p, '?'), c))

    print('\n========== 文本段 ==========')
    print('token 命中: %d/%d = %.4f' % (txt['ok'], txt['n'], txt['ok']/max(txt['n'],1)))
    print('完全一致条数: %d/%d' % (txt['perfect'], args.n))
    for g, p in txt['samples']:
        print('  GOLD: %s' % g)
        print('  PRED: %s' % p)


if __name__ == '__main__':
    main()

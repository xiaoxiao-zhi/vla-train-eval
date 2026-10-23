#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""专门测有真实 QA 文本(非 silence)的样本: 50 条, 文本通道 teacher-forcing 对照."""
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
BASE_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
            "chengzhixu/HF_MODEL/JarvisVLA-s2-duplex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/duplex_v1/checkpoint-362')
    ap.add_argument('--n', type=int, default=50)
    args = ap.parse_args()

    # silence id (duplex_tokens.json)
    tokens = json.load(open(os.path.join(HERE, 'duplex_tokens.json')))
    silence_id = tokens['ids']['<|silence|>']
    print('silence_id =', silence_id, flush=True)

    # 找 txt_ids 非 silence 的行号 (QA answer 轮)
    qa_idx = []
    with open(DS_JSONL) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if d['target'].get('txt_ids') and d['target']['txt_ids'] != [silence_id]:
                qa_idx.append(i)
    print('全量 %d 行中 QA 文本活跃行: %d' % (i+1, len(qa_idx)), flush=True)
    step = max(1, len(qa_idx) // args.n)
    picks = qa_idx[::step][:args.n]
    print('测 %d 条 (stride=%d)' % (len(picks), step), flush=True)

    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model.eval()
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    ds = DuplexIndexedDataset(DS_JSONL, FRAMES, BASE_DIR)

    n_ok = n_tot = perfect = 0
    shown = 0
    for k, i in enumerate(picks):
        batch = duplex_collate([ds[i]])
        batch.pop('act_span', None)
        txt_span = batch.pop('txt_span')[0]
        labels = batch.pop('labels')[0]
        fwd = {kk: (v.to(model.device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
        with torch.no_grad():
            logits = model(**fwd).logits[0]
        pos = (labels != -100).nonzero().squeeze(-1)
        pred = logits[pos - 1].argmax(-1).cpu()
        gold = labels[pos]
        ok = (pred == gold)
        # 文本段范围
        lo, hi = txt_span if isinstance(txt_span[0], int) else txt_span[0]
        m = (pos >= lo) & (pos < hi)
        n_ok += int(ok[m].sum()); n_tot += int(m.sum())
        if bool(ok[m].all()):
            perfect += 1
        elif shown < 4:
            shown += 1
            g = tok.decode(gold[m], skip_special_tokens=False)
            p = tok.decode(pred[m], skip_special_tokens=False)
            print('\n[失配 %d] 行 %d' % (shown, i))
            print('  GOLD: %s' % g[:90])
            print('  PRED: %s' % p[:90])
        if (k+1) % 10 == 0:
            print('  %d/%d  txt_acc=%.3f perfect=%d' % (
                k+1, len(picks), n_ok/max(n_tot,1), perfect), flush=True)

    print('\n===== QA 文本通道 (真实回答轮) =====')
    print('token 命中: %d/%d = %.4f' % (n_ok, n_tot, n_ok/max(n_tot,1)))
    print('完全一致: %d/%d 条 (%.0f%%)' % (perfect, len(picks), 100*perfect/len(picks)))


if __name__ == '__main__':
    main()

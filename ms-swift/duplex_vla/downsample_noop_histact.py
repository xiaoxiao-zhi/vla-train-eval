#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""histact 数据 no-op 下采样 (同 downsample_noop_v4 语义, 仅换数据路径).

target = no-op动作 + <|silence|> 的样本只保留 5%.
被删帧不作为学习目标, 但保留在其他样本的 prompt 历史里.
输入: duplex_vla_v4_histact/duplex_train.jsonl
输出: duplex_vla_v4_histact/duplex_train_filtered.jsonl
(原 v4 数据不受影响)
"""
import json
import random
import re

SRC = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/duplex_vla_v4_histact/duplex_train.jsonl')
DST = SRC.replace('duplex_train.jsonl', 'duplex_train_filtered.jsonl')

ACT_RE = re.compile(r'<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>', re.S)
NOP = ('<|reserved_special_token_178|><|reserved_special_token_219|>'
       '<|reserved_special_token_240|><|reserved_special_token_179|>')
KEEP_RATE = 0.05

random.seed(42)
nop_pool = []
kept_other = 0
with open(DST, 'w') as out:
    for line in open(SRC):
        d = json.loads(line)
        tgt = d['messages'][1]['content']
        m = ACT_RE.search(tgt)
        is_nop_sil = (m is not None and m.group(0) == NOP
                      and tgt[m.end():].strip() == '<|silence|>')
        if is_nop_sil:
            nop_pool.append(line)
        else:
            out.write(line)
            kept_other += 1

n_keep = int(len(nop_pool) * KEEP_RATE)
keep_ids = set(random.sample(range(len(nop_pool)), n_keep))
kept_nop = 0
with open(DST, 'a') as f:
    for i, line in enumerate(nop_pool):
        if i in keep_ids:
            f.write(line)
            kept_nop += 1

print('no-op+silence: %d -> 保留 %d (5%%)' % (len(nop_pool), kept_nop))
print('其他: %d (全保留)' % kept_other)
print('输出: %d 条 -> %s' % (kept_other + kept_nop, DST))

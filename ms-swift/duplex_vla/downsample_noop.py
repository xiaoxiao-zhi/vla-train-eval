#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下采样 duplex 训练数据: target = no-op动作 + <|silence|> 的样本只保留 5%.

其余样本(no-op+回答 / 非no-op)全部保留. 固定种子可复现.
输出: duplex_train_filtered.jsonl (同目录)
"""
import json
import random
import re

SRC = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/duplex_vla_v3/duplex_train.jsonl')
DST = SRC.replace('duplex_train.jsonl', 'duplex_train_filtered.jsonl')

ACT_RE = re.compile(r'<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>', re.S)
NOP = '<|reserved_special_token_178|><|reserved_special_token_219|><|reserved_special_token_240|><|reserved_special_token_179|>'
KEEP_RATE = 0.05

random.seed(42)
kept_nop = kept_other = 0
rng_kept = []
out = open(DST, 'w')
# 两遍式: 先全部写出非no-op, no-op样本缓存后按比例保留 (保证顺序无所谓的随机性)
nop_pool = []
for line in open(SRC):
    d = json.loads(line)
    tgt = d['messages'][-1]['content']
    m = ACT_RE.search(tgt)
    is_nop_sil = (m is not None and m.group(0) == NOP
                  and tgt[m.end():].strip() == '<|silence|>')
    if is_nop_sil:
        nop_pool.append(line)
    else:
        out.write(line)
        kept_other += 1
out.close()

n_keep = int(len(nop_pool) * KEEP_RATE)
for line in random.sample(nop_pool, n_keep):
    pass  # sample 打乱顺序, 重新写回 (下面用排序保持原文件顺序感)
keep_ids = set(random.sample(range(len(nop_pool)), n_keep))
with open(DST, 'a') as f:
    for i, line in enumerate(nop_pool):
        if i in keep_ids:
            f.write(line)
            kept_nop += 1

total = kept_other + kept_nop
print('no-op+silence 样本: %d -> 保留 %d (5%%)' % (len(nop_pool), kept_nop))
print('其他样本: %d (全保留)' % kept_other)
print('输出: %d 条 -> %s' % (total, DST))

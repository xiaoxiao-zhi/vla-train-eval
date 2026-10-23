#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照实验数据: v4 全双工 -> 仅动作 target (输入不变).

规则 (与 duplex_train_filtered.jsonl 的最小 diff):
  - user content (输入) 完全不变: 指令/silence/QA问题/帧占位/历史轮
  - assistant content 只保留动作段 178..179, 去掉尾部 "\n<|silence|>" / "\n回答文本"
  - 其余字段 (base/t/images/qa_meta) 原样保留

输出: dataset/duplex_vla_v4_actonly/duplex_train_filtered.jsonl
"""
import json
import os
import re

SRC = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/duplex_vla_v4/duplex_train_filtered.jsonl')
OUT_DIR = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/duplex_vla_v4_actonly')
ACT_RE = re.compile(r'(<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>)(.*)', re.S)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, 'duplex_train_filtered.jsonl')

n_in = n_out = 0
bad = 0
with open(out_path, 'w') as fo:
    for line in open(SRC):
        d = json.loads(line)
        a = d['messages'][1]['content']
        m = ACT_RE.match(a)
        if not m:
            bad += 1
            continue  # 原则: 无动作段的行剔除并计数 (不应出现, 见校验)
        d['messages'][1]['content'] = m.group(1)
        fo.write(json.dumps(d, ensure_ascii=False) + '\n')
        n_out += 1
        n_in += 1

print('输入 %d 行, 输出 %d 行, 无动作段剔除 %d' % (n_in, n_out, bad))
assert bad == 0, '存在无动作段样本, 需人工检查'

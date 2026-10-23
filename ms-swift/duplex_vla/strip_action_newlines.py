#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""去掉 assistant 动作 token 后的两个换行: '179|>\\n\\n' -> '179|>'.

纯 jsonl 字符串改写 (不重跑 tokenization), stage2 两份 + duplex 一份.
user 轮 (\\nobservation: \\n<image>) 不含该模式, 不受影响.
"""
import json

BASE = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
        'chengzhixu/dataset/duplex_vla_v3')
PAT = '<|reserved_special_token_179|>\\n\\n'  # jsonl 内为转义形式 (字面 \n 两个字符)
REP = '<|reserved_special_token_179|>'

for name in ['stage2_train_hist0.jsonl', 'stage2_train_hist1.jsonl', 'duplex_train.jsonl']:
    src = '%s/%s' % (BASE, name)
    tmp = src + '.tmp'
    n_line = n_fix = 0
    with open(src) as f, open(tmp, 'w') as out:
        for line in f:
            if PAT in line:
                line = line.replace(PAT, REP)
                n_fix += 1
            out.write(line)
            n_line += 1
    import os
    os.replace(tmp, src)
    print('%s: %d 行, 修改 %d 行' % (name, n_line, n_fix), flush=True)
print('done')

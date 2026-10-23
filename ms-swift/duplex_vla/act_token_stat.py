#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 v4 全双工数据 target 动作 token 出现次数总览表.

target = 动作段 (178...179) + 文本. 统计每个动作保留字 token 的出现次数.
含义参考:
  178/179 动作起止 | 180-189 hotbar1-9 | 190/191/192 无/forward/back
  193/194/195 无/left/right | 196/197/198 无/sprint/sneak | 199/200 无/use
  201/202 无/drop | 203/204 无/attack | 205/206 无/jump | 176/177 无/inventory
  209-229 pitch 21档(219=中) | 230-250 yaw 21档(240=中)
"""
import json
import re
from collections import Counter

SRC = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/duplex_vla_v4/duplex_train_filtered.jsonl')
OUT = '/apdcephfs/private_chengzhixu/act_token_stat.out'

TOKEN_RE = re.compile(r'<\|reserved_special_token_(\d+)\|>')

MEANING = {}
MEANING.update({178: '动作开始', 179: '动作结束'})
MEANING.update({i: 'hotbar%d' % (i - 179) for i in range(180, 190)})
MEANING.update({190: '(移动)无', 191: 'forward', 192: 'back'})
MEANING.update({193: '(平移)无', 194: 'left', 195: 'right'})
MEANING.update({196: '(姿态)无', 197: 'sprint', 198: 'sneak'})
MEANING.update({199: '(使用)无', 200: 'use'})
MEANING.update({201: '(丢弃)无', 202: 'drop'})
MEANING.update({203: '(攻击)无', 204: 'attack'})
MEANING.update({205: '(跳跃)无', 206: 'jump'})
MEANING.update({176: '(背包)无', 177: 'inventory'})
MEANING.update({209 + i: 'pitch_%02d/20' % i for i in range(21)})
MEANING.update({230 + i: 'yaw_%02d/20' % i for i in range(21)})

n = 0
c = Counter()
ACT_RE = re.compile(r'<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>', re.S)
for line in open(SRC):
    d = json.loads(line)
    tgt = d['messages'][1]['content']
    m = ACT_RE.search(tgt)
    if m is None:
        continue
    n += 1
    for t in TOKEN_RE.findall(m.group(0)):
        c[int(t)] += 1

total = sum(c.values())
lines = ['样本数: %d, 动作 token 总数: %d' % (n, total), '']
lines.append('%-6s %-14s %10s %7s' % ('token', '含义', '次数', '占比'))
lines.append('-' * 42)
for tid in sorted(c):
    lines.append('%-6d %-14s %10d %6.2f%%' % (tid, MEANING.get(tid, '?'), c[tid], 100 * c[tid] / total))
open(OUT, 'w').write('\n'.join(lines))
print('done -> %s' % OUT)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一致性验证: histact 剥掉历史动作后 == v4 (同 base+t 逐字段).

对每个抽样样本 (base, t) 验证:
  1. histact 的 user 中, 把窗口内历史帧的 动作段\n 去掉后 == v4 的 user
  2. images / qa_meta / target(assistant) 完全相等
窗口判定: images 数 n_img, 历史帧 = t-IMG_WINDOW..t-1 中 >=0 的; 其中
窗口内(带图)帧的 assistant = 动作+\n+文本, 窗口外帧 = 仅文本.
"""
import json
import re
import sys

V4 = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
      'chengzhixu/dataset/duplex_vla_v4/duplex_train_filtered.jsonl')
HA = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
      'chengzhixu/dataset/duplex_vla_v4_histact/duplex_train_filtered.jsonl')

ACT_SEG = re.compile(r'<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>\n')
IMG_WINDOW = 2

N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 200
import random
random.seed(0)

# 1. 从 v4 抽样 (含段首/普通/QA帧)
picked = {}
for line in open(V4):
    d = json.loads(line)
    key = (d['base'], d['t'])
    if key not in picked and random.random() < 0.002:
        picked[key] = d
    if len(picked) >= N_SAMPLE:
        break

# 2. 从 histact 找对应样本
want = set(picked)
match = {}
for line in open(HA):
    d = json.loads(line)
    key = (d['base'], d['t'])
    if key in want and key not in match:
        match[key] = d
        if len(match) == len(want):
            break

# 3. 逐字段验证
n_ok = n_bad = 0
for key, v4d in picked.items():
    had = match.get(key)
    if had is None:
        print('!! 未找到对应:', key); n_bad += 1; continue
    # images / qa_meta / target 必须完全相等
    if v4d['images'] != had['images'] or v4d['qa_meta'] != had['qa_meta'] \
       or v4d['messages'][1]['content'] != had['messages'][1]['content']:
        print('!! 字段不一致:', key); n_bad += 1; continue
    # user: histact 去掉窗口内历史动作后 == v4 user
    n_img = len(v4d['images'])
    hist_win_start = max(0, v4d['t'] - IMG_WINDOW)
    stripped = ACT_SEG.sub('', had['messages'][0]['content'])
    if stripped != v4d['messages'][0]['content']:
        print('!! user 不一致 after strip:', key)
        print('   v4 :', repr(v4d['messages'][0]['content'][:120]))
        print('   ha :', repr(stripped[:120]))
        n_bad += 1; continue
    n_ok += 1

print('抽样 %d: 一致 %d, 不一致 %d' % (len(picked), n_ok, n_bad))
assert n_bad == 0, '存在不一致样本!'
print('PASS: histact == v4 + 历史动作 (仅此差异)')

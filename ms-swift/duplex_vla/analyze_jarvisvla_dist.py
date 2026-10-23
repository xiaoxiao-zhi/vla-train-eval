#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JarvisVLA 原始 SFT 数据 (minecraft-vla-sft/train_text.jsonl, 377万) 动作分布统计.

输出: 总样本数 / no-op率 / 按键逐类频次 / camera 双轴逐档频次 / 任务类型分布.
"""
import json
import re
from collections import Counter

F = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
     'chengzhixu/dataset/minecraft-vla-sft/train_text.jsonl')
TOK_RE = re.compile(r'<\|reserved_special_token_(\d+)\|>')
BTN = {191: 'forward', 192: 'back', 194: 'left', 195: 'right', 197: 'sprint',
       198: 'sneak', 200: 'use', 202: 'drop', 204: 'attack', 206: 'jump',
       189: 'hotbar'}

btn_c = Counter()
cx_c, cy_c = Counter(), Counter()
task_c = Counter()
n = n_noop = n_nocam = n_noact = 0
for line in open(F):
    d = json.loads(line)
    n += 1
    # 任务类型 (label[2] 如 mine_block:xxx)
    task = d['label'][2].split(':')[0] if len(d['label']) > 2 else '?'
    task_c[task] += 1
    nums = [int(x) for x in TOK_RE.findall(d['action'])]
    inner = [x for x in nums if 180 <= x <= 250]
    btns = [x for x in inner if 180 <= x <= 206]
    cams = [x for x in inner if 209 <= x <= 250]
    cx = [x - 209 - 10 for x in cams if 209 <= x <= 229]
    cy = [x - 230 - 10 for x in cams if 230 <= x <= 250]
    if not nums:
        n_noact += 1
    for b in btns:
        btn_c[189 if 180 <= b <= 189 else b] += 1
    if cx:
        cx_c[cx[0]] += 1
    if cy:
        cy_c[cy[0]] += 1
    else:
        n_nocam += 1
    # no-op: 无按键 + 双轴中心
    if not btns and cx and cy and cx[0] == 0 and cy[0] == 0:
        n_noop += 1

print('总样本: %d | 无动作token: %d | 无camera: %d' % (n, n_noact, n_nocam))
print('no-op帧(无按键+双轴中心): %d (%.2f%%)' % (n_noop, n_noop * 100.0 / max(n, 1)))
print()
print('== 按键分布 ==')
for k in sorted(btn_c, key=lambda x: -btn_c[x]):
    print('  %-9s %8d  %.2f%%/帧' % (BTN.get(k, str(k)), btn_c[k], btn_c[k] * 100.0 / max(n, 1)))
print()
for label, C in [('camera_x/pitch', cx_c), ('camera_y/yaw', cy_c)]:
    tot = sum(C.values())
    print('== %s (有效 %d, %.1f%%) ==' % (label, tot, tot * 100.0 / max(n, 1)))
    for off in range(-10, 11):
        c = C.get(off, 0)
        print('  %+4d  %8d  %.2f%%  %s' % (off, c, c * 100.0 / max(tot, 1), '#' * int(c * 100.0 / max(tot, 1) / 1.5)))
    print()
print('== 任务类型分布 ==')
for k, v in task_c.most_common(15):
    print('  %-28s %8d  %.2f%%' % (k, v, v * 100.0 / max(n, 1)))

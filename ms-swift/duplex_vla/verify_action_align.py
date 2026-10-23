#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""抽查 mcvla_video_sft 动作切分对齐 (kill 段 2 + 非 kill 段 2).

验证:
 1. vpt jsonl 第 f 行的 tick == f (行号对齐铁律)
 2. mp4 帧数 == n_frames == len(actions)
 3. actions[i] == action_to_token_text(jsonl 第 (abs_start+i) 行) (内容重建一致)
"""
import json
import os
import subprocess
import sys
from itertools import islice

HERE = '/apdcephfs/private_chengzhixu/ms-swift/mcvla'
sys.path.insert(0, HERE)
from convert_actions_to_tokens import action_to_token_text

VPT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/vpt')
TS = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
      'chengzhixu/dataset/mcvla_video_sft/train.jsonl')
OUT = '/apdcephfs/private_chengzhixu/verify_align.out'

LOC = {}
for _d in ('6.x', '7.x', '10.x'):
    for _f in os.listdir(os.path.join(VPT, _d)):
        if _f.endswith('.mp4'):
            LOC[_f[:-4]] = _d

kills, others = [], []
for line in open(TS):
    d = json.loads(line)
    (kills if 'kill' in d['task'] else others).append(d)
    if len(kills) >= 2 and len(others) >= 2:
        break
picks = kills[:2] + others[:2]

report = []
for d in picks:
    vid = d['video_id']
    base = vid.rsplit('-', 2)[0] if vid.count('-') > 2 else vid
    # video_id = base + '-o0-o1'; base 本身含 '-'
    m = vid.rsplit('-', 2)
    base = '-'.join(m[:-2]) if len(m) == 3 else vid
    src = LOC.get(base)
    if not src:
        report.append('%s: no_vpt_src SKIP' % vid)
        continue
    # 1. jsonl 行号对齐
    lo, hi = d['abs_start'], d['abs_end']
    path = os.path.join(VPT, src, base + '.jsonl')
    with open(path) as f:
        seg = [json.loads(l) for l in islice(f, lo, hi + 1)]
    tick_ok = None
    if seg and 'tick' in seg[0]:
        tick_ok = all(r.get('tick') == lo + i for i, r in enumerate(seg))
    # 2. mp4 帧数
    mp4 = os.path.join(os.path.dirname(TS), d['video'])
    try:
        n_mp4 = int(subprocess.run(
            ['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', mp4],
            capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:
        n_mp4 = -1
    # 3. 内容重建一致
    re_tok = [action_to_token_text(r)[0] for r in seg]
    content_ok = re_tok[:len(d['actions'])] == d['actions'] and len(re_tok) == len(d['actions'])
    report.append(
        '%s [%s]\n'
        '  行号对齐(tick): %s | mp4帧=%d n_frames=%d 动作数=%d | 内容重建一致: %s'
        % (vid[:60], d['task'], tick_ok, n_mp4, d['n_frames'], len(d['actions']), content_ok))

open(OUT, 'w').write('\n'.join(report) + '\n')
print('done')

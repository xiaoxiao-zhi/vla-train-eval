#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 video_tasks_balanced.jsonl 构建 指令-视频-动作 数据集 (视频级样本).

每段输出:
  videos/{video_id}.mp4      — vpt 原视频按当前 abs_start/abs_end 截取 (截断段自动用新边界)
  train.jsonl 一行:
    {video_id, task, instruction(随机1条), video, abs_start, abs_end,
     n_frames, truncated, actions: [逐帧动作token文本...]}

对齐铁律:
  - 视频帧 f(绝对帧号) <-> vpt jsonl 第 f 行 (已验证 tick==行号 连续)
  - 动作逐行经 convert_actions_to_tokens.action_to_token_text 转保留字 (与 stage2 同链路)
  - kill 尾延+5 帧的动作: 优先读 jsonl 越界行(视频未结束则存在), 真缺才补 noop

用法 (GPU 机):
  python build_video_sft.py [--workers 16]
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
from itertools import islice
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert_actions_to_tokens import action_to_token_text  # noqa: E402

VPT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/vpt')
BAL = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/minecraft-vla-sft/video_tasks_balanced.jsonl')
OUT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/mcvla_video_sft')

LOC = {}
for _d in ('6.x', '7.x', '10.x'):
    for _f in os.listdir(os.path.join(VPT, _d)):
        if _f.endswith('.mp4'):
            LOC[_f[:-4]] = _d

NOOP_JSON = {"mouse": {"buttons": [], "newButtons": []},
             "keyboard": {"keys": [], "newKeys": []}}
try:
    NOOP_TOKEN = action_to_token_text(NOOP_JSON)[0]
except Exception:
    NOOP_TOKEN = None  # 兜底: 用段内最后一个动作


def load_actions(base, lo, hi):
    """读 vpt jsonl 行 [lo, hi] (绝对帧号), 转 token 文本列表. 缺行补 noop."""
    path = os.path.join(VPT, LOC[base], base + '.jsonl')
    out = []
    with open(path) as f:
        for line in islice(f, lo, hi + 1):
            out.append(action_to_token_text(json.loads(line))[0])
    n_expect = hi - lo + 1
    if len(out) < n_expect:  # 视频末尾越界 (kill 尾延等)
        fill = NOOP_TOKEN if NOOP_TOKEN else (out[-1] if out else '')
        out.extend([fill] * (n_expect - len(out)))
    return out


def process(r):
    vid = r['video_id']
    base = re.sub(r'-\d+-\d+$', '', vid)
    src = LOC.get(base)
    if not src:
        return None, vid, 'no_video'
    try:
        dst = os.path.join(OUT, 'videos', vid + '.mp4')
        if not os.path.exists(dst):  # 幂等
            subprocess.run(
                ['ffmpeg', '-y', '-loglevel', 'error',
                 '-i', os.path.join(VPT, src, base + '.mp4'),
                 '-vf', "select='between(n,%d,%d)'" % (r['abs_start'], r['abs_end']),
                 '-fps_mode', 'passthrough', dst],
                check=True, timeout=600)
        actions = load_actions(base, r['abs_start'], r['abs_end'])
        # 指令: 多条随机选一 (每段独立种子, 多进程下可复现)
        instr = r['instruction']
        if isinstance(instr, list):
            instr = random.Random(abs(hash(vid)) % (2**31)).choice(instr)
        instr = instr.replace('\nobservation:', '').replace('observation:', '').strip()
        row = {'video_id': vid, 'task': r['task'], 'instruction': instr,
               'video': 'videos/%s.mp4' % vid,
               'abs_start': r['abs_start'], 'abs_end': r['abs_end'],
               'n_frames': r['abs_end'] - r['abs_start'] + 1,
               'truncated': r.get('truncated', False),
               'actions': actions}
        return row, vid, None
    except Exception as e:
        return None, vid, str(e)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'videos'), exist_ok=True)

    rows = [json.loads(l) for l in open(BAL)]
    # 段内随机种子: 每段独立可复现
    for i, r in enumerate(rows):
        random.Random(i).random  # noqa (占位说明: process 内 random.choice 用全局种子)
    random.seed(42)

    done = 0
    skip = []
    with open(os.path.join(OUT, 'train.jsonl'), 'w') as out, \
         Pool(args.workers) as pool:
        for row, vid, err in pool.imap_unordered(process, rows, chunksize=8):
            if err:
                skip.append((vid, err))
                continue
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            done += 1
            if done % 500 == 0:
                print('完成 %d/%d (跳过 %d)' % (done, len(rows), len(skip)), flush=True)

    with open(os.path.join(OUT, 'skipped.jsonl'), 'w') as f:
        for vid, err in skip:
            f.write(json.dumps({'video_id': vid, 'err': err}) + '\n')
    print('全部完成: %d 成功, %d 跳过 -> %s' % (done, len(skip), OUT), flush=True)


if __name__ == '__main__':
    main()

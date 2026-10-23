#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""视频级数据 -> stage2 同款帧级 swift 格式 (history=1).

每段视频:
  1. ffmpeg 全帧抽取 -> frames/{video_id}_f%06d.jpg (段内相对帧号 0 起)
  2. 展开:
     t=0 : 单轮   [指令+帧0 -> 动作0]
     t>=1: 两轮   [指令+帧(t-1) / 动作(t-1) / 帧(t) -> 动作(t)]   (与旧 mcvla_sft 同构)

格式 (与旧 stage2 train.jsonl 逐字段一致):
  user: "{指令}\nobservation: \n<image>"  /  "observation: \n<image>"
  assistant: "{动作token}\n"
  images: 绝对路径

用法 (GPU 机): python video_to_frames_sft.py [--workers 16]
"""
import argparse
import json
import os
import subprocess
from multiprocessing import Pool

SRC = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/mcvla_video_sft')
OUT_JSONL = os.path.join(SRC, 'train_frames.jsonl')


def frame_path(vid, t):
    return os.path.join(SRC, 'frames', '%s_f%06d.jpg' % (vid, t))


def process(r):
    vid = r['video_id']
    n = r['n_frames']
    actions = r['actions']
    instr = r['instruction']
    mp4 = os.path.join(SRC, r['video'])
    # 1) 抽帧 (幂等: 已有同数量帧则跳过)
    have = sum(1 for t in range(n) if os.path.exists(frame_path(vid, t)))
    if have < n:
        os.makedirs(os.path.join(SRC, 'frames'), exist_ok=True)
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4,
             '-start_number', '0',
             os.path.join(SRC, 'frames', '%s_f%%06d.jpg' % vid)],
            check=True, timeout=600)
    # 2) 展开样本 (history=1)
    rows = []
    for t in range(n):
        if t == 0:
            msgs = [{'role': 'user',
                     'content': '%s\nobservation: \n<image>' % instr},
                    {'role': 'assistant', 'content': actions[0] + '\n'}]
            imgs = [frame_path(vid, 0)]
        else:
            msgs = [{'role': 'user',
                     'content': '%s\nobservation: \n<image>' % instr},
                    {'role': 'assistant', 'content': actions[t-1] + '\n'},
                    {'role': 'user', 'content': 'observation: \n<image>'},
                    {'role': 'assistant', 'content': actions[t] + '\n'}]
            imgs = [frame_path(vid, t-1), frame_path(vid, t)]
        rows.append(json.dumps({'messages': msgs, 'images': imgs},
                               ensure_ascii=False))
    return vid, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(SRC, 'train.jsonl'))]
    print('视频段 %d, 预计帧样本 ~%d' % (len(rows), sum(r['n_frames'] for r in rows)), flush=True)

    done = 0
    with open(OUT_JSONL, 'w') as out, Pool(args.workers) as pool:
        for vid, lines in pool.imap(process, rows, chunksize=4):
            for line in lines:
                out.write(line + '\n')
            done += 1
            if done % 400 == 0:
                print('  %d/%d 段' % (done, len(rows)), flush=True)
    total = sum(1 for _ in open(OUT_JSONL))
    print('完成: %d 段 -> %d 条帧样本 -> %s' % (len(rows), total, OUT_JSONL), flush=True)


if __name__ == '__main__':
    main()

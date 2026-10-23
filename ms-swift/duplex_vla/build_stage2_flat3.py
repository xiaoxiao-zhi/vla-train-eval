#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stage2 数据构建 v4-flat: 单轮拍平格式 (3图连排, 无历史动作token).

与 v3 多轮 hist2 的最小 diff (已跑通基线: duplex_vla_v3/stage2_train_hist2.jsonl):
  - 旧 (hist2 多轮): user/assistant 交替 3 轮, 历史轮 assistant = 历史动作+"\n\n"
  - 新 (flat3 单轮): 历史动作全部丢弃, 观测占位直接连排
    user: "指令\nobservation: \n<image>observation: \n<image>observation: \n<image>"
    asst: "动作\n\n"  (仅当前帧动作, s2v2 同款尾部; 动作文本源数据已编好, 直接用)
  - images 不变: [idx-2, idx-1, idx] 三张帧图 (段首 idx<2 自动退化少占位)
  - QA 30 万原样混入 (纯文本单轮, 不动)

数据源: mcvla_video_sft/train.jsonl (5845 段, kill 无尾延, 动作已是token文本)
输出:   dataset/duplex_vla_v4/stage2_train_flat3.jsonl
"""
import argparse
import json
import os
import random

VIDEO_SFT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
             'chengzhixu/dataset/mcvla_video_sft/train.jsonl')
FRAMES_DIR = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
              'chengzhixu/dataset/mcvla_video_sft/frames')
QA_PATH = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/general_qa_300k/train.jsonl')
OUT_DIR = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/duplex_vla_v4')

HIST = 2  # 历史帧数 (加当前帧共 3 图, 同旧 hist2 的窗口)
OBS = 'observation: \n<image>'


def build(traj):
    """一条轨迹 -> 每帧一条 flat 样本 (段首退化少占位). 纯字符串操作, 无tokenizer."""
    vid, instruction, actions = traj
    n = len(actions)
    frame = lambda i: os.path.join(FRAMES_DIR, '%s_f%06d.jpg' % (vid, i))  # noqa: E731
    samples = []
    for idx in range(n):
        h = min(idx, HIST)  # 段首可用历史帧数
        user = instruction + '\n' + OBS * (h + 1)
        act = actions[idx].rstrip('\n')  # 去自带\n, 尾部不加任何字符
        samples.append({
            'messages': [
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': act},
            ],
            'images': [frame(i) for i in range(idx - h, idx + 1)],
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    random.seed(42)

    trajs = []
    for l in open(VIDEO_SFT):
        d = json.loads(l)
        trajs.append((d['video_id'], d['instruction'], d['actions']))
    if args.limit:
        trajs = trajs[:args.limit]
    print('VLA 轨迹: %d 段 (无尾延)' % len(trajs), flush=True)

    qa_lines = [l.rstrip('\n') for l in open(QA_PATH)]
    print('QA: %d 条' % len(qa_lines), flush=True)

    vla = []
    for i, traj in enumerate(trajs):
        for s in build(traj):
            vla.append(json.dumps(s, ensure_ascii=False))
        if (i + 1) % 1000 == 0:
            print('  %d/%d 段, 累计 %d 条' % (i + 1, len(trajs), len(vla)), flush=True)
    print('VLA 完成: %d 条' % len(vla), flush=True)

    out_path = os.path.join(OUT_DIR, 'stage2_train_flat3.jsonl')
    lines = vla + qa_lines
    random.shuffle(lines)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('写出: %d 行 -> %s' % (len(lines), out_path), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stage2 数据构建 v3 (完全照抄 mcvla_stage2_v2 已跑通格式 + kill 无尾延 + 双窗口).

s2v2 格式 (实测):
  hist1 (1历史帧): 滚动 2 轮, 每轮 1 图
    user0: "指令\\nobservation: \\n<image>"  assistant0: "动作\\n\\n"
    user1: "observation: \\n<image>"         assistant1: "动作\\n\\n"  (target)
    images: [历史帧, 当前帧]
  hist0 (仅当前帧): 单轮
    user: "指令\\nobservation: \\n<image>"   assistant: "动作\\n\\n"
    images: [当前帧]
QA 30 万原样混入 (纯文本单轮).
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, '/apdcephfs/private_chengzhixu/ms-swift/duplex_vla')
import build_duplex_dataset as B  # noqa: E402

VIDEO_SFT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
             'chengzhixu/dataset/mcvla_video_sft/train.jsonl')
FRAMES_DIR = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
              'chengzhixu/dataset/mcvla_video_sft/frames')
QA_PATH = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/general_qa_300k/train.jsonl')
OUT_DIR = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/duplex_vla_v3')

WINDOWS = (2, 3)  # 历史帧数 (本次构建目标; 0/1 已有)

TOK = None


def init_worker():
    global TOK
    TOK = B.get_tok()


def build(task):
    """worker: 一条轨迹 -> 各窗口的滚动样本 {K: [...]} (K = 历史帧数, 段首自动退化)."""
    vid, instruction, actions = task
    n = len(actions)
    act_txts = [TOK.decode(B.enc_action(a)).rstrip('\n') for a in actions]  # 去自带的\n(198), s2v2 同款
    frame = lambda i: os.path.join(FRAMES_DIR, '%s_f%06d.jpg' % (vid, i))  # noqa: E731
    res = {K: [] for K in WINDOWS}
    for K in WINDOWS:
        for idx in range(n):
            h = min(idx, K)  # 段首可用历史帧数 (idx<K 时退化)
            msgs = [{'role': 'user', 'content': '%s\nobservation: \n<image>' % instruction}]
            for j in range(idx - h, idx):  # 历史轮
                msgs.append({'role': 'assistant', 'content': act_txts[j] + '\n\n'})
                msgs.append({'role': 'user', 'content': 'observation: \n<image>'})
            msgs.append({'role': 'assistant', 'content': act_txts[idx] + '\n\n'})  # target
            res[K].append({
                'messages': msgs,
                'images': [frame(i) for i in range(idx - h, idx + 1)],
            })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
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

    qa_lines = []
    for l in open(QA_PATH):
        d = json.loads(l)
        qa_lines.append(json.dumps(d, ensure_ascii=False))
    print('QA: %d 条' % len(qa_lines), flush=True)

    out_files = {K: os.path.join(OUT_DIR, 'stage2_train_hist%d.jsonl' % K)
                 for K in WINDOWS}
    vla = {K: [] for K in WINDOWS}
    with Pool(args.workers, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(build, trajs, chunksize=8)):
            for K in WINDOWS:
                vla[K].extend(json.dumps(s, ensure_ascii=False) for s in res[K])
            if (i + 1) % 1000 == 0:
                print('  %d/%d 段, %s' % (
                    i + 1, len(trajs),
                    ', '.join('hist%d=%d' % (K, len(vla[K])) for K in WINDOWS)), flush=True)
    print('VLA 完成: %s' % ', '.join('hist%d=%d' % (K, len(vla[K])) for K in WINDOWS), flush=True)

    for K in WINDOWS:
        lines = vla[K] + qa_lines
        random.shuffle(lines)
        with open(out_files[K], 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('写出: %d 行 -> %s' % (len(lines), out_files[K]), flush=True)


if __name__ == '__main__':
    main()

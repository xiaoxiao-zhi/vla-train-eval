#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工 v2 数据构建: 从 mcvla_video_sft (20万帧筛选清单) + general_qa_300k.

复刻 build_duplex_turns.py 的已验证逻辑 (插入点合法性 / answer 分片 /
等长双通道 / silence 占位 / OBS_WINDOW=2), 数据源换成:
  - 轨迹: video_tasks_duplex20w.jsonl (1,345 段) + mcvla_video_sft/train.jsonl (actions/instruction)
  - QA:   general_qa_300k/train.jsonl (随机 1,345 对, 1:1 配轨迹, 与旧设计一致)
  - 帧:   mcvla_video_sft/frames/ (已抽好, 与缺帧段零重叠)

输出: dataset/duplex_vla_v2/duplex_turns_train.jsonl
用法 (GPU 机): python build_duplex_v2.py [--workers 16]
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_duplex_dataset as B  # noqa: E402  (get_tok / enc_action, TOKENIZER_DIR=官方)

DUP20W = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
          'chengzhixu/dataset/minecraft-vla-sft/video_tasks_duplex20w.jsonl')
VIDEO_SFT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
             'chengzhixu/dataset/mcvla_video_sft/train.jsonl')
FRAMES_DIR = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
              'chengzhixu/dataset/mcvla_video_sft/frames')
QA_PATH = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/general_qa_300k/train.jsonl')
OUT_DIR = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/duplex_vla_v2')
TOKENS_PATH = os.path.join(HERE, 'duplex_tokens.json')
OBS_WINDOW = (2,)

CTRL = None  # worker 内控制 token id


def init_worker():
    global CTRL
    m = json.load(open(TOKENS_PATH))
    CTRL = m['ids']


def process_traj(task):
    vid, instruction, actions, (q, a, ans_ids) = task
    n = len(actions)
    silence = [CTRL['<|silence|>']]

    # ---- 插入点合法性 (与旧版一致): 从 k 起的动作 token 数装得下 answer ----
    lens = [len(B.enc_action(act)) for act in actions]
    lo = min(3, max(0, n - 1))
    valid = []
    for k in range(lo, n):
        s, j = 0, k
        while j < n and s < len(ans_ids):
            s += lens[j]
            j += 1
        if s >= len(ans_ids):
            valid.append(k)
    if not valid:
        valid = [lo]
    r = random.Random(hash(vid) & 0xffffffff)
    k = r.choice(valid)

    # ---- 每轮 4 通道 ----
    turns = []
    pos = 0
    for idx in range(n):
        action_ids = B.enc_action(actions[idx])
        text_ids = None
        if idx >= k and pos < len(ans_ids):
            seg = ans_ids[pos:pos + len(action_ids)]
            pos += len(seg)
            text_ids = seg
        if idx == 0:
            user_ids = B.get_tok()(instruction, add_special_tokens=False)['input_ids']
        elif idx == k:
            user_ids = B.get_tok()(q, add_special_tokens=False)['input_ids']
        else:
            user_ids = None
        turns.append({
            't': idx,
            'user_ids': user_ids,             # None -> silence
            'action_ids': action_ids,
            'text_ids': text_ids,             # None -> silence
            'frame': '%s_f%06d.jpg' % (vid, idx),
        })

    # ---- 单轮拆分 (每时间步一条样本) ----
    K = OBS_WINDOW[0]
    samples = []
    for t_idx in range(n):
        cur = turns[t_idx]
        ctx = []
        for i in range(t_idx):
            tn = turns[i]
            ctx.append({
                'i': tn['t'],
                'user_ids': tn['user_ids'] if tn['user_ids'] is not None else silence,
                'txt_ids': tn['text_ids'] if tn['text_ids'] is not None else silence,
                'frame': tn['frame'] if i >= t_idx - K else None,
                'action_ids': tn['action_ids'] if i >= t_idx - K else None,
            })
        images = [c['frame'] for c in ctx if c['frame']] + [cur['frame']]
        samples.append({
            'base': vid,
            't': cur['t'],
            'context': ctx,
            'current': {
                'user_ids': cur['user_ids'] if cur['user_ids'] is not None else silence,
                'frame': cur['frame'],
            },
            'target': {
                'action_ids': cur['action_ids'],
                'txt_ids': cur['text_ids'] if cur['text_ids'] is not None else silence,
            },
            'qa_meta': {'question': q, 'answer': a, 'insert_step': k},
            'images': [os.path.join(FRAMES_DIR, f) for f in images],
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 轨迹: 20w 清单 x video_sft 的 actions/instruction
    want = {json.loads(l)['video_id'] for l in open(DUP20W)}
    trajs = []
    for l in open(VIDEO_SFT):
        d = json.loads(l)
        if d['video_id'] in want:
            trajs.append((d['video_id'], d['instruction'], d['actions']))
    print('轨迹: %d 段 (清单 %d)' % (len(trajs), len(want)), flush=True)

    # QA: 随机抽同数量 (1:1)
    qa_pool = []
    for l in open(QA_PATH):
        d = json.loads(l)
        q = d['messages'][0]['content']
        a = d['messages'][1]['content']
        qa_pool.append((q, a))
    picks = random.sample(qa_pool, len(trajs))
    tasks = []
    for (vid, instr, acts), (q, a) in zip(trajs, picks):
        ans_ids = B.get_tok()(a, add_special_tokens=False)['input_ids']
        tasks.append((vid, instr, acts, (q, a, ans_ids)))
    print('QA 配对: %d 对 (池 %d)' % (len(tasks), len(qa_pool)), flush=True)

    total = 0
    out_path = os.path.join(OUT_DIR, 'duplex_turns_train.jsonl')
    with open(out_path, 'w') as out, Pool(args.workers, initializer=init_worker) as pool:
        for i, samples in enumerate(pool.imap_unordered(process_traj, tasks, chunksize=8)):
            for s in samples:
                out.write(json.dumps(s, ensure_ascii=False) + '\n')
            total += len(samples)
            if (i + 1) % 300 == 0:
                print('  %d/%d 段, 累计 %d 轮' % (i + 1, len(tasks), total), flush=True)
    print('完成: %d 段 -> %d 轮样本 -> %s' % (len(tasks), total, out_path), flush=True)


if __name__ == '__main__':
    main()

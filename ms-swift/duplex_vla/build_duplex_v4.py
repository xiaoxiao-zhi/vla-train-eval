#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工 v4 数据构建: 单轮拍平格式 (动作窗口0 + 多图单 message).

与 v3 (多轮 messages) 的差异:
  - 整个滚动历史拍平成一条 user content, 每帧用
    <|obs_user|> / <|obs_asst|> 两个特殊 token 标记边界 (纯字符串拼接)
  - 动作窗口 0: 历史帧的 assistant 部分只留文本 (silence/回答片段), 无动作
  - 图片窗口 2: 窗口内帧带 "observation: \n<image>" 占位, 其余帧纯文本
  - target = 当前帧 动作+文本 (唯一 loss 位置, 配合 default 即 last 语义)
  - 词表: +3 token (<|silence|>, <|obs_user|>, <|obs_asst|>)

数据源同 v3: DUP20W 1345 段 + video_sft (无尾延动作) + general_qa_300k
输出: dataset/duplex_vla_v4/duplex_train.jsonl
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, '/apdcephfs/private_chengzhixu/ms-swift/duplex_vla')
import build_duplex_dataset as B  # noqa: E402

DUP20W = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
          'chengzhixu/dataset/minecraft-vla-sft/video_tasks_duplex20w.jsonl')
VIDEO_SFT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
             'chengzhixu/dataset/mcvla_video_sft/train.jsonl')
FRAMES_DIR = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
              'chengzhixu/dataset/mcvla_video_sft/frames')
QA_PATH = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/general_qa_300k/train.jsonl')
OUT_DIR = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/duplex_vla_v4')

SILENCE = '<|silence|>'
OBS_USER = '<|obs_user|>'
OBS_ASST = '<|obs_asst|>'
IMG_WINDOW = 2          # 图片窗口 (当前帧 + 最近2帧)

TOK = None


def init_worker():
    global TOK
    TOK = B.get_tok()


def safe_seg(ans_ids, budget):
    """取 ans_ids 前缀 (<=budget), 保证 decode->encode 可还原."""
    seg = ans_ids[:budget]
    while seg:
        txt = TOK.decode(seg)
        if TOK(txt, add_special_tokens=False)['input_ids'] == seg:
            return seg, txt
        seg = seg[:-1]
    return [], ''


def process_traj(task):
    vid, instruction, actions, (q, a, ans_ids) = task
    n = len(actions)
    r = random.Random(hash(vid) & 0xffffffff)

    # ---- QA 插入点 (v3 逻辑) ----
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
    k_ins = r.choice(valid)

    # ---- 每帧: user文本 / 回答分片 ----
    turns = []
    pos = 0
    for idx in range(n):
        if idx == 0:
            user_txt = instruction
        elif idx == k_ins:
            user_txt = q
        else:
            user_txt = SILENCE
        ans_txt = None
        if idx >= k_ins and pos < len(ans_ids):
            seg, seg_txt = safe_seg(ans_ids[pos:], lens[idx])
            if seg:
                pos += len(seg)
                ans_txt = seg_txt
        turns.append({'user_txt': user_txt, 'ans_txt': ans_txt})

    # ---- 每时间步一条样本 (单轮拍平) ----
    samples = []
    for t_idx in range(n):
        cur = turns[t_idx]
        parts = []
        for i in range(t_idx):  # 历史帧 (动作窗口0: assistant 只留文本)
            tn = turns[i]
            in_img = i >= t_idx - IMG_WINDOW
            u = tn['user_txt'] + ('\nobservation: \n<image>' if in_img else '')
            a = tn['ans_txt'] if tn['ans_txt'] is not None else SILENCE
            parts.append(OBS_USER + u + OBS_ASST + a)
        # 当前帧 user (必有图)
        parts.append(OBS_USER + cur['user_txt'] + '\nobservation: \n<image>')
        prompt = ''.join(parts)

        tgt = TOK.decode(B.enc_action(actions[t_idx])) + (
            cur['ans_txt'] if cur['ans_txt'] is not None else SILENCE)
        images = [os.path.join(FRAMES_DIR, '%s_f%06d.jpg' % (vid, i))
                  for i in range(max(0, t_idx - IMG_WINDOW), t_idx + 1)]
        samples.append({
            'base': vid,
            't': t_idx,
            'messages': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': tgt},
            ],
            'images': images,
            'qa_meta': {'question': q, 'answer': a, 'insert_step': k_ins},
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    want = {json.loads(l)['video_id'] for l in open(DUP20W)}
    trajs = []
    for l in open(VIDEO_SFT):
        d = json.loads(l)
        if d['video_id'] in want:
            trajs.append((d['video_id'], d['instruction'], d['actions']))
    if args.limit:
        trajs = trajs[:args.limit]
    print('轨迹: %d 段' % len(trajs), flush=True)

    qa_pool = []
    for l in open(QA_PATH):
        d = json.loads(l)
        qa_pool.append((d['messages'][0]['content'], d['messages'][1]['content']))
    picks = random.sample(qa_pool, len(trajs))
    tok = B.get_tok()
    tasks = []
    for (vid, instr, acts), (q, a) in zip(trajs, picks):
        ans_ids = tok(a, add_special_tokens=False)['input_ids']
        tasks.append((vid, instr, acts, (q, a, ans_ids)))
    print('QA 配对: %d 对' % len(tasks), flush=True)

    total = 0
    out_path = os.path.join(OUT_DIR, 'duplex_train.jsonl')
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

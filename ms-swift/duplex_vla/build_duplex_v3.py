#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工 v3 数据构建: 标准对话格式 (swift 直接训练, 与 stage2 对齐).

与 v2 的差异 (仅输出格式, 轮次构造逻辑复用):
  - 去掉全部 duplex 控制符 (<|duplex_user|>/<|duplex_obs|>/<|duplex_act|>/<|duplex_txt|>)
  - 输出标准 messages jsonl:
      user:      content = [image?(仅最近2轮), text(指令/QA问题/<|silence|>)]
      assistant: content = 动作token文本 + 文本(回答片段/<|silence|>), 直接拼接
  - 历史轮 assistant 动作同样仅保留最近 2 轮 (更早轮只剩文本)
  - 词表唯一改动: <|silence|> (训练前扩词表)

数据源 (同 v2): video_tasks_duplex20w + mcvla_video_sft + general_qa_300k
输出: dataset/duplex_vla_v3/duplex_train.jsonl
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
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
           'chengzhixu/dataset/duplex_vla_v3')
SILENCE = '<|silence|>'
OBS_WINDOW = 2

TOK = None  # worker 内


def init_worker():
    global TOK
    TOK = B.get_tok()


def safe_seg(ans_ids, budget):
    """取 ans_ids 前缀 (<=budget 个 token), 保证 decode->encode 可还原."""
    seg = ans_ids[:budget]
    while seg:
        txt = TOK.decode(seg)
        if TOK(txt, add_special_tokens=False)['input_ids'] == seg:
            return seg, txt
        seg = seg[:-1]  # 边界回退 (多字节 token 切开时)
    return [], ''


def process_traj(task):
    vid, instruction, actions, (q, a, ans_ids) = task
    n = len(actions)
    r = random.Random(hash(vid) & 0xffffffff)

    # ---- QA 插入点 (v2 逻辑): 从 k 起累计动作 token 数装得下答案 ----
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
    k = r.choice(valid)

    # ---- 每轮: user文本 / 动作ids / 回答分片(跨帧流式, 每帧<=动作token数) ----
    turns = []
    pos = 0
    for idx in range(n):
        action_ids = B.enc_action(actions[idx])
        if idx == 0:
            user_txt = instruction
        elif idx == k:
            user_txt = q
        else:
            user_txt = SILENCE
        ans_txt = None
        if idx >= k and pos < len(ans_ids):
            seg, seg_txt = safe_seg(ans_ids[pos:], len(action_ids))
            if seg:
                pos += len(seg)
                ans_txt = seg_txt
        turns.append({
            't': idx,
            'user_txt': user_txt,
            'action_ids': action_ids,
            'ans_txt': ans_txt,          # None -> silence
            'frame': '%s_f%06d.jpg' % (vid, idx),
        })

    # ---- 单轮拆分: 每时间步一条样本, s2v2 骨架 (observation+<image>占位) ----
    def split(K):
        samples = []
        for t_idx in range(n):
            cur = turns[t_idx]
            msgs = []
            for i in range(t_idx):
                tn = turns[i]
                in_win = i >= t_idx - K  # 图片/动作窗口
                # user 历史轮: 窗口内带 <image> 占位 (s2v2 骨架), 窗口外纯文本
                u = tn['user_txt']
                if in_win:
                    u = '%s\nobservation: \n<image>' % u
                msgs.append({'role': 'user', 'content': u})
                # assistant 历史轮: 窗口内=动作\n\n+文本, 窗口外=纯文本
                if in_win:
                    a_txt = TOK.decode(tn['action_ids']).rstrip('\n') + '\n\n' + (
                        tn['ans_txt'] if tn['ans_txt'] is not None else SILENCE)
                else:
                    a_txt = tn['ans_txt'] if tn['ans_txt'] is not None else SILENCE
                msgs.append({'role': 'assistant', 'content': a_txt})
            # 当前轮 user: 文本 + observation + <image>
            msgs.append({'role': 'user',
                         'content': '%s\nobservation: \n<image>' % cur['user_txt']})
            # target: 动作\n\n + 文本(silence/回答片段)
            tgt = TOK.decode(cur['action_ids']).rstrip('\n') + '\n\n' + (
                cur['ans_txt'] if cur['ans_txt'] is not None else SILENCE)
            msgs.append({'role': 'assistant', 'content': tgt})

            images = [os.path.join(FRAMES_DIR, turns[i]['frame'])
                      for i in range(max(0, t_idx - K), t_idx + 1)]
            samples.append({
                'base': vid,
                't': cur['t'],
                'messages': msgs,
                'images': images,
                'qa_meta': {'question': q, 'answer': a, 'insert_step': k},
            })
        return samples

    # 同一轮次切分, 两种历史窗口 (视频/帧/动作源只读一次)
    return split(2), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 段 (冒烟)')
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

    total1 = 0
    p1 = os.path.join(OUT_DIR, 'duplex_train.jsonl')  # 全双工, 历史窗口 2
    with open(p1, 'w') as o1, \
         Pool(args.workers, initializer=init_worker) as pool:
        for i, (s1, _s2) in enumerate(pool.imap_unordered(process_traj, tasks, chunksize=8)):
            for s in s1:
                o1.write(json.dumps(s, ensure_ascii=False) + '\n')
            total1 += len(s1)
            if (i + 1) % 300 == 0:
                print('  %d/%d 段, 累计 %d 轮' % (i + 1, len(tasks), total1), flush=True)
    print('完成: %d 段 -> %d 轮样本 -> %s' % (len(tasks), total1, p1), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工 VLA 单轮数据集构建 (每个时间步 t 一条样本).

复用 build_duplex_dataset.py 的已验证逻辑 (轨迹聚合 / QA 固定配对 /
插入点合法性 / 等长双通道), 新增:

  - 控制 token: 5 个新 token 在官方词表尾部顺序分配 id, 写出 duplex_tokens.json
      <|duplex_user|> <|duplex_obs|> <|duplex_act|> <|duplex_txt|> <|silence|>
  - 单轮拆分: 轨迹每个 t 一条样本
      历史交互槽 (全部轮):   user_ids / txt_ids  (空槽 = silence_id)
      近 K 轮 (默认 2):      frame / action_ids  (窗口外置 null, 无占位)
      当前轮:                user_ids + frame  ->  target: action_ids + txt_ids

============================================================
输出 (单文件 jsonl, 方便训练流式读取):
  <out>/duplex_turns_train.jsonl   每行一个时间步样本:
  {
    "base": "<轨迹名>", "t": <当前帧号>,
    "context": [                          # i < t, 时间升序
      { "i": <帧号>,
        "user_ids":   [...] | [silence_id],   # i=0 指令 / i=k QA问题 / 其他 silence
        "txt_ids":    [...] | [silence_id],   # QA期间文本段 / 其他 silence
        "frame":      "<base>_f<i>.jpg" | null,   # 仅最近 K 轮
        "action_ids": [...] | null }              # 仅最近 K 轮
    ],
    "current": { "user_ids": [...] | [silence_id], "frame": "<base>_f<t>.jpg" },
    "target":  { "action_ids": [...], "txt_ids": [...] | [silence_id] },
    "qa_meta": { "question", "answer", "insert_step": k },
    "images":  [窗口内历史帧 + 当前帧, 按序列顺序]
  }
  <out>/progress.json             已完成轨迹 base (断点续跑)
  duplex_vla/duplex_tokens.json   控制 token 映射 (token <-> id, 构建时自动生成/复用)

用法:
  python build_duplex_turns.py --n 5 --out <dir>            # 试运行
  python build_duplex_turns.py --n all --out <dir>          # 全量 (50~100帧, 1288 轨迹)
  可选: --obs-window 2 --min-frames 50 --max-frames 100 --workers 8
"""
import argparse
import json
import os
import random
from multiprocessing import Pool

import build_duplex_dataset as B   # 复用: load_trajectories / enc_action / get_tok / 路径常量

DUPLEX_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENS_PATH = os.path.join(DUPLEX_DIR, "duplex_tokens.json")

CTRL_TOKENS = ["<|duplex_user|>", "<|duplex_obs|>", "<|duplex_act|>",
               "<|duplex_txt|>", "<|silence|>"]


def init_ctrl_tokens():
    """控制 token id = 官方 tokenizer 词表长度 + 0..4 (首次生成, 之后复用)."""
    if os.path.exists(TOKENS_PATH):
        m = json.load(open(TOKENS_PATH))
        return m["ids"]
    tok = B.get_tok()
    base = len(tok)   # 官方词表实际长度 (151916), 尾部追加
    ids = {name: base + i for i, name in enumerate(CTRL_TOKENS)}
    json.dump({"ids": ids, "vocab_base": base, "source": B.TOKENIZER_DIR},
              open(TOKENS_PATH, "w"), ensure_ascii=False, indent=1)
    return ids


CTRL = None   # worker 进程内: {"<|duplex_user|>": id, ...}


def process_traj(task):
    """一条轨迹 -> 该轨迹全部时间步的单轮样本列表 (worker 进程内执行)."""
    traj, (q, a, ans_ids) = task
    steps = traj["steps"]
    n = len(steps)
    silence = [CTRL["<|silence|>"]]

    # ---- 与整轨迹版一致: 插入点合法性 + 逐轮文本段分配 ----
    lens = [len(B.enc_action(act)) for _, act in steps]
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
    r = random.Random(hash(traj["base"]) & 0xffffffff)
    k = r.choice(valid)

    # 每轮的 4 通道内容 (先算好, 拆分时引用)
    turns = []
    pos = 0
    for idx, (t, act) in enumerate(steps):
        action_ids = B.enc_action(act)
        text_ids = None
        if idx >= k and pos < len(ans_ids):
            seg = ans_ids[pos:pos + len(action_ids)]
            pos += len(seg)
            text_ids = seg
        if idx == 0:
            user_ids = B.get_tok()(traj["title"], add_special_tokens=False)["input_ids"]
        elif idx == k:
            user_ids = B.get_tok()(q, add_special_tokens=False)["input_ids"]
        else:
            user_ids = None
        turns.append({
            "t": t,
            "user_ids": user_ids,             # None -> silence
            "action_ids": action_ids,
            "text_ids": text_ids,             # None -> silence
            "frame": "%s_f%06d.jpg" % (traj["base"], t),
        })

    # ---- 单轮拆分: 每个时间步 t 一条样本 ----
    K = OBS_WINDOW[0]
    samples = []
    for t_idx in range(n):
        cur = turns[t_idx]
        ctx = []
        for i in range(t_idx):
            tn = turns[i]
            ctx.append({
                "i": tn["t"],
                "user_ids": tn["user_ids"] if tn["user_ids"] is not None else silence,
                "txt_ids": tn["text_ids"] if tn["text_ids"] is not None else silence,
                "frame": tn["frame"] if i >= t_idx - K else None,
                "action_ids": tn["action_ids"] if i >= t_idx - K else None,
            })
        images = [c["frame"] for c in ctx if c["frame"]] + [cur["frame"]]
        samples.append({
            "base": traj["base"],
            "t": cur["t"],
            "context": ctx,
            "current": {
                "user_ids": cur["user_ids"] if cur["user_ids"] is not None else silence,
                "frame": cur["frame"],
            },
            "target": {
                "action_ids": cur["action_ids"],
                "txt_ids": cur["text_ids"] if cur["text_ids"] is not None else silence,
            },
            "qa_meta": {"question": q, "answer": a,
                        "insert_step": steps[k][0] if k < n else None},
            "images": images,
        })
    return samples


OBS_WINDOW = [2]   # worker 可读的窗口 K (主进程注入)


def main():
    global CTRL
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="5", help="轨迹条数: 整数 或 'all'")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--obs-window", type=int, default=2, help="观测/动作历史窗口 K")
    ap.add_argument("--min-frames", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    out_jsonl = os.path.join(args.out, "duplex_turns_train.jsonl")
    progress_path = os.path.join(args.out, "progress.json")
    OBS_WINDOW[0] = args.obs_window

    # 控制 token (主进程初始化一次, fork 后 worker 共享)
    ids = init_ctrl_tokens()
    CTRL = ids
    print("控制 token id:", {k.replace("<|duplex_", "").replace("|>", ""): v
                              for k, v in ids.items()}, flush=True)

    # QA 池 (与整轨迹版同款: 固定 seed shuffle, 不重复配对)
    print("加载并编码 QA 池...", flush=True)
    tok = B.get_tok()
    pool = []
    with open(B.QA_PATH) as f:
        for line in f:
            d = json.loads(line)
            q, a = d["messages"][0]["content"], d["messages"][1]["content"]
            if 5 <= len(a) <= 200:
                i = tok(a, add_special_tokens=False)["input_ids"]
                if len(i) <= 60:
                    pool.append((q, a, i))
    random.Random(args.seed).shuffle(pool)
    print("QA 池: %d 条" % len(pool), flush=True)

    # 轨迹 (帧数范围过滤)
    print("聚合轨迹...", flush=True)
    trajs = B.load_trajectories()
    n_all = len(trajs)
    trajs = [t for t in trajs
             if args.min_frames <= len(t["steps"]) <= args.max_frames]
    print("轨迹: %d -> %d (帧数 %d~%d 筛选后)" % (
        n_all, len(trajs), args.min_frames, args.max_frames), flush=True)
    qa_of = {t["base"]: pool[i % len(pool)] for i, t in enumerate(trajs)}

    # 断点续跑
    done = set()
    if os.path.exists(progress_path):
        done = set(json.load(open(progress_path)))
        print("续跑: 已完成 %d 条轨迹, 跳过" % len(done), flush=True)
    todo = [(t, qa_of[t["base"]]) for t in trajs if t["base"] not in done]
    if args.n != "all":
        todo = todo[:int(args.n)]
    print("待处理: %d 条轨迹 (workers=%d, obs_window=%d)" % (
        len(todo), args.workers, args.obs_window), flush=True)

    n_samples = n_qa_targets = 0
    with open(out_jsonl, "a") as fout, Pool(args.workers) as mp:
        for i, samples in enumerate(mp.imap_unordered(process_traj, todo, chunksize=2)):
            for s in samples:
                fout.write(json.dumps(s, ensure_ascii=False) + "\n")
                n_samples += 1
                if len(s["target"]["txt_ids"]) > 1 or (
                        s["target"]["txt_ids"] and s["target"]["txt_ids"][0] != CTRL["<|silence|>"]):
                    n_qa_targets += 1
            done.add(samples[0]["base"] if samples else "")
            if (i + 1) % 100 == 0:
                fout.flush()
                print("  轨迹 %d/%d (样本 %d)" % (i + 1, len(todo), n_samples), flush=True)
    json.dump(sorted(d for d in done if d), open(progress_path, "w"))

    print()
    print("=" * 50)
    print("完成: %d 轨迹 -> %d 单轮样本 (其中带文本输出的目标轮 %d, %.1f%%)" % (
        len(todo), n_samples, n_qa_targets, 100.0 * n_qa_targets / max(n_samples, 1)))
    print("输出: %s" % out_jsonl)


if __name__ == "__main__":
    main()

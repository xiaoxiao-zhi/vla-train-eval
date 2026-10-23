#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工 VLA 数据集构建 (正式版).

设计 (2026-09-22 与用户对齐):
  - 轨迹筛选: 帧总数在 [--min-frames, --max-frames] (默认 50~100) 内才处理, 否则跳过
  - 样本 = 整段轨迹 (每帧一轮交互, 样本覆盖全部帧)
  - 任务指令仅出现在第 0 轮 user 消息
  - 每条轨迹随机插 1 个 general-QA, 插入点 k 为随机帧 (避开首尾各 3 帧)
  - QA 不重复: 全量 QA 池固定 seed shuffle 后, 轨迹按排序序号 i -> QA[i] 固定配对
  - assistant 输出 = 动作 token (每轮必有) + 文本 token (QA 期间附加, 先动作后文本)
  - 等长双通道: 每轮 len(text_ids) == len(action_ids); 最后收尾轮允许短
  - **主存 token id** (text 为 decode 副本, decode 再 encode 不保证还原, 训练用 ids)

============================================================
保存格式 (明确):
  <out>/duplex_train.jsonl   每行一个轨迹样本, schema:
  {
    "base":        "<轨迹片段名>",
    "instruction": "<任务指令文本>",
    "qa": {
      "question":         "<QA 问题文本>",
      "answer":           "<QA 答案文本>",
      "answer_ids":       [int, ...],   # 答案完整 token id (tokenizer: JarvisVLA-Qwen2-VL-7B)
      "insert_step":      int,          # 插入帧号 t=k
      "answer_token_total": int
    },
    "turns": [                          # 按 t 升序, 覆盖轨迹全部帧
      {
        "t":          int,              # 帧号
        "user_text":  str,              # t=0: "<指令>\nobservation: \n"
                                        # t=k: "<QA问题>\nobservation: \n"
                                        # 其他: "observation: \n"
        "frame":      "<base>_f<t>.jpg",  # 帧文件名 (相对 mcvla_sft/frames/)
        "action_ids": [int, ...],       # 动作 token id (BEGIN..END, 不含换行)
        "action":     str,              # 动作保留字文本 (可读副本)
        "text_ids":   [int,...] | null, # QA 期间的文本段 id (等长于 action_ids), 无则 null
        "text":       str | null        # 文本段 decode 副本 (仅展示)
      }, ...
    ]
  }
  <out>/progress.json        已完成轨迹 base 列表 (断点续跑, 重跑自动跳过)

用法:
  python build_duplex_dataset.py --n 5 --out /path/to/dir          # 试运行 5 条
  python build_duplex_dataset.py --n all --out /path/to/dir        # 全量 (全部轨迹)
  python build_duplex_dataset.py --n all --out /path/to/dir --workers 16
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from multiprocessing import Pool

SFT = "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft"
QA_PATH = "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/general_qa/train.jsonl"
TOKENIZER_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B")

TOK = None            # worker 进程内懒加载
ACT_CACHE = {}        # 动作文本 -> ids 缓存 (动作序列种类有限, 避免重复 tokenize)


def get_tok():
    global TOK
    if TOK is None:
        from transformers import AutoTokenizer
        TOK = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return TOK


def enc_action(text):
    if text not in ACT_CACHE:
        ACT_CACHE[text] = get_tok()(text, add_special_tokens=False)["input_ids"]
    return ACT_CACHE[text]


def load_trajectories():
    """从 mcvla_sft/train.jsonl 聚合完整轨迹 (单进程扫一遍)."""
    segs = defaultdict(dict)
    titles = {}
    with open(os.path.join(SFT, "train.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            msgs = d["messages"]
            action = msgs[-1]["content"].strip()
            frame = d["images"][-1]
            m = re.match(r"(.+)_f(\d+)\.jpg$", os.path.basename(frame))
            if not m:
                continue
            base, t = m.group(1), int(m.group(2))
            segs[base][t] = action
            if base not in titles:
                titles[base] = msgs[0]["content"].split("\nobservation")[0]
    trajs = []
    for b, steps in segs.items():
        trajs.append({
            "base": b,
            "title": titles[b],
            "steps": sorted(steps.items()),   # [(t, action_text)]
        })
    trajs.sort(key=lambda x: x["base"])
    return trajs


def process_one(task):
    """单条轨迹 -> 全双工样本 dict (worker 进程内执行).

    task = (traj, qa_item): qa_item = (question, answer, answer_ids)
    由主进程预先分配 (全量 QA 池固定 seed shuffle 后按轨迹序号取, 不重复且续跑确定).
    """
    traj, (q, a, ans_ids) = task
    steps = traj["steps"]
    n = len(steps)
    # 每轮动作 token 数 (文本段额度)
    lens = [len(enc_action(act)) for _, act in steps]
    # 合法插入点 k: 不能太早 (>=3, 先做几步动作再被问),
    # 不能太晚 (从 k 起连续轮的额度累计必须覆盖答案全部 token, 保证说得完)
    lo = min(3, max(0, n - 1))
    valid = []
    for k in range(lo, n):
        s, j = 0, k
        while j < n and s < len(ans_ids):
            s += lens[j]
            j += 1
        if s >= len(ans_ids):
            valid.append(k)
    if not valid:      # 极端情况 (答案比整条轨迹额度还长): 理论上不会发生 (答案<=60)
        valid = [lo]
    k = rng_choice(traj["base"], valid)
    # 逐轮构造
    turns, pos = [], 0
    for idx, (t, action) in enumerate(steps):
        action_ids = enc_action(action)
        text_ids = None
        if idx >= k and pos < len(ans_ids):
            seg = ans_ids[pos:pos + len(action_ids)]
            pos += len(seg)
            text_ids = seg
        if idx == 0:
            user_text = "%s\nobservation: \n" % traj["title"]
        elif idx == k:
            user_text = "%s\nobservation: \n" % q
        else:
            user_text = "observation: \n"
        turns.append({
            "t": t,
            "user_text": user_text,
            "frame": "%s_f%06d.jpg" % (traj["base"], t),
            "action_ids": action_ids,
            "action": action,
            "text_ids": text_ids,
            "text": get_tok().decode(text_ids) if text_ids else None,
        })
    return {
        "base": traj["base"],
        "instruction": traj["title"],
        "qa": {"question": q, "answer": a, "answer_ids": ans_ids,
               "insert_step": steps[k][0] if k < n else None,
               "answer_token_total": len(ans_ids)},
        "turns": turns,
    }


def rng_choice(base, valid):
    """轨迹名哈希确定地从合法集合选插入点 (同轨迹重跑结果一致)."""
    r = random.Random(hash(base) & 0xffffffff)
    return r.choice(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="5",
                    help="处理轨迹条数: 整数 或 'all' (全部轨迹)")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--workers", type=int, default=8, help="并行进程数")
    ap.add_argument("--min-frames", type=int, default=50,
                    help="轨迹帧数下限 (帧总数不在此范围的轨迹跳过)")
    ap.add_argument("--max-frames", type=int, default=100,
                    help="轨迹帧数上限 (帧总数不在此范围的轨迹跳过)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    out_jsonl = os.path.join(args.out, "duplex_train.jsonl")
    progress_path = os.path.join(args.out, "progress.json")

    # QA 池: 主进程预编码 (答案 token id), fork 后 worker 共享只读
    print("加载并编码 QA 池...", flush=True)
    tok = get_tok()
    pool = []
    with open(QA_PATH) as f:
        for line in f:
            d = json.loads(line)
            q = d["messages"][0]["content"]
            a = d["messages"][1]["content"]
            if 5 <= len(a) <= 200:
                ids = tok(a, add_special_tokens=False)["input_ids"]
                if len(ids) <= 60:      # 答案过长直接淘汰 (说不完)
                    pool.append((q, a, ids))
    # QA 不重复且续跑确定: 全量池固定 seed shuffle, 轨迹按排序序号取对应 QA
    random.Random(args.seed).shuffle(pool)
    print("QA 池: %d 条 (答案 <= 60 token)" % len(pool), flush=True)

    # 轨迹 (帧数范围过滤: 不在 [min, max] 内的跳过)
    print("聚合轨迹...", flush=True)
    trajs = load_trajectories()
    n_all = len(trajs)
    trajs = [t for t in trajs
             if args.min_frames <= len(t["steps"]) <= args.max_frames]
    print("轨迹总数: %d -> %d (帧数 %d~%d 筛选后)" % (
        n_all, len(trajs), args.min_frames, args.max_frames), flush=True)

    # 轨迹 i -> QA[i]: 固定映射 (排序后序号), 保证不重复、重跑/续跑一致
    qa_of = {t["base"]: pool[i % len(pool)] for i, t in enumerate(trajs)}

    # 断点续跑
    done = set()
    if os.path.exists(progress_path):
        done = set(json.load(open(progress_path)))
        print("续跑: 已完成 %d 条, 跳过" % len(done), flush=True)
    todo = [(t, qa_of[t["base"]]) for t in trajs if t["base"] not in done]
    if args.n != "all":
        todo = todo[:int(args.n)]
    print("待处理: %d 条 (workers=%d)" % (len(todo), args.workers), flush=True)

    # 多进程处理 + 逐条落盘
    n_ok = n_turns = n_qa_turns = 0
    with open(out_jsonl, "a") as fout, Pool(args.workers) as mp:
        for i, sample in enumerate(mp.imap_unordered(process_one, todo, chunksize=4)):
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                fout.flush()
                print("  %d/%d" % (i + 1, len(todo)), flush=True)
            n_ok += 1
            n_turns += len(sample["turns"])
            n_qa_turns += sum(1 for tn in sample["turns"] if tn["text_ids"])
            done.add(sample["base"])
    json.dump(sorted(done), open(progress_path, "w"))

    print()
    print("=" * 50)
    print("完成: %d 条轨迹, %d 轮交互, 其中 %d 轮带文本通道 (%.1f%%)" % (
        n_ok, n_turns, n_qa_turns, 100.0 * n_qa_turns / max(n_turns, 1)))
    print("输出: %s" % out_jsonl)
    print("进度: %s (可续跑)" % progress_path)


if __name__ == "__main__":
    main()

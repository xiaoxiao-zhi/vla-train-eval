#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工 VLA 样例数据构建 (试运行: 每条轨迹插 1 个 QA).

设计 (2026-09-22 与用户对齐):
  - 样本 = 整段轨迹的所有帧 (每帧一轮交互)
  - 任务指令仅出现在第 0 轮 user 消息
  - QA 问题插在随机帧 k 的 user 消息 (question + observation)
  - assistant 输出 = 动作 token (每轮必有) + 文本 token (QA 期间附加)
  - 每轮文本段 token 数 == 该轮动作 token 数 N (等长双通道, 保证节拍一致)
  - QA 答案按 N 切片, 从第 k 轮起逐轮分配, 用尽后恢复纯动作
  - **主存 token id** (文本仅作可读副本: decode 再 encode 不保证还原)

输出: duplex_vla/samples/sample_<i>.json
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

SFT = "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft"
QA = "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/general_qa/train.jsonl"
TOKENIZER_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B")
OUT_DIR = "/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/samples"


def load_trajectories(limit_frames=(20, 60), want=5, seed=0):
    """从 train.jsonl 聚合完整轨迹: base -> {t: (title, action_text, frame_jpg)}."""
    rng = random.Random(seed)
    segs = defaultdict(dict)
    titles = {}
    with open(os.path.join(SFT, "train.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            msgs = d["messages"]
            # history=1 格式: 末条 assistant = 帧t 的动作标签, images[-1] = 帧t
            action = msgs[-1]["content"].strip()
            frame = d["images"][-1]
            m = re.match(r"(.+)_f(\d+)\.jpg$", os.path.basename(frame))
            if not m:
                continue
            base, t = m.group(1), int(m.group(2))
            segs[base][t] = action
            titles[base] = msgs[0]["content"].split("\nobservation")[0]
    # 挑帧数合适的片段
    cand = [b for b, steps in segs.items() if limit_frames[0] <= len(steps) <= limit_frames[1]]
    rng.shuffle(cand)
    out = []
    for b in cand[:want]:
        steps = sorted(segs[b].items())  # [(t, action_text)]
        out.append({
            "base": b,
            "title": titles[b],
            "turns": [{"t": t, "action": a} for t, a in steps],
        })
    return out


def load_qa(want_pool=50, seed=1, max_answer_tokens=60):
    """从 general_qa 挑答案较短的 QA 对."""
    rng = random.Random(seed)
    pool = []
    with open(QA) as f:
        for line in f:
            d = json.loads(line)
            q = d["messages"][0]["content"]
            a = d["messages"][1]["content"]
            if 5 <= len(a) <= 200:  # 粗过滤, token 层面再筛
                pool.append((q, a))
    rng.shuffle(pool)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    trajs = load_trajectories(want=args.n)
    qa_pool = load_qa()
    qa_idx = 0
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, traj in enumerate(trajs):
        turns = traj["turns"]
        n = len(turns)
        # 每轮动作 token id (保留字每个是单 id, 不含末尾换行)
        for tn in turns:
            ids = tok(tn["action"], add_special_tokens=False)["input_ids"]
            tn["action_ids"] = ids
            tn["action_len"] = len(ids)
        # QA: 随机插入点 k (避开首尾), 答案 token 数 <= 轨迹剩余容量
        k = random.randint(3, max(3, n - 4))
        # 找一条答案能在剩余轮内说短的 QA
        while True:
            q, a = qa_pool[qa_idx % len(qa_pool)]; qa_idx += 1
            ans_ids = tok(a, add_special_tokens=False)["input_ids"]
            if len(ans_ids) <= 60:
                break
        # 按每轮动作 token 数切片分配文本段
        segs, pos = [], 0
        for j, tn in enumerate(turns[k:], start=k):
            N = tn["action_len"]
            seg = ans_ids[pos:pos + N]
            if not seg:
                break
            segs.append(seg)
            pos += len(seg)
            tn["text_ids"] = seg
            tn["text"] = tok.decode(seg)  # 可读副本 (仅展示, 训练以 ids 为准)
        sample = {
            "base": traj["base"],
            "instruction": traj["title"],
            "qa": {"question": q, "answer": a, "answer_ids": ans_ids,
                   "insert_step": k, "answer_token_total": len(ans_ids)},
            "turns": [{
                "t": tn["t"],
                "user_text": (("%s\nobservation: \n" % traj["title"]) if tn["t"] == 0 else
                              (("%s\nobservation: \n" % q) if tn["t"] == k else "observation: \n")),
                "frame": "%s_f%06d.jpg" % (traj["base"], tn["t"]),
                "action_ids": tn["action_ids"],
                "action": tn["action"],
                "text_ids": tn.get("text_ids"),
                "text": tn.get("text"),
            } for tn in turns],
        }
        p = os.path.join(OUT_DIR, "sample_%d.json" % i)
        json.dump(sample, open(p, "w"), ensure_ascii=False, indent=1)
        # 摘要
        nqa = sum(1 for tn in turns if tn.get("text_ids"))
        print("[%d] %s: %d 轮, QA插在 t=%s, 答案 %d token 分 %d 轮 (每轮=动作token数)" % (
            i, traj["base"][:40], n, turns[k]["t"], len(ans_ids), nqa))
        print("    Q: %s" % q[:80])
        print("    A: %s" % a[:80])


if __name__ == "__main__":
    main()

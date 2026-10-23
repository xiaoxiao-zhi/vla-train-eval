#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建通用 QA 数据集 (ms-swift 标准 messages 格式 jsonl).

配比 (默认全量): trivia_qa ~138k + web_questions ~5.8k + alpaca ~52k
                + mc-qa (Minecraft 知识 QA) ~276k  => 约 47 万行.

来源:
  - trivia_qa:    rc.nocontext/train.parquet, question -> user,
                  answer.value (缺失时从 aliases 随机选一个) -> assistant
  - web_questions: train + test 合并, 多答案随机选一个
  - alpaca:       train.parquet, query = instruction\ninput, output -> assistant
  - mc-qa:        mincraft-vlp/mc-qa-250312.jsonl (官方 MC-VLP 数据, 276k),
                  conversations[{role, content:[{type,text}]}] -> messages,
                  保留原 system/user/assistant 轮次 (含少量多轮)

输出:
  <out-dir>/train.jsonl   (约 472k 行, 已全局 shuffle)
  <out-dir>/stats.json

用法 (默认全量):
  python make_general_qa.py --out-dir <dataset>/general_qa [--seed 42]
  (--n-xxx 0 表示全量; 传正数则采样)
"""
import argparse
import json
import os
import random
from collections import Counter

import pyarrow.parquet as pq

BASE = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/"
        "share_303556863/chengzhixu/dataset")


def sample(rows, n, rng, name):
    if n <= 0 or len(rows) < n:
        if n > 0:
            print("[warn] %s: 可用 %d < 目标 %d, 全量采用" % (name, len(rows), n))
        return list(rows)
    return rng.sample(rows, n)


def load_trivia_qa(n, rng):
    path = os.path.join(BASE, "trivia_qa", "rc.nocontext", "train-00000-of-00001.parquet")
    rows = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2048, columns=["question", "answer"]):
        for d in batch.to_pylist():
            q = (d["question"] or "").strip()
            ans = d["answer"] or {}
            a = (ans.get("value") or "").strip()
            if not a:
                aliases = [x.strip() for x in (ans.get("aliases") or []) if x and x.strip()]
                if aliases:
                    a = rng.choice(aliases)
            if q and a:
                rows.append((q, a))
    print("trivia_qa 有效样本: %d" % len(rows))
    return sample(rows, n, rng, "trivia_qa")


def load_web_questions(n, rng):
    rows = []
    for split in ["train", "test"]:
        path = os.path.join(BASE, "web_question", "data", "%s-00000-of-00001.parquet" % split)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512, columns=["question", "answers"]):
            for d in batch.to_pylist():
                q = (d["question"] or "").strip()
                answers = [a.strip() for a in (d["answers"] or []) if a and a.strip()]
                if q and answers:
                    rows.append((q, rng.choice(answers)))
    print("web_questions 有效样本: %d" % len(rows))
    return sample(rows, n, rng, "web_questions")


def load_alpaca(n, rng):
    data_dir = os.path.join(BASE, "alpaca", "data")
    f = [x for x in os.listdir(data_dir) if x.endswith(".parquet")][0]
    pf = pq.ParquetFile(os.path.join(data_dir, f))
    rows = []
    for batch in pf.iter_batches(batch_size=2048, columns=["instruction", "input", "output"]):
        for d in batch.to_pylist():
            inst = (d["instruction"] or "").strip()
            inp = (d["input"] or "").strip()
            out = (d["output"] or "").strip()
            if not inst and not inp:
                continue
            q = "%s\n%s" % (inst, inp) if (inst and inp) else (inst or inp)
            if q and out:
                rows.append((q, out))
    print("alpaca 有效样本: %d" % len(rows))
    return sample(rows, n, rng, "alpaca")


def load_mc_qa(n, rng):
    """官方 MC-VLP 的 Minecraft 知识 QA (276k): 配方/生物/机制等.
    conversations -> messages, 保留 system 与多轮."""
    path = os.path.join(BASE, "mincraft-vlp", "mc-qa-250312.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            convs = d.get("conversations") or []
            msgs = []
            for c in convs:
                texts = [x.get("text", "") for x in (c.get("content") or [])
                         if x.get("text")]
                text = "\n".join(t for t in texts if t).strip()
                if not text:
                    continue
                msgs.append((c.get("role") or "user", text))
            if len(msgs) >= 2 and msgs[-1][0] == "assistant":
                rows.append(msgs)
    print("mc-qa 有效样本: %d" % len(rows))
    return sample(rows, n, rng, "mc-qa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(BASE, "general_qa"))
    ap.add_argument("--n-trivia", type=int, default=0, help="0=全量")
    ap.add_argument("--n-webq", type=int, default=0, help="0=全量")
    ap.add_argument("--n-alpaca", type=int, default=0, help="0=全量")
    ap.add_argument("--n-mcq", type=int, default=-1,
                    help="-1=不加载(默认), 0=全量, >0=采样 (Minecraft 知识 QA)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_rows = []  # (source, messages)

    for source, loader, n in [
            ("trivia_qa", load_trivia_qa, args.n_trivia),
            ("web_questions", load_web_questions, args.n_webq),
            ("alpaca", load_alpaca, args.n_alpaca),
            ("mc_qa", load_mc_qa, args.n_mcq if args.n_mcq >= 0 else None)]:
        if n is None:
            continue
        for item in loader(n, rng):
            if source == "mc_qa":
                msgs = [{"role": r, "content": t} for r, t in item]
            else:
                q, a = item
                msgs = [{"role": "user", "content": q},
                        {"role": "assistant", "content": a}]
            all_rows.append((source, msgs))

    rng.shuffle(all_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "train.jsonl")
    with open(out_path, "w") as f:
        for source, msgs in all_rows:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    stats = {
        "total": len(all_rows),
        "by_source": dict(Counter(s for s, _ in all_rows)),
        "seed": args.seed,
        "path": out_path,
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("DONE:", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

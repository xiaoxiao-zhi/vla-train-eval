#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""官方 minecraft-vla-sft parquet -> 纯文本 jsonl (无图片数据/路径).

每行: {"id", "label", "instruction", "action"}
  - label:        原始标签数组 (含结构化任务标签 label[2])
  - instruction:  user 消息文本部分 (去掉 <image> 占位)
  - action:       assistant 动作保留字序列

并行: --workers 进程池, 每分片一个任务, 输出 part 文件后合并.
用法 (GPU 机 swift 环境):
  python parquet_to_textjsonl.py [--workers 16]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

DATA_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
            "chengzhixu/dataset/minecraft-vla-sft")
OUT_PATH = os.path.join(DATA_DIR, "train_text.jsonl")


def conv_text(content):
    """swift 多模态 content (list of {type,text}) -> 纯文本 (去掉 <image> 占位)."""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            t = item.get("text", "")
            if t.strip() != "<image>":
                parts.append(t)
    return "".join(parts).strip()


def convert_shard(pq_path):
    """一个 parquet 分片 -> 一个 part jsonl. 返回 (路径, 行数)."""
    import pyarrow.parquet as pq
    out_part = pq_path + ".part"
    n = 0
    table = pq.read_table(pq_path, columns=["id", "label", "conversations"])
    ids = table.column("id").to_pylist()
    labels = table.column("label").to_pylist()
    convs = table.column("conversations").to_pylist()
    with open(out_part, "w", encoding="utf-8") as f:
        for rid, label, conv in zip(ids, labels, convs):
            instruction, action = "", ""
            for msg in conv:
                role = msg.get("role")
                text = conv_text(msg.get("content"))
                if role == "user":
                    instruction = text
                elif role == "assistant":
                    action = text
            f.write(json.dumps({
                "id": rid,
                "label": list(label) if label else [],
                "instruction": instruction,
                "action": action,
            }, ensure_ascii=False) + "\n")
            n += 1
    return out_part, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(DATA_DIR, "data", "train-*.parquet")))
    print("分片数 %d, 并行 %d" % (len(shards), args.workers), flush=True)

    total = 0
    parts = []
    with Pool(args.workers) as pool:
        for i, (part, n) in enumerate(pool.imap_unordered(convert_shard, shards)):
            parts.append(part)
            total += n
            if (i + 1) % 20 == 0 or i + 1 == len(shards):
                print("已转换 %d/%d 分片, 累计 %d 行" % (i + 1, len(shards), total), flush=True)

    # 按分片序合并 (保证 id 顺序稳定)
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for part in sorted(parts):
            with open(part, encoding="utf-8") as f:
                for line in f:
                    out.write(line)
            os.remove(part)
    print("完成: %s 共 %d 行" % (OUT_PATH, total), flush=True)


if __name__ == "__main__":
    main()

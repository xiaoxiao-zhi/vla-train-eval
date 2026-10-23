#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段二数据准备: mcvla VLA 样本 + 通用 QA 样本 合并成一个流式 jsonl.

  - mcvla_sft_50_200/train.jsonl (50~200帧筛选, ~424k 行, messages + images)
  - general_qa/train.jsonl        (100,000 行, 纯文本 messages)
  -> stage2/train.jsonl           (~524k 行, 全局 shuffle, 固定种子可复现)

不做任何行内容改写 (图片是绝对路径引用, 无需拷贝;
注意 mcvla_sft_50_200/frames 是指向 mcvla_sft/frames 的软链, 帧复用).
用法:
  python merge_stage2_data.py [--seed 42]
"""
import argparse
import json
import os
import random

BASE = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/"
        "share_303556863/chengzhixu/dataset")
MCVLA = os.path.join(BASE, "mcvla_sft_50_200", "train.jsonl")
QA = os.path.join(BASE, "general_qa", "train.jsonl")
OUT_DIR = os.path.join(BASE, "mcvla_stage2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mcvla", default=MCVLA)
    ap.add_argument("--qa", default=QA)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    lines = []
    for path, name in [(args.mcvla, "mcvla"), (args.qa, "general_qa")]:
        n = 0
        with open(path) as f:
            for line in f:
                if line.strip():
                    lines.append(line)
                    n += 1
        print("%s: %d 行" % (name, n))

    random.Random(args.seed).shuffle(lines)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "train.jsonl")
    with open(out, "w") as f:
        f.writelines(lines)
    print("DONE: %d 行 -> %s" % (len(lines), out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把两轮格式 (title+prev图 / prev_action / cur图 / cur_action) 的 train.jsonl
转换成单轮格式 (title+cur图 / cur_action), 帧文件复用, 无需重新抽帧.

旧 t>0 行: 4 messages + 2 images
新行 (统一): 2 messages + 1 image
  user:      "<title>\nobservation: \n<image>"
  assistant: "<cur_action>"
  images:    [cur_frame]

用法: python convert_to_single_turn.py [--in ...] [--out ...] [--limit N(调试)]
"""
import argparse
import json
import os


def convert_row(d):
    msgs = d["messages"]
    if len(msgs) == 2:  # t=0 已是单轮
        return {"messages": [msgs[0], msgs[1]], "images": d["images"][:1]}
    if len(msgs) != 4:
        return None
    # title = user[0] 里 "observation:" 之前的部分
    user0 = msgs[0]["content"]
    idx = user0.find("observation:")
    title = user0[:idx] if idx >= 0 else user0
    # 去掉 title 末尾多余换行, 统一拼接
    title = title.rstrip("\n")
    cur_action = msgs[3]["content"]
    cur_frame = d["images"][1]  # images = [prev, cur]
    return {
        "messages": [
            {"role": "user", "content": "%s\nobservation: \n<image>" % title},
            {"role": "assistant", "content": cur_action},
        ],
        "images": [cur_frame],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=("/apdcephfs/share_chengzhixu/cephfs/"
                   "apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/train.jsonl"))
    ap.add_argument("--out", default=("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/"
                   "share_303556863/chengzhixu/dataset/mcvla_sft/train_single.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="调试: 只转前 N 行")
    args = ap.parse_args()

    n_in = n_out = n_skip = 0
    with open(args.inp) as fi, open(args.out, "w") as fo:
        for line in fi:
            if args.limit and n_in >= args.limit:
                break
            n_in += 1
            d = json.loads(line)
            r = convert_row(d)
            if r is None:
                n_skip += 1
                continue
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
            if n_out % 50000 == 0:
                print("  %d rows ..." % n_out, flush=True)
    print("输入 %d 行 -> 输出 %d 行 (跳过 %d)" % (n_in, n_out, n_skip))
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()

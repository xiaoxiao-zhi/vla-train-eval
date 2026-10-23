#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""展示一条训练样本的实际 input_ids / labels 对照 (调试用)."""
import sys

sys.path.insert(0, "/apdcephfs/private_chengzhixu/ms-swift/duplex_vla")
from duplex_dataset import DuplexStreamingDataset

ds = DuplexStreamingDataset(
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/duplex_vla/duplex_turns_train.jsonl",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B",
    shuffle_buffer=64, seed=0, max_samples=500)
tok = ds._renderer.tok
CTRL = {v: k.replace("<|duplex_", "").replace("|>", "") for k, v in ds._renderer.CTRL.items()}
CTRL[ds._renderer.SIL] = "SIL"

for d in ds:
    if d["txt_span"][1] - d["txt_span"][0] > 2:
        break
ids, labels = d["input_ids"], d["labels"]
a0, a1 = d["act_span"]
t0, t1 = d["txt_span"]
n = len(ids)
PAD = ds._renderer.img_pad


def name(i):
    if ids[i] in CTRL:
        return CTRL[ids[i]]
    tk = tok.convert_ids_to_tokens(ids[i])
    if tk == "<|image_pad|>":
        return "PAD"
    if tk is None:
        return "<id:%d>" % ids[i]
    return tk.replace("<|reserved_special_token_", "R").replace("|>", "")


def show(lo, hi, lab):
    out, i = [], lo
    while i < hi:
        if ids[i] == PAD:
            j = i
            while j < hi and ids[j] == PAD:
                j += 1
            out.append("PAD*%d" % (j - i))
            i = j
        else:
            out.append(name(i))
            i += 1
    print("%-16s ids[%d:%d]:" % (lab, lo, hi))
    print("                 " + " ".join(out)[:118])
    print("  labels:         " + " ".join(str(x) if x != -100 else "." for x in labels[lo:hi])[:118])


print("总长 %d | 图像 %d 帧 | labels有效 %d" % (n, d["n_images"], sum(1 for x in labels if x != -100)))
print()
show(0, 22, "chat外壳+片段0头")
# 历史片段中首个带OBS的(近K轮)与首个窗口外QA相关片段
runs = []
i = 0
while i < n:
    if ids[i] == PAD:
        j = i
        while j < n and ids[j] == PAD:
            j += 1
        runs.append((i, j))
        i = j
    else:
        i += 1
for r, (i, j) in enumerate(runs):
    show(max(0, i - 2), min(n, j + 2), "历史帧%d前后" % r)
show(a0 - 12, n, "当前轮+目标段")
print()
print("== 目标段逐 token (id <-> label) ==")
for i in range(a0, n):
    print("  pos %4d: id=%6d  %-18s label=%s" % (i, ids[i], name(i), labels[i]))

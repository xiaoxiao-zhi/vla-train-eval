#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把训练样本序列完整 decode 成可读文本, 写入文件."""
import json

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
    "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B")

d = json.load(open("/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/sample_ids_labels.json"))
ids, labels = d["input_ids"], d["labels"]
a0, a1 = d["act_span"]
t0, t1 = d["txt_span"]

CTRL = {151908: "<user>", 151909: "<obs>", 151910: "<act>",
        151911: "<txt>", 151912: "<silence>"}

out, pos = [], 0
while pos < len(ids):
    x = ids[pos]
    if x in CTRL:
        out.append(CTRL[x])
    elif x == 151655:  # image_pad 折叠
        j = pos
        while j < len(ids) and ids[j] == 151655:
            j += 1
        out.append("<图像x%d>" % (j - pos))
        pos = j
        continue
    elif x == 151835:
        out.append("<BEGIN>")
    elif x == 151836:
        out.append("<END>")
    elif x == 151644:
        out.append("\n<|im_start|>")
    elif x == 151645:
        out.append("<|im_end|>\n")
    elif x == 198:
        out.append("\n")
    else:
        s = tok.decode([x])
        # 动作保留字 151837~151911 区间内的简化显示
        tk = tok.convert_ids_to_tokens(x)
        if tk and tk.startswith("<|reserved_special_token_"):
            out.append("R%s" % tk.replace("<|reserved_special_token_", "").replace("|>", ""))
        else:
            out.append(s)
    pos += 1

text = "".join(out)
lines = [
    "=" * 70,
    "全双工训练样本完整序列 (总 %d token, 其中 labels 有效 %d 个)" % (
        len(ids), sum(1 for x in labels if x != -100)),
    "act_span=%s txt_span=%s  (目标段 = [act]..动作..[txt]..文本..<im_end>)" % (
        d["act_span"], d["txt_span"]),
    "=" * 70,
    "",
]
# 全文, 目标段前插标记行
before = text[:a0]
target = text[a0:]
lines.append(before)
lines.append("\n" + "▼" * 35 + " 目标段开始 (唯一计算 loss 的部分) " + "▼" * 35 + "\n")
lines.append(target)
lines.append("\n" + "=" * 70)
lines.append("目标段逐 token:")
for i in range(a0, len(ids)):
    tk = tok.convert_ids_to_tokens(ids[i])
    nm = CTRL.get(ids[i], tk if tk else "?")
    if ids[i] == 151835:
        nm = "<BEGIN>"
    if ids[i] == 151836:
        nm = "<END>"
    if ids[i] == 151645:
        nm = "<|im_end|>"
    lines.append("  pos %4d  id=%6d  %-16s label=%s" % (i, ids[i], nm, labels[i]))

open("/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/sample_full_text.txt", "w").write("\n".join(lines))
print("已写出: duplex_vla/sample_full_text.txt (%d 字符)" % len("\n".join(lines)))

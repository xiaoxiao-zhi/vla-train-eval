#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: 清单视频 -> 对应 json 的存在性."""
import json
import os

LIST = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
        "chengzhixu/dataset/mcvla_train_videos.json")
HARD = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
        "chengzhixu/dataset/minecraft_action/"
        "Player0-f153ac423f61-20211202-205900-1043-1083.json")

vids = json.load(open(LIST))
v0 = vids[0][:-5] + ".json"
print("v0:", repr(v0))
print("v0 == HARD:", v0 == HARD)
print("exists(HARD):", os.path.exists(HARD))
print("exists(v0):", os.path.exists(v0))
try:
    st = os.stat(HARD)
    print("stat(HARD) ok, size:", st.st_size)
except OSError as e:
    print("stat(HARD) error:", e)

d = os.path.dirname(v0)
names = os.listdir(d)
print("listdir 条数:", len(names))
print("HARD 文件名 in listdir:", os.path.basename(HARD) in names)
# 找一个实际存在的条目看前缀
sample = [n for n in names if n.endswith(".json")][:2]
print("listdir json 样例:", sample)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 train_text.jsonl 聚合视频段级任务表.

每个视频段 (视频id + 起始帧号唯一标识, 即 id 去掉行内帧号后缀) 输出一行:
  {"video_id", "start_frame", "end_frame", "n_frames", "task", "instruction"}

- task/instruction: 段内标签与指令 (已验证段内 100% 单一任务; 指令唯一性顺带校验,
  若段内指令不唯一则保留列表并标记)
"""
import json
import os
from collections import defaultdict

SRC = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
       "chengzhixu/dataset/minecraft-vla-sft/train_text.jsonl")
DST = os.path.join(os.path.dirname(SRC), "video_tasks.jsonl")

stats = {}  # video_id -> {"tasks": set, "instrs": set, "frames": [min, max, n]}

with open(SRC, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        vid = d["id"].rsplit("_", 1)[0]
        frame = int(d["id"].rsplit("_", 1)[1])
        task = d["label"][2] if len(d["label"]) > 2 else "?"
        s = stats.get(vid)
        if s is None:
            stats[vid] = {"tasks": {task}, "instrs": {d["instruction"]},
                          "min_f": frame, "max_f": frame, "n": 1}
        else:
            s["tasks"].add(task)
            s["instrs"].add(d["instruction"])
            s["min_f"] = min(s["min_f"], frame)
            s["max_f"] = max(s["max_f"], frame)
            s["n"] += 1

multi_task = sum(1 for s in stats.values() if len(s["tasks"]) > 1)
multi_instr = sum(1 for s in stats.values() if len(s["instrs"]) > 1)
with open(DST, "w", encoding="utf-8") as out:
    for vid in sorted(stats):
        s = stats[vid]
        out.write(json.dumps({
            "video_id": vid,
            "start_frame": s["min_f"],
            "end_frame": s["max_f"],
            "n_frames": s["n"],
            "task": sorted(s["tasks"])[0] if len(s["tasks"]) == 1 else sorted(s["tasks"]),
            "instruction": sorted(s["instrs"])[0] if len(s["instrs"]) == 1 else sorted(s["instrs"]),
        }, ensure_ascii=False) + "\n")

print("视频段数: %d  (多任务段: %d, 多指令段: %d)" % (len(stats), multi_task, multi_instr))
print("输出: %s" % DST)

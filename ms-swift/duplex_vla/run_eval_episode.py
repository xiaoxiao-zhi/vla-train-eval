#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单局评测 (独立进程入口, 由 eval_4tasks_sub.py 调度).

用法: python run_eval_episode.py <model_tag> <task> <ep> <out_root>
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, "/apdcephfs/private_chengzhixu/JarvisVLA")
os.chdir("/apdcephfs/private_chengzhixu/JarvisVLA/jarvisvla/evaluate")

from jarvisvla.evaluate.evaluate import evaluate

MODELS = {
    "stage2": {
        "base_url": "http://localhost:8000/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/jarvisvla_s2/v0-20260922-172519/checkpoint-2423-qwen2-vl"),
    },
    "official": {
        "base_url": "http://localhost:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B"),
    },
}
TASKS = {
    "kill/kill_zombie": ("normal", 2, 300),
    "mine/mine_stone": ("normal", 2, 500),
    "craft/craft_crafting_table": ("recipe", 4, 400),
    "smelt/iron_ingot": ("recipe", 4, 400),
}


def main():
    model_tag, task, ep, out_root = sys.argv[1:5]
    itype, hist, frames = TASKS[task]
    m = MODELS[model_tag]
    ep_dir = os.path.join(out_root, model_tag, task.replace("/", "_"), "ep%s" % ep)
    os.makedirs(ep_dir, exist_ok=True)
    t0 = time.time()
    try:
        ok, steps = evaluate(
            video_path=os.path.join(ep_dir, "video.mp4"),
            checkpoints=m["ckpt"],
            environment_config=dict(env_config=task, max_frames=frames, verbos=False),
            base_url=m["base_url"],
            model_config=dict(temperature=0.9, history_num=hist,
                              instruction_type=itype, action_chunk_len=1),
            io_log_dir=None)
    except Exception as e:
        traceback.print_exc()
        json.dump({"ok": False, "steps": -1, "err": str(e)},
                  open(os.path.join(ep_dir, "result.json"), "w"))
        print("[%s %s ep%s] EXC %s" % (model_tag, task, ep, str(e)[:100]), flush=True)
        return
    json.dump({"ok": bool(ok), "steps": steps},
              open(os.path.join(ep_dir, "result.json"), "w"))
    print("[%s %s ep%s] success=%s steps=%s %.0fs" % (
        model_tag, task.split("/")[1], ep, ok, steps, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

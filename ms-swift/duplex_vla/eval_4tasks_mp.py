#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""4 任务 action 评测 (训练前 vs 训练后), 多进程并行 (不用 ray), 每任务每模型 3 局.

任务配置遵循官方 rollout-*.sh:
  kill/kill_zombie:          normal, hist=2, frames=300
  mine/mine_stone:           normal, hist=2, frames=500
  craft/craft_crafting_table: recipe, hist=4, frames=400
  smelt/iron_ingot:          recipe, hist=4, frames=400

用法 (GPU 机 JarvisVLA 环境):
  python eval_4tasks_mp.py --out <目录> [--workers 4] [--repeats 3]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, "/apdcephfs/private_chengzhixu/JarvisVLA")
os.chdir("/apdcephfs/private_chengzhixu/JarvisVLA/jarvisvla/evaluate")

MODELS = {
    "stage2": {
        "base_url": "http://localhost:8000/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/jarvisvla_s2/v0-20260922-172519/checkpoint-2423-vllm"),
    },
    "official": {
        "base_url": "http://localhost:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B"),
    },
}

# task -> (instruction_type, history_num, max_frames)  官方 rollout-*.sh 同款
TASKS = {
    "kill/kill_zombie": ("normal", 2, 300),
    "mine/mine_stone": ("normal", 2, 500),
    "craft/craft_crafting_table": ("recipe", 4, 400),
    "smelt/iron_ingot": ("recipe", 4, 400),
}


def run_one(job):
    """单局评测 (独立子进程; Minecraft 单实例)."""
    model_tag, task, ep, out_root = job
    from jarvisvla.evaluate.evaluate import evaluate
    itype, hist, frames = TASKS[task]
    m = MODELS[model_tag]
    ep_dir = os.path.join(out_root, model_tag, task.replace("/", "_"), "ep%d" % ep)
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
        print("[%s %s ep%d] 异常: %s" % (model_tag, task, ep, str(e)[:120]), flush=True)
        ok, steps = False, -1
    print("[%s %s ep%d] success=%s steps=%s %.0fs" % (
        model_tag, task.split("/")[1], ep, ok, steps, time.time() - t0), flush=True)
    return (model_tag, task, ep, bool(ok), steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/apdcephfs/private_chengzhixu/JarvisVLA/outputs/rollout/eval_4tasks")
    ap.add_argument("--workers", type=int, default=4, help="并行 Minecraft 实例数")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    jobs = [(mt, task, ep, args.out)
            for mt in MODELS for task in TASKS for ep in range(args.repeats)]
    print("总任务: %d 局 (2模型 x 4任务 x %d局), 并行 %d" % (
        len(jobs), args.repeats, args.workers), flush=True)

    results = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            results.append(r)

    # 汇总
    print()
    print("=" * 62)
    print("%-28s %-10s %-10s" % ("任务", "official", "stage2"))
    print("-" * 62)
    summary = {}
    for task in TASKS:
        row = {}
        for mt in MODELS:
            eps = [r for r in results if r[0] == mt and r[1] == task]
            row[mt] = "%d/%d" % (sum(e[3] for e in eps), len(eps))
        summary[task] = row
        print("%-28s %-10s %-10s" % (task, row["official"], row["stage2"]))
    json.dump({"results": results, "summary": summary},
              open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("=" * 62)
    print("视频: %s/<model>/<task>/ep*/video.mp4" % args.out)


if __name__ == "__main__":
    main()

import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""4任务x2模型x3局 评测调度器: subprocess 独立进程 (无 ray / 无 fork Pool).

每局一个独立 python 进程 (run_eval_episode.py), 并发 --workers 路,
错峰启动 (每路间隔 --stagger 秒, 避开 Minecraft 实例启动竞争).
结果由各局 result.json 汇总.
"""
import argparse
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
      "chengzhixu/conda_envs/JarvisVLA/bin/python")

MODELS = ["stage2", "official"]
TASKS = ["kill/kill_zombie", "mine/mine_stone",
         "craft/craft_crafting_table", "smelt/iron_ingot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/apdcephfs/private_chengzhixu/JarvisVLA/outputs/rollout/eval_4tasks")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--stagger", type=float, default=10.0,
                    help="各并行实例错峰启动间隔秒")
    args = ap.parse_args()

    jobs = [(mt, task, ep) for mt in MODELS for task in TASKS for ep in range(args.repeats)]
    # 跳过已有 result.json 的局 (断点续跑)
    def done(job):
        mt, task, ep = job
        return os.path.exists(os.path.join(
            args.out, mt, task.replace("/", "_"), "ep%d" % ep, "result.json"))
    skipped = sum(1 for j in jobs if done(j))
    jobs = [j for j in jobs if not done(j)]
    # 长任务优先 (mine 500帧 > craft/smelt 400 > kill 300), 减少尾部等待
    order = {"mine/mine_stone": 0, "craft/craft_crafting_table": 1,
             "smelt/iron_ingot": 2, "kill/kill_zombie": 3}
    jobs.sort(key=lambda j: order[j[1]])
    print("总 %d 局 (跳过已完成 %d), 并发 %d, 错峰 %.0fs" % (
        len(jobs), skipped, args.workers, args.stagger), flush=True)

    running = []   # [(Popen, job, logf)]
    logs_dir = os.path.join(args.out, "_logs")
    os.makedirs(logs_dir, exist_ok=True)
    env = dict(os.environ, DISPLAY=":99")

    def launch(job):
        mt, task, ep = job
        tag = "%s_%s_ep%s" % (mt, task.split("/")[1], ep)
        logf = open(os.path.join(logs_dir, tag + ".log"), "w")
        p = subprocess.Popen(
            [PY, os.path.join(HERE, "run_eval_episode.py"), mt, task, str(ep), args.out],
            stdout=logf, stderr=subprocess.STDOUT, env=env)
        running.append((p, job, logf))
        print("启动: %s (pid %d)" % (tag, p.pid), flush=True)

    ji = 0
    last_launch = 0.0
    while ji < len(jobs) or running:
        # 补位启动 (错峰)
        while (ji < len(jobs) and len(running) < args.workers
               and time.time() - last_launch >= args.stagger):
            launch(jobs[ji])
            ji += 1
            last_launch = time.time()
        # 回收
        for item in running[:]:
            p, job, logf = item
            if p.poll() is not None:
                running.remove(item)
                logf.close()
                print("结束: %s_%s_ep%s rc=%d (剩 %d)" % (
                    job[0], job[1].split("/")[1], job[2], p.returncode,
                    len(running) + len(jobs) - ji), flush=True)
        time.sleep(5)

    # 汇总
    print()
    print("=" * 60)
    print("%-28s %-10s %-10s" % ("任务", "official", "stage2"))
    print("-" * 60)
    summary = {}
    for task in TASKS:
        row = {}
        for mt in MODELS:
            eps = []
            for ep in range(args.repeats):
                rp = os.path.join(args.out, mt, task.replace("/", "_"),
                                  "ep%d" % ep, "result.json")
                if os.path.exists(rp):
                    eps.append(json.load(open(rp)).get("ok", False))
            row[mt] = "%d/%d" % (sum(eps), len(eps))
        summary[task] = row
        print("%-28s %-10s %-10s" % (task, row["official"], row["stage2"]))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("=" * 60)


if __name__ == "__main__":
    main()

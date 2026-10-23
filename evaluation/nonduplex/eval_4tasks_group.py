import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""组级评测调度器: 每个 (model, task) 组一个进程 (run_eval_task.py),
组内复用 Minecraft 实例连续 fast-reset 跑 3 局.

- 断点续跑: 组内缺失的局才跑 (已有 result.json 的局跳过)
- 并发 --workers 组, 错峰 --stagger 秒
"""
import argparse
import json
import os
import subprocess
import time
import sys
import math
import signal

sys.path.insert(0, '/apdcephfs/private_chengzhixu/JarvisVLA')
from duplex.eval_server_urls import URLSlots, parse_base_urls
from duplex.vllm_eval_pool import terminate_owned

HERE = os.path.dirname(os.path.abspath(__file__))
PY = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
      "chengzhixu/conda_envs/JarvisVLA/bin/python")

MODELS = ["s2v3_hist1"]
TASKS = ["mine/mine_stone", "craft/craft_crafting_table",
         "smelt/iron_ingot", "kill/kill_zombie"]
REPEATS = 3


def pending_eps(out, mt, task):
    """该组还缺哪些局."""
    missing = []
    for ep in range(REPEATS):
        rp = os.path.join(out, mt, task.replace("/", "_"), "ep%d" % ep, "result.json")
        if not os.path.exists(rp):
            missing.append(ep)
        else:
            with open(rp) as f:
                result = json.load(f)
            if "err" in result or result.get("steps", -1) < 0:
                missing.append(ep)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/apdcephfs/private_chengzhixu/JarvisVLA/outputs/rollout/eval_4tasks")
    ap.add_argument("--models", default=None,
                    help="逗号分隔模型列表 (默认用文件顶部 MODELS)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stagger", type=float, default=10.0)
    ap.add_argument('--base-urls', default=os.environ.get('EVAL_BASE_URLS', ''),
                    help='每个并行任务独占一个vLLM服务地址')
    args = ap.parse_args()
    if args.workers < 1 or not math.isfinite(args.stagger) or args.stagger < 0:
        ap.error('Invalid workers or stagger')
    models = args.models.split(",") if args.models else MODELS
    urls = parse_base_urls(args.base_urls)
    if urls and len(models) != 1:
        ap.error('An independent server pool must serve one model')
    if urls:
        args.workers = min(args.workers, len(urls))
    slots = URLSlots(urls, args.workers)

    jobs = []  # [(mt, task, eps)]
    for mt in models:
        for task in TASKS:
            eps = pending_eps(args.out, mt, task)
            if eps:
                jobs.append((mt, task, eps))
    print("待跑 %d 组: %s" % (len(jobs), [(j[0], j[1].split("/")[1], j[2]) for j in jobs]), flush=True)

    logs_dir = os.path.join(args.out, "_logs")
    os.makedirs(logs_dir, exist_ok=True)
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":99"))  # 可用环境变量覆盖(并行评测分显)

    running = []
    ji = 0
    last_launch = 0.0
    assignments = {}
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        while ji < len(jobs) or running:
            while (ji < len(jobs) and len(running) < args.workers
                   and (not running or time.time() - last_launch >= args.stagger)):
                mt, task, eps = jobs[ji]
                tag = "%s_%s_eps%s" % (mt, task.split("/")[1], ",".join(map(str, eps)))
                logf = open(os.path.join(logs_dir, tag + ".log"), "a")
                url = slots.acquire(tag)
                child_env = dict(env)
                if url:
                    child_env['EVAL_BASE_URL'] = url
                    assignments[tag] = url
                try:
                    p = subprocess.Popen(
                        [PY, os.path.join(HERE, "run_eval_task.py"), mt, task, args.out,
                         ",".join(map(str, eps))],
                        stdout=logf, stderr=subprocess.STDOUT, env=child_env, start_new_session=True)
                except BaseException:
                    logf.close()
                    slots.release(tag)
                    raise
                running.append((p, jobs[ji], logf, tag))
                with open(os.path.join(logs_dir, mt + '_task_servers.json'), 'w') as file:
                    json.dump(assignments, file, indent=2)
                print("启动组: %s (pid %d, url=%s)" % (tag, p.pid, url or 'model default'), flush=True)
                ji += 1
                last_launch = time.time()
            for item in running[:]:
                p, job, logf, tag = item
                if p.poll() is not None:
                    running.remove(item)
                    slots.release(tag)
                    logf.close()
                    print("组结束: %s_%s rc=%d (剩 %d)" % (
                        job[0], job[1].split("/")[1], p.returncode,
                        len(running) + len(jobs) - ji), flush=True)
                    if p.returncode:
                        raise subprocess.CalledProcessError(p.returncode, p.args)
            if ji < len(jobs) or running:
                time.sleep(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        terminate_owned([item[0] for item in running])
        for _, _, logf, _ in running:
            logf.close()

    # 汇总
    print()
    print("=" * 60)
    print("-" * 60)
    summary_path = os.path.join(args.out, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    for task in TASKS:
        row = dict(summary.get(task, {}))
        for mt in models:
            eps = []
            for ep in range(REPEATS):
                rp = os.path.join(args.out, mt, task.replace("/", "_"), "ep%d" % ep, "result.json")
                if not os.path.exists(rp):
                    raise SystemExit("Missing episode result: %s" % rp)
                with open(rp) as f:
                    result = json.load(f)
                if "err" in result or result.get("steps", -1) < 0:
                    raise SystemExit("Invalid episode result: %s: %s" % (rp, result))
                eps.append(result["ok"])
            row[mt] = "%d/%d" % (sum(eps), len(eps))
        summary[task] = row
        print("%-28s %s" % (task, "  ".join("%s=%s" % kv for kv in row.items())))
    with open(summary_path + ".tmp", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(summary_path + ".tmp", summary_path)
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据生产者: 从 minecraft_action 流式生成 SFT 训练数据 (边抽帧边写 jsonl).

管线角色:
  生产者(本脚本, CPU): 逐片段 [抽帧 jpg -> 生成滑窗样本 -> 追加写 train.jsonl]
  消费者(GPU 训练):    swift sft --streaming true 流式读取 train.jsonl

样本格式 (由 --history N 控制历史窗口, N=0 为纯单帧):
  history=0 (默认, 单帧):  指令+当前帧 -> 当前动作
    {"messages": [
        {"role": "user",      "content": "<title>\\nobservation: \\n<image>"},
        {"role": "assistant", "content": "<动作t>"}],
     "images": ["<片段>_f(t).jpg"]}

  history=N (N>=1):  指令 + 过去N帧(及其动作) + 当前帧 -> 当前动作
    {"messages": [
        {"role": "user",      "content": "<title>\\nobservation: \\n<image>"},   # 帧 t-N
        {"role": "assistant", "content": "<动作t-N>"},
        ...
        {"role": "user",      "content": "observation: \\n<image>"},            # 帧 t-1
        {"role": "assistant", "content": "<动作t-1>"},
        {"role": "user",      "content": "observation: \\n<image>"},            # 帧 t
        {"role": "assistant", "content": "<动作t>"}],                          # <- 标签
     "images": ["_f(t-N).jpg", ..., "_f(t-1).jpg", "_f(t).jpg"]}               # N+1 张图

  边界: 片段开头不足 N 帧历史时按实际可用帧构造 (t=0 退化为单帧带 title).

  设计动机: 两轮格式 (history=1) 时, 模型学到 "抄上一帧动作" 捷径
  (训练集 35% 样本 cur==prev, fwd+jump 条件复制率 58.6%), rollout 陷入
  forward+jump 不动点; history=0 掐断该捷径的历史输入端, 强迫模型看图.
  history 保留为超参数, 方便后续对比实验.

顺序保证: 每个片段先抽帧后写行, jsonl 中出现的样本其 jpg 必然已存在.

断点重启: done_segments.jsonl 记录已完成片段, 重启自动跳过;
          抽帧幂等(已存在的 jpg 跳过), 崩溃片段重做不会产生坏数据.
          注意: 换 --history 后需清空 done_segments.jsonl 或换 --out-dir
          (否则已完成片段被跳过, 新格式不会重写).

用法:
  python make_sft_stream.py --out-dir /path/mcvla_sft [--history 0] [--stride 1] [--workers 8]
  (默认消费 minecraft_action 全部片段; stride=k 时每 k 帧取一个时间步)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert_actions_to_tokens import action_to_token_text  # noqa: E402

DEFAULT_ACTION_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/"
                      "share_303556863/chengzhixu/dataset/minecraft_action")

_write_lock = threading.Lock()


def load_json_safe(path):
    try:
        with open(path) as f:
            d = json.load(f)
        if d.get("actions") is None or not d.get("title"):
            return None
        return d
    except (ValueError, OSError):
        return None


def get_action_texts(d):
    if d.get("action_tokens_text"):
        return d["action_tokens_text"]
    return [action_to_token_text(a)[0] for a in d["actions"]]


def frame_path(frames_dir, base, idx):
    return os.path.join(frames_dir, "%s_f%06d.jpg" % (base, idx))


def extract_frames(mp4, indices, frames_dir, base):
    """一次 ffmpeg 调用抽出指定帧 (已存在的跳过). 返回成功抽到的帧集合."""
    need = [i for i in indices if not os.path.exists(frame_path(frames_dir, base, i))]
    done = set(indices) - set(need)
    if not need:
        return done
    os.makedirs(frames_dir, exist_ok=True)
    sel = "+".join("eq(n\\,%d)" % i for i in need)
    tmp_prefix = os.path.join(frames_dir, ".tmp_%s_%%05d.jpg" % base)
    # 存原始 640x360, 训练时由 Qwen processor 的 smart_resize 自动处理 (->644x364, ~299 token/图)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", mp4,
           "-vf", "select='%s'" % sel, "-vsync", "0", tmp_prefix]
    try:
        # timeout 防坏 mp4/IO 挂起导致整个生产进程卡死
        if subprocess.run(cmd, capture_output=True, timeout=600).returncode != 0:
            return done
    except subprocess.TimeoutExpired:
        print("  [timeout] ffmpeg 600s 未完成, 跳过片段 %s (need=%d帧)" % (
            base, len(need)), flush=True)
        return done
    # ffmpeg 序号从 1 开始, 按顺序映射回帧号
    for k, idx in enumerate(need, start=1):
        tmp = os.path.join(frames_dir, ".tmp_%s_%05d.jpg" % (base, k))
        dst = frame_path(frames_dir, base, idx)
        try:
            os.replace(tmp, dst)
            done.add(idx)
        except OSError:
            pass
    return done


def build_rows(base, title, texts, stride, frames_dir, history=0):
    """生成滑窗样本行: 指令 + 过去 history 帧(及动作) + 当前帧 -> 当前动作.
    history=0 即纯单帧. 片段开头不足 history 帧时按实际可用帧构造."""
    rows = []
    for t in range(0, len(texts), stride):
        first = max(0, t - history)
        msgs, imgs = [], []
        ok = True
        for k in range(first, t):
            jpg = frame_path(frames_dir, base, k)
            if not os.path.exists(jpg):
                ok = False
                break
            if k == first:
                msgs.append({"role": "user",
                             "content": "%s\nobservation: \n<image>" % title})
            else:
                msgs.append({"role": "user", "content": "observation: \n<image>"})
            msgs.append({"role": "assistant", "content": texts[k]})
            imgs.append(jpg)
        if not ok:
            continue
        cur_jpg = frame_path(frames_dir, base, t)
        if not os.path.exists(cur_jpg):
            continue
        if not msgs:  # history=0 或 t=0: 唯一 user 带 title
            msgs.append({"role": "user",
                         "content": "%s\nobservation: \n<image>" % title})
        else:
            msgs.append({"role": "user", "content": "observation: \n<image>"})
        msgs.append({"role": "assistant", "content": texts[t]})
        imgs.append(cur_jpg)
        rows.append({"messages": msgs, "images": imgs})
    return rows


def process_segment(jp, args, frames_dir):
    """处理一个片段: 抽帧 -> 返回样本行. 返回 (片段名, 行数) 或 None."""
    mp4 = jp[:-5] + ".mp4"
    if not os.path.exists(mp4):
        return None
    d = load_json_safe(jp)
    if d is None:
        return None
    texts = get_action_texts(d)
    if not texts:
        return None
    base = os.path.splitext(os.path.basename(jp))[0]
    title = d["title"].strip()

    # 该片段需要抽的帧: 每个时间步 t 的历史窗口 [t-history, t] 并集
    grid = list(range(0, len(texts), args.stride))
    need = sorted({k for t in grid for k in range(max(0, t - args.history), t + 1)})
    extract_frames(mp4, need, frames_dir, base)
    rows = build_rows(base, title, texts, args.stride, frames_dir, args.history)
    if not rows:
        return None
    return base, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-dir", default=DEFAULT_ACTION_DIR)
    ap.add_argument("--video-list", default=None,
                    help="视频路径清单 json (如 mcvla_train_videos.json); "
                         "提供后只处理清单内的片段, 忽略 action-dir 的全量扫描")
    ap.add_argument("--out-dir", required=True,
                    help="输出目录 (生成 train.jsonl / frames/ / progress.json)")
    ap.add_argument("--stride", type=int, default=1,
                    help="每 stride 帧取一个时间步样本 (1=全量)")
    ap.add_argument("--history", type=int, default=0,
                    help="历史窗口: 保留过去 N 帧观测及其动作作为上下文 "
                         "(0=纯单帧(默认), 1=原两轮格式, N>=2 多轮)")
    ap.add_argument("--workers", type=int, default=8, help="并行抽帧线程数")
    ap.add_argument("--limit", type=int, default=0, help="最多处理片段数(调试用)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    jsonl_path = os.path.join(args.out_dir, "train.jsonl")
    done_path = os.path.join(args.out_dir, "done_segments.jsonl")
    progress_path = os.path.join(args.out_dir, "progress.json")

    # 断点: 已完成片段
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            done = {l.strip() for l in f if l.strip()}
    print("已完成片段: %d" % len(done), flush=True)

    if args.video_list:
        # 按清单: mp4 路径 -> 对应 json, 需两者都存在 ('.mp4' 为 4 字符)
        with open(args.video_list) as f:
            vids = json.load(f)
        pairs = [v[:-4] + ".json" for v in vids]
        pairs = [p for p in pairs if os.path.exists(p)]
        print("清单视频: %d, 找到对应 json: %d" % (len(vids), len(pairs)), flush=True)
    else:
        pairs = sorted(glob.glob(os.path.join(args.action_dir, "*.json")))
    todo = [p for p in pairs if os.path.splitext(os.path.basename(p))[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print("待处理片段: %d / %d" % (len(todo), len(pairs)), flush=True)

    t0 = time.time()
    n_seg = n_row = 0
    with open(jsonl_path, "a") as out, open(done_path, "a") as done_f:
        def write(item):
            nonlocal n_row
            if item is None:
                return
            base, rows = item
            with _write_lock:
                for r in rows:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                done_f.write(base + "\n")
                done_f.flush()

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for item in ex.map(lambda jp: process_segment(jp, args, frames_dir), todo):
                write(item)
                if item is not None:
                    n_seg += 1
                    n_row += len(item[1])
                if n_seg and n_seg % 20 == 0:
                    speed = n_seg / (time.time() - t0)
                    with open(progress_path, "w") as pf:
                        json.dump({"segments_done": n_seg, "rows_written": n_row,
                                   "segments_per_sec": round(speed, 3),
                                   "elapsed_s": round(time.time() - t0, 1)}, pf)
                    print("[%d segs, %d rows, %.1f seg/s]" % (
                        n_seg, n_row, speed), flush=True)

    with open(progress_path, "w") as pf:
        json.dump({"segments_done": n_seg, "rows_written": n_row,
                   "finished": True,
                   "elapsed_s": round(time.time() - t0, 1)}, pf)
    print("DONE: %d segments, %d rows -> %s" % (n_seg, n_row, jsonl_path))


if __name__ == "__main__":
    main()

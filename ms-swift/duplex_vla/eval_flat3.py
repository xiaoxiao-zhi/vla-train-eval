#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stage2 flat3 评测: 4任务×3局, prompt 与 flat3 训练格式严格对齐.

flat3 训练格式 (build_stage2_flat3.py):
  user:   "指令\nobservation: \n<image>observation: \n<image>observation: \n<image>"
  asst:   "<|...178|>...动作...<|...179|>"          (无 \n\n 尾巴)
  images: [f_{t-2}, f_{t-1}, f_t] (段首退化少占位)

评测差异 (与 run_eval_task.py 的 VLLM_AGENT 相比):
  - 单轮拍平 (非多轮), 无历史动作 token, 无 thought/recipe 前缀
  - 指令用自然语言 (instructions.json, 同训练数据风格), craft/smelt 不用官方 recipe 协议
  - 动作 decode: 外部 AutoTokenizer (OneActionTokenizer 无 .tok, v4 初版教训),
    解析失败直接崩溃 (不静默兜底)

架构沿用 eval_4tasks_group/run_eval_task:
  - 调度模式 (默认): 每 (model,task) 组一个子进程, 断点续跑, 最后汇总 summary.json
  - --one 模式: 单组执行, Minecraft 只起一次, fast-reset 连跑各局

用法:
  先起 vLLM: vllm serve <ckpt> --port 8100 --served-model-name flat3 \
      --runner generate --limit-mm-per-prompt '{"image": 8}'
  python eval_flat3.py --out <目录> --workers 4
"""
import argparse
import base64
import io
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PY = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
      "chengzhixu/conda_envs/JarvisVLA/bin/python")
JARVIS = "/apdcephfs/private_chengzhixu/JarvisVLA"
INSTR_LIB = os.path.join(JARVIS, "jarvisvla/evaluate/assets/instructions.json")

MODELS = {
    "s2v4_flat3": {
        "base_url": "http://localhost:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/s2v4_flat3/v0-20260930-225227/"
                 "v0-20260930-225519/checkpoint-2900"),
    },
}
TASKS = {
    "kill/kill_zombie": 300,
    "mine/mine_stone": 500,
    "craft/craft_crafting_table": 400,
    "smelt/iron_ingot": 400,
}
HIST = 2       # flat3: 2 历史帧 + 当前帧 (段首退化)
REPEATS = 3
OBS = "observation: \n<image>"
ACT_RE = re.compile(r'(<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>)', re.S)


# ================= 单组执行 (子进程) =================

def load_task_cfg(task):
    import hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    config_name = Path("%s.yaml" % task).stem
    hydra.initialize(config_path=os.path.join("./config", Path(task).parent), version_base="1.3")
    return hydra.compose(config_name=config_name)


def build_env(cfg, record_dir):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.entry import CameraConfig
    from minestudio.simulator.callbacks import (
        SpeedTestCallback, RecordCallback, RewardsCallback, TaskCallback,
        FastResetCallback, SummonMobsCallback, CommandsCallback)
    from jarvisvla.evaluate.evaluate import _create_init_inventory_callback
    callbacks = [
        FastResetCallback(biomes=cfg.candidate_preferred_spawn_biome,
                          random_tp_range=cfg.random_tp_range,
                          start_time=cfg.start_time),
        SpeedTestCallback(50),
        TaskCallback(getattr(cfg, "task_conf", None)),
        RewardsCallback(getattr(cfg, "reward_conf", None)),
        _create_init_inventory_callback(cfg),
        CommandsCallback(getattr(cfg, "command", [])),
        RecordCallback(record_path=Path(record_dir), fps=30, show_actions=False),
    ]
    if cfg.mobs:
        callbacks.append(SummonMobsCallback(cfg.mobs))
    env = MinecraftSim(
        action_type="env", seed=cfg.seed,
        obs_size=cfg.origin_resolution, render_size=cfg.resize_resolution,
        camera_config=CameraConfig(**cfg.camera_config),
        preferred_spawn_biome=getattr(cfg, "preferred_spawn_biome", None),
        callbacks=callbacks)
    return env, callbacks


def img_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def vllm_chat(session, base_url, model_name, prompt, frames_pil):
    """flat3 单轮拍平 prompt -> vLLM chat. 与训练格式一致, 发啥存啥可回放."""
    import requests
    content = []
    img_i = 0
    parts = prompt.split(OBS)
    for i, seg in enumerate(parts):
        if i > 0:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + img_b64(frames_pil[img_i])}})
            img_i += 1
        if seg:
            content.append({"type": "text", "text": seg})
    r = session.post("%s/chat/completions" % base_url, json={
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 64, "temperature": 0,
        "skip_special_tokens": False,
    }, proxies={"http": None, "https": None}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def pick_instruction(env_prompt, rng):
    """自然语言指令 (instructions.json instruct 池, 同训练数据风格; 无 recipe/thought)."""
    lib = json.load(open(INSTR_LIB))
    instructs = lib.get(env_prompt, {}).get("instruct")
    if not instructs:
        natural = env_prompt.replace("_", " ").replace(":", " ")
        instructs = [natural]
    return rng.choice(instructs)


def run_one(model_tag, task, out_root, eps, seed):
    model = MODELS[model_tag]
    max_frames = TASKS[task]
    sys.path.insert(0, JARVIS)
    os.chdir(os.path.join(JARVIS, "jarvisvla/evaluate"))

    import requests
    import numpy as np
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_
    from PIL import Image
    from transformers import AutoTokenizer
    from jarvisvla.inference.action_mapping import OneActionTokenizer

    # 动作 decode: 外部 tokenizer (OneActionTokenizer 无 .tok), 启动自检
    tkz = AutoTokenizer.from_pretrained(model["ckpt"], trust_remote_code=True)
    act_tok = OneActionTokenizer(tokenizer_type="qwen2_vl")
    _chk = tkz("<|reserved_special_token_204|>", add_special_tokens=False)["input_ids"]
    assert len(_chk) == 1, "tokenizer 无法把动作特殊 token 映射为单 id: %s" % _chk

    session = requests.Session()
    task_dir = os.path.join(out_root, model_tag, task.replace("/", "_"))
    cfg = load_task_cfg(task)
    env, callbacks = build_env(cfg, os.path.join(task_dir, "ep%d" % eps[0]))
    from minestudio.simulator.callbacks import RecordCallback
    record_callback = next(c for c in callbacks if isinstance(c, RecordCallback))

    env_prompt = cfg.task_conf[0]["text"]
    rng = random.Random(seed)

    for ep in eps:
        ep_dir = os.path.join(task_dir, "ep%d" % ep)
        os.makedirs(ep_dir, exist_ok=True)
        t0 = time.time()
        try:
            record_callback.record_path = Path(ep_dir)
            env.action_type = "env"      # pre_agent 需要离散动作格式
            obs, info = env.reset()

            # GUI 预打开 (craft/smelt 任务定义的一部分, 与 run_eval_task 同款)
            pre_agent = None
            worker_type = getattr(cfg, "worker", None)
            if worker_type == "craft":
                from jarvisvla.evaluate.env_helper.craft_agent import CraftWorker
                pre_agent = CraftWorker(env, if_discrete=True)
            elif worker_type == "smelt":
                from jarvisvla.evaluate.env_helper.smelt_agent import SmeltWorker
                pre_agent = SmeltWorker(env, if_discrete=True)
            if getattr(cfg, "need_gui", False):
                if getattr(cfg, "need_crafting_table", False):
                    pre_agent.open_crating_table_wo_recipe()
                elif getattr(cfg, "need_furnace", False):
                    pre_agent.open_furnace_wo_recipe()

            env.action_type = "agent"
            instruction = pick_instruction(env_prompt, rng)
            hist = []                     # 历史帧 PIL (旧->新)
            ok, steps = False, max_frames
            for i in range(max_frames):
                cur = Image.fromarray(info["pov"])
                frames = hist[-HIST:] + [cur]     # 段首退化: 不足 HIST 时自然少
                prompt = instruction + "\n" + OBS * len(frames)
                out_text = vllm_chat(session, model["base_url"], model_tag, prompt, frames)

                # 动作解析: 必须有动作段, 没有/解码失败直接崩溃 (不静默兜底)
                m = ACT_RE.search(out_text)
                if not m:
                    raise RuntimeError("ep%d step%d: 输出无动作段: %r" % (ep, i, out_text[:120]))
                env_action = act_tok.decode(
                    tkz(m.group(1), add_special_tokens=False)["input_ids"])[0]

                obs, reward, terminated, truncated, info = env.step(env_action)
                hist.append(cur)
                if reward > 0:
                    ok, steps = True, i
                    for _ in range(20):   # 成功后 20 步 (对齐官方)
                        cur2 = Image.fromarray(info["pov"])
                        fr = hist[-HIST:] + [cur2]
                        p2 = instruction + "\n" + OBS * len(fr)
                        o2 = vllm_chat(session, model["base_url"], model_tag, p2, fr)
                        m2 = ACT_RE.search(o2)
                        if m2:
                            a2 = act_tok.decode(
                                tkz(m2.group(1), add_special_tokens=False)["input_ids"])[0]
                            obs, reward, terminated, truncated, info = env.step(a2)
                        hist.append(cur2)
                    break
            record_callback._save_episode()
            record_callback.episode_id += 1
            json.dump({"ok": bool(ok), "steps": steps},
                      open(os.path.join(ep_dir, "result.json"), "w"))
            print("[%s %s ep%s] ok=%s steps=%s %.0fs" % (
                model_tag, task.split("/")[1], ep, ok, steps, time.time() - t0), flush=True)
        except Exception as e:
            traceback.print_exc()
            json.dump({"ok": False, "steps": -1, "err": str(e)},
                      open(os.path.join(ep_dir, "result.json"), "w"))
            print("[%s %s ep%s] EXC %s" % (model_tag, task, ep, str(e)[:100]), flush=True)

    env.close()
    print("[%s %s] 组完成" % (model_tag, task), flush=True)


# ================= 调度器 (主进程) =================

def pending_eps(out, mt, task):
    missing = []
    for ep in range(REPEATS):
        rp = os.path.join(out, mt, task.replace("/", "_"), "ep%d" % ep, "result.json")
        if not os.path.exists(rp):
            missing.append(ep)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/apdcephfs/private_chengzhixu/JarvisVLA/outputs/rollout/eval_flat3")
    ap.add_argument("--models", default=None, help="逗号分隔 (默认 MODELS 全部)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stagger", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--one", nargs=4, default=None, metavar=("MODEL", "TASK", "OUT", "EPS"),
                    help="内部: 单组执行")
    args = ap.parse_args()

    if args.one:
        mt, task, out, eps_s = args.one
        run_one(mt, task, out, [int(x) for x in eps_s.split(",")], args.seed)
        return

    models = args.models.split(",") if args.models else list(MODELS)
    jobs = []
    for mt in models:
        for task in TASKS:
            eps = pending_eps(args.out, mt, task)
            if eps:
                jobs.append((mt, task, eps))
    print("待跑 %d 组: %s" % (len(jobs), [(j[0], j[1].split("/")[1], j[2]) for j in jobs]), flush=True)

    logs_dir = os.path.join(args.out, "_logs")
    os.makedirs(logs_dir, exist_ok=True)
    env = dict(os.environ, DISPLAY=":99")

    running, ji, last_launch = [], 0, 0.0
    while ji < len(jobs) or running:
        while (ji < len(jobs) and len(running) < args.workers
               and (not running or time.time() - last_launch >= args.stagger)):
            mt, task, eps = jobs[ji]
            tag = "%s_%s_eps%s" % (mt, task.split("/")[1], ",".join(map(str, eps)))
            logf = open(os.path.join(logs_dir, tag + ".log"), "a")
            p = subprocess.Popen(
                [PY, os.path.abspath(__file__), "--one", mt, task, args.out,
                 ",".join(map(str, eps))],
                stdout=logf, stderr=subprocess.STDOUT, env=env)
            running.append((p, jobs[ji], logf))
            print("启动组: %s (pid %d)" % (tag, p.pid), flush=True)
            ji += 1
            last_launch = time.time()
        for item in running[:]:
            p, job, logf = item
            if p.poll() is not None:
                running.remove(item)
                logf.close()
                print("组结束: %s_%s rc=%d (剩 %d)" % (
                    job[0], job[1].split("/")[1], p.returncode,
                    len(running) + len(jobs) - ji), flush=True)
        time.sleep(5)

    # 汇总
    print()
    print("=" * 60)
    summary = {}
    for task in TASKS:
        row = {}
        for mt in models:
            eps = []
            for ep in range(REPEATS):
                rp = os.path.join(args.out, mt, task.replace("/", "_"), "ep%d" % ep, "result.json")
                if os.path.exists(rp):
                    eps.append(json.load(open(rp)).get("ok", False))
            row[mt] = "%d/%d" % (sum(eps), len(eps))
        summary[task] = row
        print("%-28s %s" % (task, "  ".join("%s=%s" % kv for kv in row.items())))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("=" * 60)


if __name__ == "__main__":
    main()

import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""组评测: 一个 (model, task) 组一个进程, Minecraft 实例只起一次,
组内连续 fast-reset 跑全部局 (复用 JVM, 省去每局 1~2 分钟重启).

用法: python run_eval_task.py <model_tag> <task> <out_root> [eps]
  eps: 逗号分隔的局号 (默认 0,1,2); 已完成的局会被跳过由调度器决定.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/apdcephfs/private_chengzhixu/JarvisVLA")
os.chdir("/apdcephfs/private_chengzhixu/JarvisVLA/jarvisvla/evaluate")

from rich import console
from minestudio.simulator.callbacks import RecordCallback
from jarvisvla.evaluate import agent_wrapper
from duplex.minecraft_eval_protocol import (
    TASKS, DEFAULT_SEED, build_env, load_task_cfg, prepare_episode, finish_episode_recording,
    build_task_instruction,
)

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
    "s2v2": {
        "base_url": "http://localhost:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/jarvisvla_s2/v1-20260923-222449/checkpoint-2916-qwen2-vl"),
    },
    "s2v3_hist0": {
        "base_url": "http://localhost:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/s2v3_hist0/v0-20260924-201818/checkpoint-2900"),
    },
    "s2v3_hist1": {
        "base_url": "http://localhost:8200/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/s2v3_hist1/v0-20260925-055214/checkpoint-2900"),
    },
    "s2v3_hist2": {
        "base_url": "http://localhost:8300/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/s2v3_hist2/v0-20260929-011248/checkpoint-2900"),
    },
    # history=0 消融: 同 official 服务/模型, 仅输出目录与 EVAL_HIST_NUM 区分
    "official_h0": {
        "base_url": "http://localhost:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B"),
    },
    # stage2 flat3: 单轮三帧直排格式 (stage2_train_flat3.jsonl), 用 FLAT3_AGENT
    "s2v4_flat3": {
        "base_url": "http://localhost:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/s2v4_flat3/v0-20260930-225227/"
                 "v0-20260930-225519/checkpoint-2900"),
        "flat3": True,
    },
    # 同一ckpt的复测标签 (验证6/12可复现性, 输出目录与首轮隔离)
    "s2v4_flat3_r2": {
        "base_url": "http://localhost:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/s2v4_flat3/v0-20260930-225227/"
                 "v0-20260930-225519/checkpoint-2900"),
        "flat3": True,
    },
    # 重训的flat3 (2026-10-11新训, 同数据同配置不同随机) @8101
    "s2v4_flat3_new": {
        "base_url": "http://localhost:8101/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/s2v4_flat3_r2/v0-20261010-222856/"
                 "v0-20261010-223149/checkpoint-2900"),
        "flat3": True,
    },
    # 指数饱和camera加权flat3 (训练到step2000暂停) @8100 单实例
    "flat3_exp2000": {
        "base_url": "http://localhost:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/s2v4_flat3_camera_exponential/run-20261013-225717-y1zksG/"
                 "v0-20261013-230028/checkpoint-2000"),
        "flat3": True,
    },
    "flat3_exp_v3_1936": {
        "base_url": "http://127.0.0.1:8100/v1",
        "ckpt": "/apdcephfs/private_chengzhixu/eval_model_views/flat3_exp_v3_1936",
        "flat3": True,
    },
    "flat3_noop_unfiltered_ce_1936": {
        "base_url": "http://127.0.0.1:8100/v1",
        "ckpt": "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/duplex_weighted_then_noop_ce/eval_views/stage2_noop_unfiltered_ce_1936",
        "flat3": True,
    },
    # v5 duplex camera exponential (checkpoint-500) 非全双工 flat3 评测视图 @8100
    "flat3_v5_duplex_camera_exp_500": {
        "base_url": "http://127.0.0.1:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/v5_ce_camera_exponential/eval_views/duplex_camera_exp_500"),
        "flat3": True,
    },
    # v5 duplex 普通CE (checkpoint-500) 非全双工 flat3 评测视图 @8110
    "flat3_v5_duplex_ce_500": {
        "base_url": "http://127.0.0.1:8110/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/v5_ce_camera_exponential/eval_views/duplex_ce_500"),
        "flat3": True,
    },
    # v5 拆分消融: action_only 普通 CE (checkpoint-1162, 基座直训无duplex token) @8100
    "flat3_v5_action_only_ce_1162": {
        "base_url": "http://127.0.0.1:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/v5_split_action_qa/eval_views/action_only_ce_1162"),
        "flat3": True,
    },
    # v5 stage2 混训普通 CE 基座 (checkpoint-1925, action+QA混训对照) @8100
    "flat3_v5_stage2_ce_1925": {
        "base_url": "http://127.0.0.1:8100/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/v5_ce_camera_exponential/eval_views/stage2_ce_1925"),
        "flat3": True,
    },
    # Top5加权flat3 (checkpoint-2800, 用户提前停止) @8101
    "flat3_weighted": {
        "base_url": "http://localhost:8101/v1",
        "ckpt": ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/checkpoint/s2v4_flat3_top5weighted/run-20261012-224750-ClArDM/"
                 "v0-20261012-225125/checkpoint-2800"),
        "flat3": True,
    },
}


class FIXED_TASK_AGENT(agent_wrapper.VLLM_AGENT):
    def create_instruction(self, env_prompt, method):
        return build_task_instruction(env_prompt)


class FLAT3_AGENT(FIXED_TASK_AGENT):
    """s2v4_flat3 专用: 单轮三帧直排格式 (与训练数据 stage2_train_flat3.jsonl 逐字一致).

    prompt = 指令 + '\\nobservation: \\n'<img> + 'observation: \\n'<img> + 'observation: \\n'<img>
    帧窗口 = 最近2帧 + 当前帧 (开局不足用当前帧补); 无多轮结构, 无历史动作token.
    采样: 与官方协议一致 temp=0.9, top_p=0.99, top_k=-1, skip_special_tokens=False.
    """

    def forward(self, observations, instructions, verbos=False, need_crafting_table=False):
        if self.actions:  # action chunk 缓存 (action_chunk_len=1 时基本不触发)
            if len(self.actions) > 1:
                return self.actions.pop(0)
            action = self.actions[0]
            self.actions = []
            return action

        image = self.processor_wrapper.create_image_input(observations[0])
        if not hasattr(self, "_flat_instruction"):
            method = self.method_map[need_crafting_table]
            self._flat_instruction = self.create_instruction(instructions[0], method=method)
        private_instruction = self._flat_instruction

        frames = list(getattr(self, "_flat_hist", []))[-2:] + [image]
        while len(frames) < 3:
            frames.insert(0, image)
        # create_message_vllm 给每段text追加'\n':
        #   [instr+'\nobservation: ', 'observation: ', 'observation: ']
        # 渲染为 instr+'\nobservation: \n'<img>'observation: \n'<img>'observation: \n'<img> (==训练格式)
        messages = [self.processor_wrapper.create_message_vllm(
            role="user", input_type="image",
            prompt=[private_instruction + "\nobservation: ",
                    "observation: ", "observation: "],
            image=frames)]

        chat_completion = self.client.chat.completions.create(
            messages=messages, model=self.model,
            temperature=self.temperature, max_tokens=1024, top_p=0.99,
            extra_body={"skip_special_tokens": False, "top_k": -1})
        raw_outputs = chat_completion.choices[0].message.content
        self.log_inputs.append({"step": len(self.log_inputs), "messages": messages})
        self.log_outputs.append({"step": len(self.log_outputs), "output": raw_outputs})

        self._flat_hist = list(getattr(self, "_flat_hist", []))[-2:] + [image]

        outputs = self.tokenizer(raw_outputs)["input_ids"]
        actions = self.action_tokenizer.decode(outputs)
        len_action = min(self.action_chunk_len, len(actions))
        self.actions = actions[:len_action]
        return self.actions.pop(0)


def run_episode(env, record_callback, cfg, model, ep_dir, max_frames, itype, hist,
                episode=0, base_seed=DEFAULT_SEED):
    """跑一局: fast-reset + 环境准备 + rollout. 复用 env (JVM)."""
    obs, info, need_crafting_table = prepare_episode(
        env, record_callback, cfg, ep_dir, episode, base_seed)
    agent_cls = FLAT3_AGENT if model.get("flat3") else FIXED_TASK_AGENT
    agent_itype = 'normal'
    agent = agent_cls(
        checkpoint_path=model["ckpt"], base_url=model["base_url"],
        io_log_dir=None,
        temperature=0.9, history_num=hist,
        instruction_type=agent_itype, action_chunk_len=1)

    instructions = [item["text"] for item in cfg.task_conf]
    instruction = build_task_instruction(instructions[0], episode, base_seed, ep_dir)
    if model.get("flat3"):
        agent._flat_instruction = instruction
    success = (False, max_frames)
    try:
        for i in range(max_frames):
            action = agent.forward([info["pov"]], instructions, verbos=False,
                                   need_crafting_table=need_crafting_table)
            obs, reward, terminated, truncated, info = env.step(action)
            if reward > 0:
                success = (True, i)
                break
        if success[0]:  # 成功后再走 20 步 (对齐官方)
            for _ in range(20):
                action = agent.forward([info["pov"]], instructions, verbos=False,
                                       need_crafting_table=need_crafting_table)
                obs, reward, terminated, truncated, info = env.step(action)
    finally:
        # 完整落盘本局模型输入/输出 (两个json; 异常时也保存)
        try:
            json.dump(agent.log_inputs,
                      open(os.path.join(ep_dir, "inputs.json"), "w"))
            json.dump(agent.log_outputs,
                      open(os.path.join(ep_dir, "outputs.json"), "w"))
        except Exception:
            traceback.print_exc()

    # 立即保存本局视频并推进 episode_id (避免下一局 reset 时存错目录)
    finish_episode_recording(record_callback)
    return success


def main():
    model_tag, task, out_root = sys.argv[1:4]
    eps = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,1,2").split(",")]
    itype, hist, frames = TASKS[task]
    hist = int(os.environ.get("EVAL_HIST_NUM", hist))   # 消融: 可强制覆盖 history_num
    model = dict(MODELS[model_tag])
    model['base_url'] = os.environ.get('EVAL_BASE_URL', model['base_url'])
    print('TASK_SERVER %s %s' % (task, model['base_url']), flush=True)
    task_dir = os.path.join(out_root, model_tag, task.replace("/", "_"))

    base_seed = int(os.environ.get("EVAL_BASE_SEED", DEFAULT_SEED))
    cfg = load_task_cfg(task)
    env, callbacks = build_env(cfg, os.path.join(task_dir, "ep%d" % eps[0]), base_seed)
    record_callback = next(c for c in callbacks if isinstance(c, RecordCallback))

    try:
        for ep in eps:
            ep_dir = os.path.join(task_dir, "ep%d" % ep)
            os.makedirs(ep_dir, exist_ok=True)
            t0 = time.time()
            try:
                ok, steps = run_episode(env, record_callback, cfg, model, ep_dir, frames, itype, hist,
                                        episode=ep, base_seed=base_seed)
            except Exception as e:
                traceback.print_exc()
                json.dump({"ok": False, "steps": -1, "err": str(e)},
                          open(os.path.join(ep_dir, "result.json"), "w"))
                print("[%s %s ep%s] EXC %s" % (model_tag, task, ep, str(e)[:100]), flush=True)
                raise
            json.dump({"ok": bool(ok), "steps": steps},
                      open(os.path.join(ep_dir, "result.json"), "w"))
            print("[%s %s ep%s] success=%s steps=%s %.0fs" % (
                model_tag, task.split("/")[1], ep, ok, steps, time.time() - t0), flush=True)
    finally:
        env.close()
    print("[%s %s] 组完成" % (model_tag, task), flush=True)


if __name__ == "__main__":
    main()

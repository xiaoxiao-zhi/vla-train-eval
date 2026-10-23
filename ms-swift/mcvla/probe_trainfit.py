#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检验训练拟合: 50 条"当前动作 != 上一动作"的训练样本, 输入与训练完全一致
(样本自带的 messages + images, 含 swift 默认 system), 候选集内 greedy 解码,
对比模型输出 vs 训练标签.

  - 高匹配率     -> 训练逻辑正确, 模型能拟合训练数据 (probe 坍缩 = 泛化捷径)
  - 低匹配率     -> 训练/推理链路有 bug (loss 掩码错位 / 格式不一致)
  - pred==prev   -> 模型连训练样本上都抄历史 (拟合不足)

用法 (GPU 机器, 8 卡并行, 可续跑):
  for i in 0..7: CUDA_VISIBLE_DEVICES=$i python probe_trainfit.py --worker $i --nworkers 8
  python probe_trainfit.py --merge
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_conditioning import build_cand_ids, readable, toknums  # noqa: E402

CASES_FILE = "/apdcephfs/private_chengzhixu/probe_frames/trainfit_cases.json"
CKPT = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
        "chengzhixu/checkpoint/v2-20260920-214927/checkpoint-1225")
OUT_DIR = "/apdcephfs/private_chengzhixu/probe_frames/trainfit_out"
SYSTEM = "You are a helpful assistant."


def to_blocks(content):
    """swift 样本的 user content 是含字面 <image> 占位符的字符串;
    原生 chat template 不识别 -> 转成 text/image 内容块 (顺序保持)."""
    parts = content.split("<image>")
    blocks = []
    for i, p in enumerate(parts):
        if p:
            blocks.append({"type": "text", "text": p})
        if i < len(parts) - 1:
            blocks.append({"type": "image"})
    return blocks


def run_case(model, processor, case):
    import torch
    from PIL import Image
    tok = processor.tokenizer
    cand_ids = torch.tensor(build_cand_ids(tok), device="cuda:0")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    # 与训练完全一致: 样本自带 4 条 messages (title+图 / prev_action / 观测+图 / gt),
    # 去掉最后一条 assistant, 加 swift 默认 system
    msgs = [{"role": "system", "content": SYSTEM}]
    for m in case["messages"][:3]:
        c = to_blocks(m["content"]) if "<image>" in m["content"] else m["content"]
        msgs.append({"role": m["role"], "content": c})
    text = processor.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    pil = [Image.open(p).convert("RGB") for p in case["images"][:2]]
    inputs = processor(text=[text], images=pil, return_tensors="pt").to("cuda:0")
    ids = inputs.input_ids

    gen_ids = []
    for _ in range(16):
        with torch.no_grad():
            out = model(input_ids=ids, pixel_values=inputs.pixel_values,
                        image_grid_thw=inputs.image_grid_thw)
        sub = out.logits[0, -1].float()[cand_ids]
        nxt = int(cand_ids[int(sub.argmax())])
        if nxt == im_end_id:
            break
        gen_ids.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)

    pred_text = tok.decode(gen_ids, skip_special_tokens=False)
    gt_text = case["messages"][3]["content"]
    p_nums, g_nums = toknums(pred_text), toknums(gt_text)
    return {
        "pred_text": pred_text,
        "gt_text": gt_text,
        "pred_readable": readable(p_nums),
        "gt_readable": readable(g_nums),
        "match": p_nums == g_nums,
        "pred_eq_prev": p_nums == toknums(case["messages"][1]["content"]),
    }


def worker_main(args, cases):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelCls

    os.makedirs(args.out_dir, exist_ok=True)
    mine = [c for i, c in enumerate(cases) if i % args.nworkers == args.worker]
    todo = [(i, c) for i, c in enumerate(cases)
            if i % args.nworkers == args.worker
            and not os.path.exists(os.path.join(args.out_dir, "case_%03d.json" % i))]
    print("[worker %d] %d cases, %d todo" % (args.worker, len(mine), len(todo)), flush=True)
    if not todo:
        return
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = ModelCls.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    for k, (i, case) in enumerate(todo):
        res = run_case(model, processor, case)
        tmp = os.path.join(args.out_dir, ".tmp")
        with open(tmp, "w") as f:
            json.dump({"idx": i, **res}, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(args.out_dir, "case_%03d.json" % i))
        print("[worker %d] %d/%d case%03d %s | gt=%s pred=%s" % (
            args.worker, k + 1, len(todo), i,
            "MATCH" if res["match"] else "MISS ", res["gt_readable"][:60],
            res["pred_readable"][:60]), flush=True)


def merge_main(args, cases):
    n = len(cases)
    loaded = {}
    for i in range(n):
        p = os.path.join(args.out_dir, "case_%03d.json" % i)
        if os.path.exists(p):
            with open(p) as f:
                loaded[i] = json.load(f)
    if len(loaded) < n:
        print("缺 %d 个 case, 先跑 worker" % (n - len(loaded)))
        return
    match = sum(1 for r in loaded.values() if r["match"])
    eq_prev = sum(1 for r in loaded.values() if r["pred_eq_prev"])
    # 区分按钮变化/镜头变化子集
    import re
    def btns(t):
        return tuple(x for x in re.findall(r"_(\d+)\|", t) if not (209 <= int(x) <= 250))
    sub = {"btn_change": [0, 0], "cam_only": [0, 0]}
    for r in loaded.values():
        prev = r["gt_text"]  # 占位, 子集区分在挑样本时决定, 这里按 pred/gt 按钮是否同prev重算
    # 按钮维度单独统计: pred 按钮 == gt 按钮 (忽略镜头 bin)
    btn_match = sum(
        1 for r in loaded.values()
        if [t for t in r["pred_readable"].split("+") if not t.startswith(("yaw", "pitch"))] ==
           [t for t in r["gt_readable"].split("+") if not t.startswith(("yaw", "pitch"))])
    print("总样本: %d" % n)
    print("完全匹配 (含镜头bin): %d (%.0f%%)" % (match, 100 * match / n))
    print("按钮维度匹配 (忽略镜头): %d (%.0f%%)" % (btn_match, 100 * btn_match / n))
    print("pred==prev (抄历史): %d (%.0f%%)" % (eq_prev, 100 * eq_prev / n))
    print()
    print("== 未匹配样本明细 (前15) ==")
    k = 0
    for i in sorted(loaded):
        r = loaded[i]
        if not r["match"]:
            print("case%03d\n  gt  : %s\n  pred: %s\n  prev: (pred_eq_prev=%s)" % (
                i, r["gt_readable"], r["pred_readable"], r["pred_eq_prev"]))
            k += 1
            if k >= 15:
                break
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"n": n, "match": match, "btn_match": btn_match,
                   "pred_eq_prev": eq_prev, "details": loaded}, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--cases", default=CASES_FILE)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--nworkers", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    with open(args.cases) as f:
        cases = json.load(f)
    print("cases: %d" % len(cases))
    if args.merge:
        merge_main(args, cases)
    elif args.worker >= 0:
        worker_main(args, cases)


if __name__ == "__main__":
    main()

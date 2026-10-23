#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照实验: 定位 rollout 输出坍缩 (299 步全 forward+jump) 的原因.

特性: 8 卡并行 (--worker 0..7 分片) + 逐 case 落盘 (中断重跑自动跳过已完成).

  Test A: 固定 (指令, 上一帧, 上一动作), 只换 [当前帧]  -> 视觉条件化
  Test B: 固定 (帧, 上一动作), 只换 [指令]              -> 文本条件化
  Test C: 固定 (指令, 帧), 只换 [上一动作]              -> 历史自锁
  Test D: 无历史单轮 (t=0 格式), 只换 [当前帧]          -> 视觉敏感度(无历史)

用法 (GPU 机器, swift 环境):
  # 8 卡并行跑 (每卡一个 worker, 各处理 1/8 case):
  for i in 0..7: CUDA_VISIBLE_DEVICES=$i python probe_conditioning.py --worker $i --nworkers 8
  # 跑完合并汇总:
  python probe_conditioning.py --merge
  # 中断后重跑上面两条命令即可 (已完成 case 自动跳过)
"""
import argparse
import hashlib
import itertools
import json
import os

ROLLOUT_DIR = "/apdcephfs/private_chengzhixu/probe_frames/rollout"
EVAL_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
            "chengzhixu/dataset/mcvla_eval/frames")
TRAIN_FRAMES = [
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/Player0-f153ac423f61-20211202-205900-1043-1083_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/Player743-f153ac423f61-20220126-131023-2471-2513_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/cheeky-cornflower-setter-f153ac423f61-20220108-131304-1395-1452_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/gimpy-jade-panda-44ccd6301742-20220130-150752-3558-3599_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/gimpy-jade-panda-d8aacd4dbb48-20220306-140450-2506-2586_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/gimpy-jade-panda-f153ac423f61-20220225-161540-1532-1574_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/jumpy-denim-lion-7a1bdaf79a66-20220218-042228-2695-2745_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/pokey-cyan-spitz-1dcb485911e0-20220127-213349-538-688_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/shabby-pink-molly-1fc2a24ddec5-20220308-110300-2486-2547_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/shabby-pink-molly-f153ac423f61-20220213-121436-4653-4718_f000000.jpg",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames/shabby-viridian-beaver-f153ac423f61-20220116-002904-2504-2540_f000000.jpg",
]

ACT_LOCK = ("<|reserved_special_token_178|><|reserved_special_token_191|>"
            "<|reserved_special_token_206|><|reserved_special_token_219|>"
            "<|reserved_special_token_240|><|reserved_special_token_179|>")
ACT_ATTACK = ("<|reserved_special_token_178|><|reserved_special_token_204|>"
              "<|reserved_special_token_219|><|reserved_special_token_240|>"
              "<|reserved_special_token_179|>")
ACT_NOOP = ("<|reserved_special_token_178|><|reserved_special_token_219|>"
            "<|reserved_special_token_240|><|reserved_special_token_179|>")
ACT_TURN = ("<|reserved_special_token_178|><|reserved_special_token_216|>"
            "<|reserved_special_token_230|><|reserved_special_token_179|>")
ACT_BACK_LEFT = ("<|reserved_special_token_178|><|reserved_special_token_192|>"
                 "<|reserved_special_token_195|><|reserved_special_token_219|>"
                 "<|reserved_special_token_240|><|reserved_special_token_179|>")

TITLE_KILL = "Kill the zombie."
TITLES_B = [
    TITLE_KILL,
    "Chop down the oak tree to collect the oak logs.",
    "Mine the stone with your pickaxe.",
    "Craft a crafting table with the oak planks.",
    "Attack the cow.",
    "Swim across the river.",
]

TOKEN_MEANING = {
    178: "BEGIN", 179: "END",
    180: "hotbar0", 181: "hotbar.1", 182: "hotbar.2", 183: "hotbar.3", 184: "hotbar.4",
    185: "hotbar.5", 186: "hotbar.6", 187: "hotbar.7", 188: "hotbar.8", 189: "hotbar.9",
    190: "fb0", 191: "forward", 192: "back",
    193: "lr0", 194: "left", 195: "right",
    196: "ss0", 197: "sprint", 198: "sneak",
    199: "use0", 200: "use", 201: "drop0", 202: "drop",
    203: "atk0", 204: "attack", 205: "jmp0", 206: "jump",
    207: "camflag0", 208: "camflag1",
    209: "inv0", 210: "inv1",
}


def toknums(text):
    import re
    return [int(m) for m in re.findall(r"reserved_special_token_(\d+)", text)]


def readable(nums):
    out = []
    for n in nums:
        if 209 <= n <= 229 and n != 219:
            out.append("yawbin%d" % (n - 209))
        elif 230 <= n <= 250 and n != 240:
            out.append("pitchbin%d" % (n - 230))
        else:
            out.append(TOKEN_MEANING.get(n, str(n)))
    return "+".join(out)


def list_frames():
    rollout_frames = sorted(
        os.path.join(ROLLOUT_DIR, f) for f in os.listdir(ROLLOUT_DIR) if f.endswith(".jpg"))
    eval_frames = sorted(
        os.path.join(EVAL_DIR, f) for f in os.listdir(EVAL_DIR) if f.endswith(".jpg"))
    seen, eval_pick = set(), []
    for f in eval_frames:
        key = f.rsplit("_f0", 1)[0]
        if key not in seen:
            seen.add(key)
            eval_pick.append(f)
    train_frames = [f for f in TRAIN_FRAMES if os.path.exists(f)]
    return rollout_frames, eval_pick, train_frames


def build_cases():
    """构造全部 case (顺序确定, 各 worker 按 idx%nworkers 分片)."""
    rollout_frames, eval_pick, train_frames = list_frames()
    prev_f = rollout_frames[0]
    cur_f_mid = rollout_frames[len(rollout_frames) // 2]

    cases = []
    for f in rollout_frames:
        cases.append({"tag": "A_rollout_frames", "desc": os.path.basename(f),
                      "title": TITLE_KILL, "prev_frame": prev_f,
                      "prev_action": ACT_LOCK, "cur_frame": f})
    for f in eval_pick:
        cases.append({"tag": "A_eval_frames", "desc": os.path.basename(f),
                      "title": TITLE_KILL, "prev_frame": prev_f,
                      "prev_action": ACT_LOCK, "cur_frame": f})
    for f in train_frames:
        cases.append({"tag": "A_train_frames", "desc": os.path.basename(f),
                      "title": TITLE_KILL, "prev_frame": prev_f,
                      "prev_action": ACT_LOCK, "cur_frame": f})
    for t in TITLES_B:
        cases.append({"tag": "B_titles", "desc": t, "title": t,
                      "prev_frame": prev_f, "prev_action": ACT_LOCK,
                      "cur_frame": cur_f_mid})
    for name, act in [("lock(fwd+jump)", ACT_LOCK), ("attack", ACT_ATTACK),
                      ("noop", ACT_NOOP), ("turn", ACT_TURN),
                      ("back+left", ACT_BACK_LEFT)]:
        cases.append({"tag": "C_prev_actions", "desc": name, "title": TITLE_KILL,
                      "prev_frame": prev_f, "prev_action": act,
                      "cur_frame": cur_f_mid})
    for f in rollout_frames + eval_pick[:5]:
        cases.append({"tag": "D_single_turn", "desc": os.path.basename(f),
                      "title": TITLE_KILL, "prev_frame": None,
                      "prev_action": None, "cur_frame": f})
    # 稳定 case id
    for c in cases:
        h = hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:10]
        c["id"] = "%s__%s" % (c["tag"], h)
    return cases


def build_messages(title, prev_frame, prev_action, cur_frame):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    if prev_frame:
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": "%s\nobservation: \n" % title},
            {"type": "image"}]})
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": prev_action}]})
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": "observation: \n"},
            {"type": "image"}]})
        images = [prev_frame, cur_frame]
    else:
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": "%s\nobservation: \n" % title},
            {"type": "image"}]})
        images = [cur_frame]
    return msgs, images


def build_cand_ids(tok):
    """与 action_constrained_loss.py 完全一致的候选集: 保留字176-250 + \\n + <|im_end|>."""
    cand = []
    for n in range(176, 251):
        tid = tok.convert_tokens_to_ids("<|reserved_special_token_%d|>" % n)
        if tid is not None and tid >= 0:
            cand.append(tid)
    for t in ("\n", "<|im_end|>"):
        cand += tok.encode(t, add_special_tokens=False)
    return sorted(set(cand))


def run_case(model, processor, case):
    """候选集内逐步 greedy 解码 (对齐 action_constrained 训练目标).

    训练时动作序列所有位置只在候选子集内 softmax, 候选外 logits 未被压制,
    自由全词表 generate 会掉进未训练区域 -> 必须受限解码.
    """
    import torch
    from PIL import Image
    tok = processor.tokenizer
    cand_ids = torch.tensor(build_cand_ids(tok), device="cuda:0")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    msgs, images = build_messages(
        case["title"], case["prev_frame"], case["prev_action"], case["cur_frame"])
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pil = [Image.open(p).convert("RGB") for p in images]
    inputs = processor(text=[text], images=pil, return_tensors="pt").to("cuda:0")
    ids = inputs.input_ids

    gen_ids, dists = [], []
    for _ in range(16):
        with torch.no_grad():
            out = model(input_ids=ids, pixel_values=inputs.pixel_values,
                        image_grid_thw=inputs.image_grid_thw)
        logits = out.logits[0, -1].float()
        sub = logits[cand_ids]
        p = torch.softmax(sub, -1)
        dists.append({int(i): round(float(v), 6) for i, v in zip(cand_ids.tolist(), p.tolist())})
        nxt = int(cand_ids[int(p.argmax())])
        if nxt == im_end_id:
            break
        gen_ids.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)

    text_out = tok.decode(gen_ids, skip_special_tokens=False)
    nums = toknums(text_out)
    return {
        "pred_text": text_out,
        "tokens": nums,
        "readable": readable(nums),
        "cand_dist_first": dists[0],
        "n_gen": len(gen_ids),
    }


def worker_main(args, cases):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelCls

    os.makedirs(args.cases_dir, exist_ok=True)
    mine = [c for i, c in enumerate(cases) if i % args.nworkers == args.worker]
    todo = [c for c in mine
            if not os.path.exists(os.path.join(args.cases_dir, c["id"] + ".json"))]
    print("[worker %d] %d cases, %d todo" % (args.worker, len(mine), len(todo)), flush=True)
    if not todo:
        return

    print("[worker %d] loading model ..." % args.worker, flush=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = ModelCls.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    for k, case in enumerate(todo):
        out = run_case(model, processor, case)
        tmp = os.path.join(args.cases_dir, ".%s.tmp" % case["id"])
        with open(tmp, "w") as f:
            json.dump({**{kk: case[kk] for kk in
                          ("id", "tag", "desc", "title", "cur_frame")},
                       **out}, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(args.cases_dir, case["id"] + ".json"))
        print("[worker %d] %d/%d %s -> %s" % (
            args.worker, k + 1, len(todo), case["id"][:40], out["readable"]), flush=True)


def merge_main(args, cases):
    import math
    details = {}
    for c in cases:
        p = os.path.join(args.cases_dir, c["id"] + ".json")
        if os.path.exists(p):
            with open(p) as f:
                details[c["id"]] = json.load(f)
    missing = [c["id"] for c in cases if c["id"] not in details]
    if missing:
        print("缺 %d 个 case, 未合并: %s ..." % (len(missing), missing[:3]))
        return
    # 精确候选集分布 (受限解码时保存的 cand_dist_first)
    def dist_of(cid):
        return details[cid]["cand_dist_first"]

    def kl(p, q):
        keys = set(p) | set(q)
        return sum(p.get(k, 0) * math.log(
            (p.get(k, 1e-12) + 1e-12) / (q.get(k, 1e-12) + 1e-12)) for k in keys)

    tags = []
    for c in cases:
        if c["tag"] not in tags:
            tags.append(c["tag"])
    summary = {}
    for tag in tags:
        items = [c for c in cases if c["tag"] == tag]
        outs = [details[c["id"]]["readable"] for c in items]
        kls = []
        for a, b in itertools.combinations(items, 2):
            kls.append(kl(dist_of(a["id"]), dist_of(b["id"])))
        summary[tag] = {
            "n": len(items),
            "distinct_outputs": len(set(outs)),
            "kl_mean": round(sum(kls) / len(kls), 4) if kls else 0.0,
            "kl_max": round(max(kls), 4) if kls else 0.0,
            "outputs": {o: outs.count(o) for o in sorted(set(outs))},
        }
    result = {"summary": summary,
              "details": {c["id"]: details[c["id"]] for c in cases}}
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\nsaved -> %s" % args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=("/apdcephfs/share_chengzhixu/cephfs/"
                                              "apdcephfs_tj5/share_303556863/chengzhixu/"
                                              "checkpoint/v2-20260920-214927/checkpoint-1225"))
    ap.add_argument("--worker", type=int, default=-1, help=">=0 跑该分片, -1 仅列 case")
    ap.add_argument("--nworkers", type=int, default=8)
    ap.add_argument("--merge", action="store_true", help="合并 case 文件出报告")
    ap.add_argument("--cases-dir", default="/apdcephfs/private_chengzhixu/probe_frames/cases")
    ap.add_argument("--out", default="/apdcephfs/private_chengzhixu/probe_frames/probe_results.json")
    args = ap.parse_args()

    cases = build_cases()
    print("total cases: %d" % len(cases))
    if args.merge:
        merge_main(args, cases)
    elif args.worker >= 0:
        worker_main(args, cases)
    else:
        for i in range(args.nworkers):
            n = sum(1 for j, _ in enumerate(cases) if j % args.nworkers == i)
            print("  worker %d: %d cases" % (i, n))


if __name__ == "__main__":
    main()

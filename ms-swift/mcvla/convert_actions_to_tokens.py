#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 minecraft_action/*.json 里的 VPT 原始动作转成 JarvisVLA 保留字.

转换链路 (与官方一致, 独立实现, 仅依赖标准库):
  1. VPT jsonl 行 (mouse/keyboard) -> MineRL env 动作
     [VPT 官方 run_inverse_dynamics_model.py : json_action_to_env_action]
  2. env 动作 -> 20维按钮 + camera mu-law 离散 bin (bins=21, 中点10)
     [VPT 官方 lib/actions.py : ActionTransformer.env2policy / CameraQuantizer]
  3. -> 12 维动作组 (组内互斥, 前后/左右同按算无, inventory 帧清空其他动作)
     [VPT 官方 lib/action_mapping.py : CameraHierarchicalMapping.from_factored]
  4. 12 维组 -> 保留字序列 (组0-7 非零才发射; camera 两 bin 恒发射; 首尾 178/179)
     [JarvisVLA jarvisvla/inference/action_mapping.py : group_action_2_token]

  与 JarvisVLA 官方一致的细节:
  - 组8(camera标志)/组9(inventory) 不发射 token (官方 group_action_2_token 只遍历组0-7 和组10/11)
  - inventory=1 的帧: 其他按钮清零, camera 强制中点 bin (10,10)
  - assistant 文本末尾带 "\n", 与 SFT 训练数据格式一致

用法:
  python convert_actions_to_tokens.py                    # 转换全部 json (原地更新)
  python convert_actions_to_tokens.py --limit 100        # 只处理前 100 个
  python convert_actions_to_tokens.py --verify           # 用 SFT 数据集验证正确性
"""
import argparse
import glob
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# ============ 1. VPT jsonl -> MineRL env 动作 ============
# [VPT 官方 run_inverse_dynamics_model.py]
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}
CAMERA_SCALER = 360.0 / 2400.0

# [VPT 官方 lib/actions.py : Buttons.ALL]
BUTTONS_ALL = [
    "attack", "back", "forward", "jump", "left", "right",
    "sprint", "sneak", "use", "drop", "inventory",
] + ["hotbar.%d" % i for i in range(1, 10)]


def json_action_to_env_action(json_action):
    """VPT jsonl 一行 -> (env_action dict, is_null_action)."""
    env_action = {name: 0 for name in BUTTONS_ALL}
    env_action["camera"] = [0.0, 0.0]  # [pitch, yaw]

    is_null = True
    for key in json_action.get("keyboard", {}).get("keys", []):
        if key in KEYBOARD_BUTTON_MAPPING:
            name = KEYBOARD_BUTTON_MAPPING[key]
            if name in env_action:
                env_action[name] = 1
                is_null = False

    mouse = json_action.get("mouse", {})
    dx = mouse.get("dx", 0.0) or 0.0
    dy = mouse.get("dy", 0.0) or 0.0
    env_action["camera"][0] = dy * CAMERA_SCALER  # pitch
    env_action["camera"][1] = dx * CAMERA_SCALER  # yaw
    if dx != 0 or dy != 0:
        is_null = False

    for btn in mouse.get("buttons", []) or []:
        if btn == 0:
            env_action["attack"] = 1
            is_null = False
        elif btn == 1:
            env_action["use"] = 1
            is_null = False
        elif btn == 2:
            env_action["pickItem"] = 1
            is_null = False
    return env_action, is_null


# ============ 2. camera mu-law 量化 ============
# [VPT 官方 lib/actions.py : CameraQuantizer, 参数与 JarvisVLA 默认一致]
CAMERA_MAXVAL = 10.0
CAMERA_BINSIZE = 1.0
CAMERA_MU = 20.0


def discretize_camera(v):
    """连续角度 -> 0..20 的 bin (10=中点不动)."""
    v = max(-CAMERA_MAXVAL, min(CAMERA_MAXVAL, v))
    v = v / CAMERA_MAXVAL
    v = math.copysign(
        math.log(1.0 + CAMERA_MU * abs(v)) / math.log(1.0 + CAMERA_MU), v
    ) * CAMERA_MAXVAL
    return int(round((v + CAMERA_MAXVAL) / CAMERA_BINSIZE))


# ============ 3. 按钮向量 -> 12 维动作组 ============
# [VPT 官方 lib/action_mapping.py : factored_buttons_to_groups / from_factored]
# 组: 0 hotbar | 1 前后 | 2 左右 | 3 sprint/sneak | 4 use | 5 drop
#     6 attack | 7 jump | 8 camera标志 | 9 inventory | 10 camera_x | 11 camera_y
CAMERA_NULL_BIN = 10


def env_action_to_group(env_action):
    buttons = env_action
    pitch_bin = discretize_camera(env_action["camera"][0])
    yaw_bin = discretize_camera(env_action["camera"][1])

    # inventory=1: 清空其他按钮, camera 强制中点 (官方 from_factored 语义)
    if buttons.get("inventory"):
        g = [0] * 12
        g[9] = 1
        g[10] = CAMERA_NULL_BIN
        g[11] = CAMERA_NULL_BIN
        return g

    g = [0] * 12
    # 组0 hotbar: 多个按下时后者优先 (官方遍历顺序)
    for i in range(1, 10):
        if buttons.get("hotbar.%d" % i):
            g[0] = i
    # 组1 前后: 同按算无
    f, b = buttons.get("forward", 0), buttons.get("back", 0)
    g[1] = 1 if (f and not b) else (2 if (b and not f) else 0)
    # 组2 左右: 同按算无
    l, r = buttons.get("left", 0), buttons.get("right", 0)
    g[2] = 1 if (l and not r) else (2 if (r and not l) else 0)
    # 组3 sprint/sneak: 后者(sneak)优先
    if buttons.get("sprint"):
        g[3] = 1
    if buttons.get("sneak"):
        g[3] = 2
    g[4] = int(bool(buttons.get("use", 0)))
    g[5] = int(bool(buttons.get("drop", 0)))
    g[6] = int(bool(buttons.get("attack", 0)))
    g[7] = int(bool(buttons.get("jump", 0)))
    # 组8 camera 标志 (不发射, 解码端由 camera!=null 恢复)
    g[8] = 0 if (pitch_bin == CAMERA_NULL_BIN and yaw_bin == CAMERA_NULL_BIN) else 1
    g[10] = pitch_bin
    g[11] = yaw_bin
    return g


# ============ 4. 12 维组 -> 保留字 ============
# [JarvisVLA jarvisvla/inference/action_mapping.py : map_control_token / group_action_2_token]
GROUP_TOKEN_START = {
    0: 180, 1: 190, 2: 193, 3: 196, 4: 199,
    5: 201, 6: 203, 7: 205, 8: 207, 9: 176, 10: 209, 11: 230,
}
ACT_BEG, ACT_END = 178, 179


def tok(n):
    return "<|reserved_special_token_%d|>" % n


def group_to_token_text(group):
    """12 维组 -> 保留字序列文本 (与 SFT 训练数据 assistant 格式一致)."""
    parts = []
    # 按钮 组0-7: 非零才发射 (官方 group_action_2_token 跳零规则)
    for place in range(8):
        if group[place] != 0:
            parts.append(tok(GROUP_TOKEN_START[place] + group[place]))
    # camera 两 bin 恒发射
    parts.append(tok(GROUP_TOKEN_START[10] + group[10]))
    parts.append(tok(GROUP_TOKEN_START[11] + group[11]))
    return tok(ACT_BEG) + "".join(parts) + tok(ACT_END) + "\n"


def action_to_token_text(json_action):
    """VPT jsonl 一行 -> (保留字文本, 12维组, is_null)."""
    env_action, is_null = json_action_to_env_action(json_action)
    g = env_action_to_group(env_action)
    return group_to_token_text(g), g, is_null


# ============ 批量转换 ============
DEFAULT_DATA_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/"
                    "share_303556863/chengzhixu/dataset/minecraft_action")


def convert_file(path, skip_done=True):
    try:
        with open(path) as f:
            d = json.load(f)
    except (ValueError, OSError):
        return "incomplete"          # make_clips 正在写/损坏, 跳过
    if d.get("actions") is None:
        return "no_actions"
    if skip_done and d.get("action_tokens_text"):
        return "skip"

    texts, groups, nulls = [], [], []
    for act in d["actions"]:
        t, g, n = action_to_token_text(act)
        texts.append(t)
        groups.append(g)
        nulls.append(n)
    d["action_tokens_text"] = texts
    d["action_groups"] = groups
    d["is_null_action"] = nulls

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, path)
    return "ok"


def main_convert(args):
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if args.limit:
        files = files[:args.limit]
    print("json files: %d" % len(files))
    from collections import Counter
    stats = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for st in ex.map(lambda p: convert_file(p, not args.redo), files):
            stats[st] += 1
    print("result:", dict(stats))


# ============ 验证: 与 SFT 数据集逐帧比对 ============
SFT_DATA_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/"
                "share_303556863/chengzhixu/dataset/minecraft-vla-sft/data")
VPT_VIDEO_DIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/"
                 "share_303556863/chengzhixu/dataset/vpt/7.x")


def main_verify(args):
    import re
    import pyarrow.parquet as pq
    id_pat = re.compile(r"^(.*)-(\d+)-(\d+)_(\d+)$")

    # 读 SFT 样本 (id + assistant 文本)
    sft = []
    pf = pq.ParquetFile(os.path.join(SFT_DATA_DIR, "train-00000-of-00216.parquet"))
    for batch in pf.iter_batches(columns=["id", "conversations"], batch_size=2048):
        for row in batch.to_pylist():
            m = id_pat.match(row["id"])
            if not m:
                continue
            text = "".join(item.get("text", "") for conv in row["conversations"]
                           if conv["role"] == "assistant"
                           for item in conv["content"] if item.get("type") == "text")
            sft.append((m.group(1), int(m.group(4)), text))
    # 按 clip 分组, 每个 clip 只打开一次源 jsonl
    by_clip = {}
    for clip, frame, text in sft:
        by_clip.setdefault(clip, []).append((frame, text))

    total = matched = no_video = mismatched = 0
    for clip, items in sorted(by_clip.items()):
        src = os.path.join(VPT_VIDEO_DIR, clip + ".jsonl")
        if not os.path.exists(src):
            no_video += len(items)
            continue
        need = {f: t for f, t in items}
        with open(src) as f:
            for i, line in enumerate(f):
                if i not in need:
                    continue
                total += 1
                text, _, _ = action_to_token_text(json.loads(line))
                if text == need[i]:
                    matched += 1
                else:
                    mismatched += 1
                    if mismatched <= 3:
                        print("MISMATCH frame %d\n  sft: %r\n  cvt: %r" % (
                            i, need[i], text))
    print("verified=%d matched=%d mismatched=%d no_video=%d" % (
        total, matched, mismatched, no_video))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--redo", action="store_true", help="已有结果也重算")
    ap.add_argument("--verify", action="store_true",
                    help="不用来转换, 而是与 SFT 数据集逐帧比对验证正确性")
    args = ap.parse_args()
    if args.verify:
        main_verify(args)
    else:
        main_convert(args)


if __name__ == "__main__":
    main()

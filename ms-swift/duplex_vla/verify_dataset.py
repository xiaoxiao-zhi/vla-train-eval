#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""duplex_dataset 验证: 流式读样本, 校验序列结构与 labels mask."""
import sys

sys.path.insert(0, "/apdcephfs/private_chengzhixu/ms-swift/duplex_vla")
from duplex_dataset import DuplexStreamingDataset

JSONL = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
         "chengzhixu/dataset/duplex_vla/duplex_turns_train.jsonl")
FRAMES = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
          "chengzhixu/dataset/mcvla_sft/frames")
TOKDIR = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
          "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B")

ds = DuplexStreamingDataset(JSONL, FRAMES, TOKDIR, shuffle_buffer=64, seed=0,
                            max_samples=500)   # 只扫前500行, 秒级验证
it = iter(ds)
n_qa = 0
for want in range(2):
    while True:
        d = next(it)
        has_txt = d["txt_span"][1] - d["txt_span"][0] > 2  # [ctrl, 1个id] 之外
        if want == 0 and not has_txt:
            break
        if want == 1 and has_txt:
            break
    ids, labels = d["input_ids"], d["labels"]
    n_lab = sum(1 for x in labels if x != -100)
    a0, a1 = d["act_span"]; t0, t1 = d["txt_span"]
    print("===== 样本%s (含QA目标=%s) =====" % (want + 1, has_txt))
    print(" 序列长度: %d, labels有效: %d (act %d + txt %d + im_end 1)"
          % (len(ids), n_lab, a1 - a0, t1 - t0))
    print(" 图像: %d 帧, pixel_values %s, grid_thw %s"
          % (d["n_images"], tuple(d["pixel_values"].shape),
             d["image_grid_thw"].tolist()))
    # labels mask 校验: 目标段外全 -100, 目标段内无 -100
    assert all(x == -100 for x in labels[:a0]), "目标段外有泄漏"
    assert all(x != -100 for x in labels[a0:]), "目标段内有漏标"
    # image_pad 数与 grid 一致
    n_pad = sum(1 for x in ids if x == ds._renderer.img_pad)
    n_grid = sum(t * h * w // 4 for t, h, w in d["image_grid_thw"].tolist())
    assert n_pad == n_grid, "image_pad 数与 grid 不一致"
    print(" image_pad=%d 与 grid 一致 ✓  labels mask ✓")
    # 可读摘要
    tok = ds._renderer.tok
    seg = tok.decode(ids[a0:t1], skip_special_tokens=False)
    print(" 目标段可读:", seg.replace("<|reserved_special_token_", "R").replace("|>", "")
          .replace("<|duplex_act|>", "[ACT]").replace("<|duplex_txt|>", "[TXT]")[:90])
print()
print("验证通过: 流式渲染 + labels mask + 图像展开全部正确")

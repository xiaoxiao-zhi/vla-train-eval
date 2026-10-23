#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工 VLA 训练数据集 (B 方案: id 级渲染, streaming).

消费 dataset/duplex_vla/duplex_turns_train.jsonl, 按 TRAIN_DATA_CONTRACT.md 渲染:

  序列 = chat外壳(system + 单user turn[历史4通道 + 当前轮] + assistant turn[目标y])
  图像 = images 帧过 Qwen2VL image processor -> pixel_values/grid_thw,
         每帧展开 N_i = prod(grid_thw_i)/4 个 image_pad 直接以 id 拼入序列
  labels = 仅 y 段 ([a]A_t[x]X_t[im_end]) 为真 id, 其余 -100
  meta   = act_span / txt_span (y 段内的全局位置区间, 供后续自定义 loss 使用)

接口 (与 collator/trainer 解耦):
  DuplexStreamingDataset(jsonl_path, frames_root, tokenizer_dir, ...)
      -> dict(input_ids, labels, pixel_values, image_grid_thw,
              act_span, txt_span, n_images)
  也提供 DuplexMapDataset (map 式, 小规模调试用)

严格约束: 一切内容拼接在 id 空间完成, 唯一经过 tokenizer 文本编码的是
          固定 chat 外壳 (完整词, 往返安全); txt/action 切片 id 原样直拼.
"""
import json
import os
import random

import torch
from torch.utils.data import Dataset, IterableDataset

CONTRACT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENS_PATH = os.path.join(CONTRACT_DIR, "duplex_tokens.json")


class DuplexRenderer:
    """jsonl 样本 -> 训练样本 (无状态, 可被 Dataset 复用)."""

    def __init__(self, frames_root, tokenizer_dir):
        from transformers import AutoTokenizer, AutoImageProcessor
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.img_proc = AutoImageProcessor.from_pretrained(
            tokenizer_dir, min_pixels=3136, max_pixels=12845056)
        self.frames_root = frames_root

        # 控制 token id: 唯一事实源 duplex_tokens.json
        ids = json.load(open(TOKENS_PATH))["ids"]
        self.CTRL = {k: v for k, v in ids.items()}
        self.SIL = ids["<|silence|>"]
        # 特殊 token id
        self.im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.img_pad = self.tok.convert_tokens_to_ids("<|image_pad|>")
        # chat 外壳固定文本 (完整词, 编码安全)
        self.pfx_sys = self.tok(
            "You are a helpful assistant.", add_special_tokens=False)["input_ids"]
        self.pfx_user = self.tok("user\n", add_special_tokens=False)["input_ids"]
        self.pfx_ast = self.tok("assistant\n", add_special_tokens=False)["input_ids"]
        self.nl = self.tok("\n", add_special_tokens=False)["input_ids"]

    # ---- 图像: 帧文件 -> pixel_values/grid_thw + 每帧展开数 ----
    def _process_images(self, frames):
        from PIL import Image
        pil = [Image.open(os.path.join(self.frames_root, f)).convert("RGB")
               for f in frames]
        out = self.img_proc(images=pil, return_tensors="pt")
        pv, thw = out["pixel_values"], out["image_grid_thw"]
        # 每帧 image token 数: t*h*w / merge_size^2 (Qwen2-VL merge=2)
        n_per = [(int(t) * int(h) * int(w)) // 4 for t, h, w in thw.tolist()]
        return pv, thw, n_per

    def render(self, s):
        """一条 jsonl 样本 -> 训练样本 dict."""
        frames = s["images"]
        pv, thw, n_per = self._process_images(frames)
        f_idx = 0  # 帧游标 (按序列出现顺序消耗)

        C = self.CTRL
        ids, labels, mm_types = [], [], []

        def emit(seq, is_target=False, is_image=False):
            ids.extend(seq)
            labels.extend(seq if is_target else [-100] * len(seq))
            mm_types.extend([1] * len(seq) if is_image else [0] * len(seq))

        # ---- chat 外壳: system ----
        emit([self.im_start])
        emit(self.tok("system\n", add_special_tokens=False)["input_ids"])
        emit(self.pfx_sys)
        emit([self.im_end] + self.nl)
        # ---- user turn: 历史4通道 + 当前轮 ----
        emit([self.im_start] + self.pfx_user)
        for c in s["context"]:
            emit([C["<|duplex_user|>"]] + c["user_ids"])
            if c["frame"] is not None:
                emit([C["<|duplex_obs|>"]])
                emit([self.img_pad] * n_per[f_idx], is_image=True)
                emit([C["<|duplex_act|>"]] + c["action_ids"])
                f_idx += 1
            emit([C["<|duplex_txt|>"]] + c["txt_ids"])
        emit([C["<|duplex_user|>"]] + s["current"]["user_ids"]
             + [C["<|duplex_obs|>"]])
        emit([self.img_pad] * n_per[f_idx], is_image=True)
        f_idx += 1
        emit([self.im_end] + self.nl)
        # ---- assistant turn: 目标 y (唯一进 labels 的段) ----
        emit([self.im_start] + self.pfx_ast)
        mark_act0 = len(ids)
        emit([C["<|duplex_act|>"]] + s["target"]["action_ids"], is_target=True)
        act_span = (mark_act0, len(ids))
        mark_txt0 = len(ids)
        emit([C["<|duplex_txt|>"]] + s["target"]["txt_ids"], is_target=True)
        txt_span = (mark_txt0, len(ids))
        emit([self.im_end], is_target=True)

        assert f_idx == len(frames), "帧消耗数不一致"
        return {
            "input_ids": ids,
            "labels": labels,
            "mm_token_type_ids": mm_types,
            "pixel_values": pv,
            "image_grid_thw": thw,
            "act_span": act_span,
            "txt_span": txt_span,
            "n_images": len(frames),
        }


class DuplexIndexedDataset(Dataset):
    """索引式 (map 式): 行号 -> 惰性渲染. 避开 IterableDataset+多进程的 dispatch 限制.

    全部行读入内存 (~400MB 字符串), __getitem__ 时才 json 解析 + 渲染;
    Trainer 走标准 map 路径: 自动 shuffle / 分布式分卡 / num_workers 均原生支持.
    """

    def __init__(self, jsonl_path, frames_root, tokenizer_dir):
        self.lines = open(jsonl_path).readlines()
        self._renderer = DuplexRenderer(frames_root, tokenizer_dir)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return self._renderer.render(json.loads(self.lines[i]))


class DuplexStreamingDataset(IterableDataset):
    """流式: 逐行读 jsonl + buffer shuffle, 内存占用 = buffer 大小.

    用法 (torch DataLoader):
        ds = DuplexStreamingDataset(jsonl, frames_root, tokenizer_dir,
                                    shuffle_buffer=4096, seed=0)
        DataLoader(ds, batch_size=..., collate_fn=duplex_collate, num_workers=8)
    """

    def __init__(self, jsonl_path, frames_root, tokenizer_dir,
                 shuffle_buffer=4096, seed=0, max_samples=0):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.max_samples = max_samples
        self._renderer = DuplexRenderer(frames_root, tokenizer_dir)

    def __iter__(self):
        # DataLoader 多 worker 分片: 各 worker 只处理 line_idx % num_workers == worker_id
        # 的行, 避免重复消费 (shuffle 退化为分片内局部 shuffle, 可接受)
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed + wid)
        buf = []
        n = 0
        with open(self.jsonl_path) as f:
            for line_no, line in enumerate(f):
                if line_no % nw != wid:
                    continue
                if self.max_samples and n >= self.max_samples:
                    break
                s = json.loads(line)
                item = self._renderer.render(s)
                n += 1
                if len(buf) < self.shuffle_buffer:
                    buf.append(item)
                    continue
                i = rng.randrange(len(buf))
                yield buf[i]
                buf[i] = item
        rng.shuffle(buf)
        yield from buf


class DuplexMapDataset(Dataset):
    """map 式 (随机访问, 小规模调试/校验用; 全量 9.3 万条约需较多内存, 训练用 streaming 版)."""

    def __init__(self, jsonl_path, frames_root, tokenizer_dir, max_samples=0):
        self.items_meta = []
        with open(jsonl_path) as f:
            for line in f:
                self.items_meta.append(json.loads(line))
                if max_samples and len(self.items_meta) >= max_samples:
                    break
        self._renderer = DuplexRenderer(frames_root, tokenizer_dir)

    def __len__(self):
        return len(self.items_meta)

    def __getitem__(self, i):
        return self._renderer.render(self.items_meta[i])

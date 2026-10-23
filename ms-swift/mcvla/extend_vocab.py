#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第一步: 按 JarvisVLA 的逻辑给 Qwen2.5-VL-7B-Instruct 扩展动作保留字词表.

JarvisVLA 官方逻辑 (jarvisvla/train/train.py:89-102 + jarvisvla/__init__.py):
    QWEN_SPECIAL_TOKENS = <repo>/assets/special_token.json
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": special_token})   # 251 个特殊 token
    (训练时 Trainer 自动 resize embeddings)

本脚本做完全相同的事, 并离线保存成一个新模型目录:
    1. 读 JarvisVLA/assets/special_token.json
       (含 <|reserved_special_token_18~250|>, 其中 176~250 为动作保留字,
        178/179 为动作起止标签, 与官方 SFT 数据格式一致)
    2. tokenizer.add_special_tokens (Qwen2.5-VL 词表中已存在的自动跳过,
       新 token 的 id 从当前词表长度顺延分配)
    3. model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    4. 保存 model / tokenizer / processor 到输出目录

用法 (在 conda_envs/swift 环境下):
    python extend_vocab.py \
        --base-model /apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/Qwen2.5-VL-7B-Instruct \
        --out-dir   /apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/Qwen2.5-VL-7B-Instruct-mcvla
"""
import argparse
import json
import os

# 与 jarvisvla/__init__.py 的 QWEN_SPECIAL_TOKENS 一致
HERE = os.path.dirname(os.path.abspath(__file__))
SPECIAL_TOKEN_JSON = os.path.join(HERE, "assets", "special_token.json")


def extend(args):
    from transformers import AutoTokenizer, AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelCls
    import torch

    # 1) 读取 JarvisVLA 的特殊 token 列表 (顺序保持与官方一致)
    with open(SPECIAL_TOKEN_JSON) as f:
        special_tokens = json.load(f)
    print("special tokens in JarvisVLA assets: %d" % len(special_tokens))

    # 2) add_special_tokens (JarvisVLA train.py:96-98 同款逻辑)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    before = len(tokenizer)
    already = [t for t in special_tokens if t in tokenizer.get_vocab()]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print("tokenizer len: %d -> %d (already present: %d, newly added: %d)" % (
        before, len(tokenizer), len(already), len(special_tokens) - len(already)))

    # 校验动作保留字全部单 token 化
    for n in (176, 178, 179, 204, 219, 240, 250):
        tok = "<|reserved_special_token_%d|>" % n
        ids = tokenizer.convert_tokens_to_ids(tok)
        assert ids is not None and ids >= 0, "token missing: %s" % tok
        print("  %s -> %d" % (tok, ids))

    # 3) resize embeddings 并保存
    print("loading model (cpu, bf16) ...")
    model = ModelCls.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu")
    old_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    new_size = model.get_input_embeddings().weight.shape[0]
    print("embedding size: %d -> %d" % (old_size, new_size))

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    # 注意顺序: processor 自带原版 tokenizer, 必须先存 processor,
    # 扩展版 tokenizer 最后保存, 否则会被覆盖
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("saved -> %s" % args.out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                    help="Qwen2.5-VL-7B-Instruct 原始模型目录")
    ap.add_argument("--out-dir", required=True,
                    help="扩展词表后的新模型目录")
    args = ap.parse_args()
    extend(args)


if __name__ == "__main__":
    main()

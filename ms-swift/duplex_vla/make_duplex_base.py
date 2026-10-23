#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成全双工训练基座: 官方 JarvisVLA-Qwen2-VL-7B 或 stage2 checkpoint + 5 个控制 token.

两条路径 (由 --base 决定):
  A. --base <官方模型目录>          从官方 JarvisVLA 起做全双工 (默认)
  B. --base <stage2 checkpoint目录>  从 stage2 训练结果续训 (保留其动作+QA能力,
                                     控制 token 新增行随机初始化从头学)

输出到新目录 (不覆盖任何现有文件; 拒绝覆盖已存在的输出目录):
  - tokenizer: 注册 5 个控制 token (id 必须与 duplex_tokens.json 一致, 断言校验)
  - 模型: resize_token_embeddings(151913), 新 5 行 embedding N(0,0.02²)
  - 兼容: swift 保存的 checkpoint 的 tokenizer_config.extra_special_tokens 为 list
          时自动修复 (transformers 新版要求 dict, 否则加载即崩)

用法 (GPU 机 swift 环境, CPU 约 4 分钟):
  python make_duplex_base.py --base <官方或stage2目录> --out <新副本目录>
"""
import argparse
import json
import os
import shutil

BASE_OFFICIAL = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
                 "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B")
DEFAULT_OUT = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
               "chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B-duplex")
TOKENS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "duplex_tokens.json")

CTRL_TOKENS = ["<|duplex_user|>", "<|duplex_obs|>", "<|duplex_act|>",
               "<|duplex_txt|>", "<|silence|>"]

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
                   "chat_template.json", "preprocessor_config.json", "processor_config.json")


def stage_tokenizer_files(base, out):
    """把 base 的 tokenizer/processor 文件复制到 out 并修复兼容问题 (不动源目录).

    swift checkpoint 的 tokenizer_config.extra_special_tokens 可能是 list,
    新版 transformers 加载会 AttributeError -> 修成 {} (这些 token 已在词表注册).
    """
    os.makedirs(out, exist_ok=True)
    for f in TOKENIZER_FILES:
        src = os.path.join(base, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f))
    tc = os.path.join(out, "tokenizer_config.json")
    if os.path.exists(tc):
        d = json.load(open(tc))
        est = d.get("extra_special_tokens")
        if isinstance(est, list):
            print("修复 extra_special_tokens: list(%d) -> {} (swift 保存格式兼容)" % len(est))
            d["extra_special_tokens"] = {}
            json.dump(d, open(tc, "w"), ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_OFFICIAL,
                    help="起点模型: 官方目录 或 stage2 checkpoint 目录 (续训)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if os.path.exists(args.out) and os.listdir(args.out):
        ap.error("输出目录非空, 拒绝覆盖: %s (如需重做请先手动删除)" % args.out)

    import torch
    from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

    # 1. tokenizer 文件 staging 到 out (含 swift 兼容修复), 注册控制 token
    stage_tokenizer_files(args.base, args.out)
    tok = AutoTokenizer.from_pretrained(args.out)
    ref = json.load(open(TOKENS_PATH))["ids"]
    n_added = tok.add_tokens(CTRL_TOKENS, special_tokens=True)
    print("add_tokens 新增: %d (起点词表 len=%d)" % (n_added, len(tok) - n_added))
    for name in CTRL_TOKENS:
        got = tok.convert_tokens_to_ids(name)
        assert got == ref[name], "id 不一致 %s: %s vs %s" % (name, got, ref[name])
    tok.save_pretrained(args.out)
    print("控制 token id 与 duplex_tokens.json 一致 ✓ (len=%d)" % len(tok))

    # 2. 模型从 base 加载 (stage2 权重) + resize (新行随机初始化)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16)
    old_v = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok))
    new_v = model.get_input_embeddings().weight.shape[0]
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        if emb[new_v - 5:].abs().sum() == 0:
            emb[new_v - 5:].normal_(mean=0.0, std=0.02)
            print("新增 5 行 embedding 已手动 N(0,0.02²) 初始化")
    assert emb[new_v - 5:].abs().sum() > 0, "新 embedding 全零"
    assert new_v == len(tok), "resize 后行数 != tokenizer 长数"
    if new_v < old_v:
        # 缩小 = 砍掉 tokenizer 未使用的预留行 (官方模型 embedding 152064, 实际 token 151908+5)
        max_id = max(tok.get_vocab().values())
        assert max_id < new_v, "被砍区间含有效 token!"
        print("embedding: %d -> %d (砍 %d 行未使用预留行, 最大有效 id=%d) ✓" % (
            old_v, new_v, old_v - new_v, max_id))
    else:
        print("embedding: %d -> %d ✓" % (old_v, new_v))

    # 3. 保存 (bf16 单文件)
    model.save_pretrained(args.out)

    # 4. 副本自校验
    tok2 = AutoTokenizer.from_pretrained(args.out)
    assert tok2.convert_tokens_to_ids("<|silence|>") == ref["<|silence|>"]
    print("副本自校验 ✓")
    print("完成: %s (起点: %s)" % (args.out, args.base))


if __name__ == "__main__":
    main()

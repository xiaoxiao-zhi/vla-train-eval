#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 全双工: 给官方 JarvisVLA-Qwen2-VL-7B 扩一个 <|silence|> token, 离线保存.

输出目录 = 官方模型副本 + tokenizer 多一个 <|silence|> (词表 152k+1).
v2 训练 (swift) 直接以此目录为 --model; 权重不复制 (软链), 只新写 tokenizer 文件.

注意: 不修改官方目录本身 (v1/其他训练不受影响).
"""
import json
import os
import shutil

SRC = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B')
DST = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B-silence')
NEW_TOKEN = '<|silence|>'

os.makedirs(DST, exist_ok=True)

# 1. processor (tokenizer + image_processor) 加 <|silence|> 并保存
#    注意: swift 的 qwen2_vl template 要求 tokenizer 带 image_processor (ProcessorMixin)
from transformers import AutoProcessor  # noqa: E402
proc = AutoProcessor.from_pretrained(SRC, trust_remote_code=True)
tok = proc.tokenizer
assert NEW_TOKEN not in tok.get_added_vocab(), '官方已有 <|silence|>? 检查!'
n_before = len(tok)
tok.add_special_tokens({'additional_special_tokens': [NEW_TOKEN]})
proc.save_pretrained(DST)
n_after = len(tok)
sid = tok.convert_tokens_to_ids(NEW_TOKEN)
print('词表: %d -> %d, <|silence|> id=%d' % (n_before, n_after, sid))

# 2. 模型 resize embedding 并完整保存 (已存在则跳过)
import os  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen2VLForConditionalGeneration  # noqa: E402
if not os.path.exists(os.path.join(DST, 'model.safetensors')):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        SRC, torch_dtype=torch.bfloat16, trust_remote_code=True)
    old_emb = model.get_input_embeddings().weight
    model.resize_token_embeddings(n_after)
    new_emb = model.get_input_embeddings().weight
    with torch.no_grad():
        new_emb[sid] = old_emb.mean(dim=0)  # 新 token 用均值初始化
    model.save_pretrained(DST, safe_serialization=True)
    print('模型已 resize (%d -> %d) 并完整保存 -> %s' % (n_before, n_after, DST))
else:
    print('模型已存在, 跳过 (只更新了 processor/tokenizer)')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v4 扩词表: checkpoint + 3 个新 token (<|silence|>, <|obs_user|>, <|obs_asst|>).

用法: python expand_ckpt_vocab_v4.py <ckpt_dir> [out_dir]

处理要点:
  1. 词表: 151908 -> 151911 (+3), 均值初始化新行
  2. config 的 vocab_size 可能是 swift padding 值 (如 152064) —— 以 tokenizer
     实际长度为准 resize, 并同步改写 config (顶层 + text_config)
  3. rope_scaling 保存后自检补回 (swift 保存必丢, 已知坑)
  4. processor 从官方基座取完整版 (含 image_processor)
"""
import json
import os
import shutil
import sys

SRC_PROC = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
            'chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B')
NEW_TOKENS = ['<|silence|>', '<|obs_user|>', '<|obs_asst|>']
MROPE = {'mrope_section': [16, 24, 24], 'rope_type': 'default', 'type': 'default'}


def main():
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else ckpt.rstrip('/') + '-v4vocab'
    assert os.path.exists(os.path.join(ckpt, 'config.json')), '无 config.json'
    if os.path.exists(os.path.join(out, 'model.safetensors')):
        print('已存在, 跳过: %s' % out)
        return

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    # 1. processor (官方完整版) + tokenizer 加 3 token
    proc = AutoProcessor.from_pretrained(SRC_PROC, trust_remote_code=True)
    tok = proc.tokenizer
    for t in NEW_TOKENS:
        assert t not in tok.get_added_vocab(), '%s 已存在?' % t
    n_before = len(tok)
    tok.add_special_tokens({'additional_special_tokens': NEW_TOKENS})
    n_after = len(tok)
    ids = {t: tok.convert_tokens_to_ids(t) for t in NEW_TOKENS}
    print('词表: %d -> %d, 新 token: %s' % (n_before, n_after, ids))

    # 2. 模型 resize (以 tokenizer 为准, 不管原 config 的 padding 值)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True)
    old_emb = model.get_input_embeddings().weight
    print('原 embedding 行数: %d (config padding 值以权重为准)' % old_emb.shape[0])
    model.resize_token_embeddings(n_after)
    with torch.no_grad():
        mean_row = old_emb[:n_before].mean(dim=0)
        for t in NEW_TOKENS:
            model.get_input_embeddings().weight[ids[t]] = mean_row
    model.config.vocab_size = n_after

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    proc.save_pretrained(out)
    for f in ('generation_config.json',):
        p = os.path.join(ckpt, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, f))

    # 3. config 自检: rope_scaling + vocab_size 双写
    cfg_p = os.path.join(out, 'config.json')
    cfg = json.load(open(cfg_p))
    fixed = []
    if cfg.get('rope_scaling') is None:
        cfg['rope_scaling'] = MROPE
        fixed.append('rope顶层')
    if isinstance(cfg.get('text_config'), dict):
        if cfg['text_config'].get('rope_scaling') is None:
            cfg['text_config']['rope_scaling'] = MROPE
            fixed.append('rope.text_config')
        cfg['text_config']['vocab_size'] = n_after
    cfg['vocab_size'] = n_after
    fixed.append('vocab_size=%d' % n_after)
    json.dump(cfg, open(cfg_p, 'w'), indent=2, ensure_ascii=False)
    print('config 修正: %s' % ', '.join(fixed))

    # 4. 回读验证
    cfg2 = json.load(open(cfg_p))
    assert cfg2['rope_scaling'] == MROPE
    assert cfg2['text_config']['vocab_size'] == n_after
    print('验证通过 -> %s' % out)


if __name__ == '__main__':
    main()

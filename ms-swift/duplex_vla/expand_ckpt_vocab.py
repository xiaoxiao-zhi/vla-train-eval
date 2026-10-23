#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给任意 checkpoint 扩 <|silence|> token (v2 全双工续训前置步骤).

用法: python expand_ckpt_vocab.py <ckpt_dir> [out_dir]
  - tokenizer/processor: 加 <|silence|> (从官方基座取完整 processor)
  - 模型: resize embedding +1, 新行均值初始化, 完整保存
  - out_dir 默认 = <ckpt_dir>-silence
"""
import json
import os
import shutil
import sys

SRC_PROC = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
            'chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B')  # 官方基座 (完整 processor 来源)
NEW_TOKEN = '<|silence|>'


def main():
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else ckpt.rstrip('/') + '-silence'
    assert os.path.exists(os.path.join(ckpt, 'config.json')), 'checkpoint 无 config.json'
    if os.path.exists(os.path.join(out, 'model.safetensors')):
        print('已存在, 跳过: %s' % out)
        return

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    # 1. processor (官方完整版) + tokenizer 扩 <|silence|>
    proc = AutoProcessor.from_pretrained(SRC_PROC, trust_remote_code=True)
    tok = proc.tokenizer
    assert NEW_TOKEN not in tok.get_added_vocab()
    n_before = len(tok)
    tok.add_special_tokens({'additional_special_tokens': [NEW_TOKEN]})
    n_after = len(tok)
    sid = tok.convert_tokens_to_ids(NEW_TOKEN)

    # 2. checkpoint 模型 resize
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True)
    old_emb = model.get_input_embeddings().weight
    model.resize_token_embeddings(n_after)
    with torch.no_grad():
        model.get_input_embeddings().weight[sid] = old_emb.mean(dim=0)
    model.config.vocab_size = n_after

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    proc.save_pretrained(out)
    # generation_config 等杂项
    for f in ('generation_config.json',):
        p = os.path.join(ckpt, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, f))

    # ★ 保存后自检: save_pretrained 可能丢 rope_scaling (swift/deepspeed 保存器同款坑,
    #   duplex_v1/checkpoint-362 与 s2v3 checkpoint 均中招), 缺了立即补回
    MROPE = {'mrope_section': [16, 24, 24], 'rope_type': 'default', 'type': 'default'}
    cfg_p = os.path.join(out, 'config.json')
    cfg = json.load(open(cfg_p))
    fixed = []
    if cfg.get('rope_scaling') is None:
        cfg['rope_scaling'] = MROPE
        fixed.append('顶层')
    if isinstance(cfg.get('text_config'), dict) and cfg['text_config'].get('rope_scaling') is None:
        cfg['text_config']['rope_scaling'] = MROPE
        fixed.append('text_config')
    if fixed:
        json.dump(cfg, open(cfg_p, 'w'), indent=2, ensure_ascii=False)
        print('已补回 rope_scaling: %s' % '+'.join(fixed))
    else:
        print('rope_scaling 检查通过 (未丢失)')

    print('扩词表完成: %d -> %d, <|silence|> id=%d\n  %s -> %s' % (
        n_before, n_after, sid, ckpt, out))


if __name__ == '__main__':
    main()

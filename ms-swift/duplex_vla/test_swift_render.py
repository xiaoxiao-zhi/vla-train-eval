#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实测 swift qwen2_vl template 对 images 的渲染位置 (hist1 样本 + duplex 样本)."""
import json

from transformers import AutoProcessor
from swift.template import get_template

BASE = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
        'chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B-silence')

processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
template = get_template(processor, template_type='qwen2_vl')
template.set_mode('train')

# ---- hist1 样本 ----
for line in open('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/duplex_vla_v3/stage2_train_hist1.jsonl'):
    d = json.loads(line)
    if d.get('images') and len(d['images']) == 2:
        break
inp = {'messages': d['messages'], 'images': d['images']}
out = template.encode(inp)
ids = out['input_ids']
txt = processor.tokenizer.decode(ids, skip_special_tokens=False)
txt = txt.replace('<|image_pad|>', '[IMG]')  # 折叠图片占位
print('===== hist1 渲染结果 =====')
print(repr(txt[:600]))
print()
print('input_ids 数:', len(ids), '| labels 非mask数:', sum(1 for x in out.get('labels', []) if x != -100))

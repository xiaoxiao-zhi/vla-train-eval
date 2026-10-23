#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v3 三份数据集全量检查: 行数 / 组成 / 图片分布 / 示例."""
import json
from collections import Counter

BASE = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
        'chengzhixu/dataset/duplex_vla_v3')
OUT = '/apdcephfs/private_chengzhixu/check_v3_data.out'
lines_out = []

for name in ['stage2_train_hist0.jsonl', 'stage2_train_hist1.jsonl', 'duplex_train.jsonl']:
    n = vla = qa = 0
    img_c = Counter()
    ex_vla = ex_qa = None
    for line in open('%s/%s' % (BASE, name)):
        d = json.loads(line)
        n += 1
        imgs = d.get('images')
        if imgs:
            vla += 1
            img_c[len(imgs)] += 1
            if ex_vla is None and len(imgs) > 1:
                ex_vla = d
        else:
            qa += 1
            if ex_qa is None and len(d['messages']) == 2:
                ex_qa = d
    lines_out.append('== %s ==' % name)
    lines_out.append('  总行数 %d | VLA(带图) %d | QA(纯文本) %d | QA占比 %.1f%%' % (
        n, vla, qa, 100 * qa / n))
    lines_out.append('  VLA 图片数分布: %s' % dict(sorted(img_c.items())))
    if ex_vla:
        lines_out.append('  [VLA示例] user: %r' % ex_vla['messages'][0]['content'][:60])
        lines_out.append('            assistant: %r' % ex_vla['messages'][1]['content'][:80])
        lines_out.append('            images: %s' % [i[-18:] for i in ex_vla['images']])
    if ex_qa:
        lines_out.append('  [QA示例] user: %r' % ex_qa['messages'][0]['content'][:60])
        lines_out.append('           assistant: %r' % ex_qa['messages'][1]['content'][:60])
    lines_out.append('')

# duplex 关键示例: QA 回答轮
for line in open('%s/duplex_train.jsonl' % BASE):
    d = json.loads(line)
    tgt = d['messages'][-1]['content']
    if not tgt.endswith('<|silence|>') and len(d['messages']) > 5:
        lines_out.append('== duplex QA回答轮示例 (t=%d, %d轮对话) ==' % (d['t'], len(d['messages']) // 2))
        lines_out.append('  历史 user(首个): %r' % d['messages'][0]['content'][:50])
        lines_out.append('  历史 assistant(首个): %r' % d['messages'][1]['content'][:70])
        lines_out.append('  当前 user: %r' % d['messages'][-2]['content'][:60])
        lines_out.append('  target assistant: %r' % tgt[:100])
        lines_out.append('  images: %s' % [i[-16:] for i in d['images']])
        break

open(OUT, 'w').write('\n'.join(lines_out))
print('done')

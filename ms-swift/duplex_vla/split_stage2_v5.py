#!/usr/bin/env python3
"""把 v5 stage2.jsonl 拆成 action(VLA) 与 general-qa 两部分.

分类: 有 images 字段的行 -> action; 纯文本行 -> general_qa.
行保持原序且逐行字节不变; 输出配套 stats.json.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
SOURCE = SHARE / 'dataset/duplex_vla_v5/stage2.jsonl'
OUT_ACTION = SHARE / 'dataset/duplex_vla_v5/stage2_action.jsonl'
OUT_QA = SHARE / 'dataset/duplex_vla_v5/stage2_general_qa.jsonl'
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')
EXPECTED = {'mine_block': 99981, 'kill_entity': 99959, 'craft_item': 97330, 'qa': 195370}


def main():
    stats_action = Path(str(OUT_ACTION) + '.stats.json')
    stats_qa = Path(str(OUT_QA) + '.stats.json')
    if any(p.exists() for p in (OUT_ACTION, OUT_QA, stats_action, stats_qa)):
        raise FileExistsError('Output already exists; refusing to overwrite')
    metadata = {}
    with (SHARE / 'dataset/mcvla_video_sft/train.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            metadata[row['video_id']] = row['task'].split(':', 1)[0]

    counts = Counter()
    seg_by_kind = Counter()
    segments = set()
    source_hash = hashlib.sha256()
    with SOURCE.open('rb') as src, \
            OUT_ACTION.open('wb') as fa, \
            OUT_QA.open('wb') as fq:
        for line in src:
            source_hash.update(line)
            row = json.loads(line)
            if row.get('images'):
                match = FRAME.fullmatch(Path(row['images'][-1]).name)
                if match is None:
                    raise ValueError('Invalid current training image: %r' % row['images'][-1])
                segment = match.group(1)
                kind = metadata[segment]
                counts[kind] += 1
                segments.add(segment)
                fa.write(line)
            else:
                counts['qa'] += 1
                fq.write(line)
    if dict(counts) != EXPECTED:
        raise ValueError('Unexpected split counts: %r != %r' % (dict(counts), EXPECTED))
    for seg in segments:
        seg_by_kind[metadata[seg]] += 1
    common = {
        'source': str(SOURCE), 'source_sha256': source_hash.hexdigest(),
        'rows_byte_exact_from_source': True, 'order_preserved': True,
        'split_rule': "rows with 'images' -> action; text-only rows -> general_qa",
    }
    stats_action.write_text(json.dumps({
        **common, 'output': str(OUT_ACTION), 'rows': counts['mine_block'] + counts['kill_entity'] + counts['craft_item'],
        'rows_by_task': {k: counts[k] for k in ('mine_block', 'kill_entity', 'craft_item')},
        'segments': len(segments), 'segments_by_task': dict(seg_by_kind),
    }, ensure_ascii=False, indent=2))
    stats_qa.write_text(json.dumps({
        **common, 'output': str(OUT_QA), 'rows': counts['qa'],
        'note': 'QA answers: 145370 Qwen2.5-7B-Instruct factoid + 50000 Alpaca original (unchanged from source)',
    }, ensure_ascii=False, indent=2))
    print('SPLIT_DONE action=%d qa=%d segments=%d' % (
        counts['mine_block'] + counts['kill_entity'] + counts['craft_item'], counts['qa'], len(segments)))


if __name__ == '__main__':
    main()

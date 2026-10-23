#!/usr/bin/env python3
"""实测核验 v5 stage2.jsonl 的视频段数量与任务分布 (流式, 不修改输入)."""
import json
import re
from collections import Counter
from pathlib import Path

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
SOURCE = SHARE / 'dataset/duplex_vla_v5/stage2.jsonl'
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')

metadata = {}
with (SHARE / 'dataset/mcvla_video_sft/train.jsonl').open() as handle:
    for line in handle:
        row = json.loads(line)
        metadata[row['video_id']] = row['task'].split(':', 1)[0]

segments = set()
seg_by_kind = Counter()
rows_by_kind = Counter()
unknown_segment = []
with SOURCE.open() as handle:
    for line in handle:
        row = json.loads(line)
        images = row.get('images')
        if not images:
            rows_by_kind['qa'] += 1
            continue
        match = FRAME.fullmatch(Path(images[-1]).name)
        if match is None:
            raise ValueError('Invalid current training image: %r' % images[-1])
        segment = match.group(1)
        if segment not in metadata:
            if len(unknown_segment) < 5:
                unknown_segment.append(segment)
            continue
        segments.add(segment)
        seg_by_kind[metadata[segment]] += 0  # 仅在首次出现时计数, 用下面方式补
        rows_by_kind[metadata[segment]] += 1

# 重新按 segment 集合统计每种任务的段数
seg_kind = {}
for seg in segments:
    seg_kind[seg] = metadata[seg]
final_seg_by_kind = Counter(seg_kind.values())

print('total_rows:', sum(rows_by_kind.values()))
print('rows_by_kind:', dict(rows_by_kind))
print('distinct_segments:', len(segments))
print('segments_by_kind:', dict(final_seg_by_kind))
print('segments_not_in_metadata:', len(unknown_segment), unknown_segment)

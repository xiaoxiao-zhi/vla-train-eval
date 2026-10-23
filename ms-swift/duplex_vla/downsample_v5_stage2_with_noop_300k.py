#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
SOURCE = SHARE / 'dataset/duplex_vla_v5/stage2_with_noop.jsonl'
OUTPUT = SHARE / 'dataset/duplex_vla_v5/stage2_with_noop_300k.jsonl'
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')
NOP = '<|reserved_special_token_178|><|reserved_special_token_219|><|reserved_special_token_240|><|reserved_special_token_179|>'
EXPECTED = {'mine_block': 114413, 'kill_entity': 112435, 'craft_item': 122175, 'qa': 195370}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.target <= 0:
        raise ValueError('Target must be positive')
    stats_path = Path(str(OUTPUT) + '.stats.json')
    temporary = Path(str(OUTPUT) + '.building')
    if any(path.exists() for path in (OUTPUT, stats_path, temporary)):
        raise FileExistsError('Output or partial output already exists')
    metadata = {}
    with (SHARE / 'dataset/mcvla_video_sft/train.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['video_id'] in metadata:
                raise ValueError('Duplicate trajectory metadata')
            metadata[row['video_id']] = {'kind': row['task'].split(':', 1)[0]}

    def classify(row):
        if not row.get('images'):
            return 'qa', None, False
        match = FRAME.fullmatch(Path(row['images'][-1]).name)
        if match is None:
            raise ValueError('Invalid current training image')
        segment = match.group(1)
        return metadata[segment]['kind'], segment, row['messages'][-1]['content'].strip() == NOP

    initial_stat = SOURCE.stat()
    before, noop_before, segment_counts = Counter(), Counter(), Counter()
    source_hash, qa_hash = hashlib.sha256(), hashlib.sha256()
    with SOURCE.open('rb') as handle:
        for line in handle:
            source_hash.update(line)
            kind, segment, is_noop = classify(json.loads(line))
            before[kind] += 1
            if is_noop:
                noop_before[kind] += 1
            if segment is not None:
                segment_counts[segment] += 1
            if kind == 'qa':
                qa_hash.update(line)
    if before != EXPECTED or len(segment_counts) != 4706:
        raise ValueError(f'Unexpected noop-restored source: {before}, {len(segment_counts)} segments')
    if source_hash.hexdigest() != json.loads(Path(str(SOURCE) + '.stats.json').read_text())['sha256']:
        raise ValueError('Noop-restored source changed since publication')
    total_vla = sum(before[kind] for kind in ('mine_block', 'kill_entity', 'craft_item'))
    if total_vla < args.target:
        raise ValueError('Source has fewer VLA rows than requested target')
    segments = sorted(segment_counts)
    random.Random(args.seed).shuffle(segments)
    kept, kept_rows = [], 0
    for segment in segments:
        candidate = kept_rows + segment_counts[segment]
        if candidate >= args.target:
            if candidate - args.target < args.target - kept_rows:
                kept.append(segment)
                kept_rows = candidate
            break
        kept.append(segment)
        kept_rows = candidate
    if abs(kept_rows - args.target) > max(segment_counts[segment] for segment in segments):
        raise ValueError('Could not approach the requested target with complete segments')
    remaining = kept_rows
    kept_set = set(kept)
    after, noop_after, retained_segments = Counter(), Counter(), Counter()
    retained_hash, second_source_hash = hashlib.sha256(), hashlib.sha256()
    try:
        with SOURCE.open('rb') as handle, temporary.open('xb') as destination:
            for line in handle:
                second_source_hash.update(line)
                kind, segment, is_noop = classify(json.loads(line))
                if segment is not None and segment not in kept_set:
                    continue
                destination.write(line)
                retained_hash.update(line)
                after[kind] += 1
                if is_noop:
                    noop_after[kind] += 1
                if segment is not None:
                    retained_segments[segment] += 1
        if second_source_hash.digest() != source_hash.digest():
            raise ValueError('Source content changed between passes')
        if retained_segments != Counter({segment: segment_counts[segment] for segment in kept}):
            raise ValueError('Kept segment sample counts changed during subsampling')
        kept_vla_rows = sum(segment_counts[segment] for segment in kept)
        if after['qa'] != before['qa'] or sum(after[kind] for kind in ('mine_block', 'kill_entity', 'craft_item')) != kept_vla_rows:
            raise ValueError(f'Unexpected output composition: kept_vla={kept_vla_rows}, planned_remaining={remaining}')
        actual, actual_noop, actual_segments, actual_qa_hash = Counter(), Counter(), Counter(), hashlib.sha256()
        actual_hash = hashlib.sha256()
        with temporary.open('rb') as handle:
            for line in handle:
                kind, segment, is_noop = classify(json.loads(line))
                actual[kind] += 1
                actual_hash.update(line)
                if is_noop:
                    actual_noop[kind] += 1
                if segment is not None:
                    actual_segments[segment] += 1
                if kind == 'qa':
                    actual_qa_hash.update(line)
        if actual != after or actual_segments != retained_segments or actual_hash.digest() != retained_hash.digest():
            raise ValueError('Output verification failed')
        if actual_qa_hash.digest() != qa_hash.digest():
            raise ValueError('QA changed during subsampling')
        if actual_noop != noop_after:
            raise ValueError('No-op accounting mismatch')
        final_stat = SOURCE.stat()
        if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError('Source changed during subsampling')
        expected_segment_types = Counter(metadata[segment]['kind'] for segment in kept_set)
        stats = {
            'source': str(SOURCE), 'output': str(OUTPUT), 'seed': args.seed, 'target_vla_rows': args.target,
            'sampling': 'Shuffle all 4706 sorted segment IDs together across the three task types using Random(seed); keep a prefix of complete segments until the retained VLA row count is closest to target at the crossing boundary. No per-frame sampling, no partial segments, no input-line shuffling; QA kept in original order.',
            'before': dict(before), 'after': dict(after),
            'vla_before': total_vla, 'vla_after': kept_vla_rows,
            'difference_from_target': kept_vla_rows - args.target,
            'noop_before_by_task': dict(noop_before), 'noop_after_by_task': dict(noop_after),
            'noop_rows_after': sum(noop_after.values()),
            'noop_pct_after': round(100 * sum(noop_after.values()) / kept_vla_rows, 4),
            'segments_before': 4706, 'segments_after': len(kept_set),
            'segments_after_by_task': dict(expected_segment_types),
            'rows_after_by_task': {kind: after[kind] for kind in ('mine_block', 'kill_entity', 'craft_item')},
            'qa': after['qa'], 'total_after': sum(after.values()),
            'source_sha256': source_hash.hexdigest(), 'sha256': actual_hash.hexdigest(),
            'qa_sha256': actual_qa_hash.hexdigest(), 'bytes': temporary.stat().st_size,
            'verification': {'all_selected_segments_fully_kept': True, 'all_dropped_segments_fully_removed': True,
                             'kept_segment_sample_counts_unchanged': True, 'qa_byte_exact_in_original_order': True,
                             'input_files_unchanged': True, 'no_new_noop_filter': True,
                             'running_training_inputs_not_modified': True},
        }
        temporary.rename(OUTPUT)
        with stats_path.open('x', encoding='utf-8') as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
        print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()

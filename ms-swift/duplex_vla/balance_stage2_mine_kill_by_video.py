#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
SOURCE = SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm_drop147_droptrade76.jsonl'
OUTPUT = SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm_clean_balanced100k.jsonl'
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')
EXPECTED = {'mine_block': 129907, 'craft_item': 97330, 'kill_entity': 122111, 'qa': 195370}


def choose_removals(segment_counts, metadata, kind, target, seed):
    segments = sorted(segment for segment in segment_counts if metadata[segment]['kind'] == kind)
    remaining = sum(segment_counts[segment] for segment in segments)
    if remaining < target:
        raise ValueError(f'{kind} has fewer samples than requested target')
    random.Random(seed).shuffle(segments)
    removed = []
    for segment in segments:
        candidate = remaining - segment_counts[segment]
        if candidate <= target:
            if abs(candidate - target) < abs(remaining - target):
                removed.append(segment)
                remaining = candidate
            break
        removed.append(segment)
        remaining = candidate
    if abs(remaining - target) > max(segment_counts[segment] for segment in segments):
        raise ValueError(f'Failed to approach target for {kind}')
    return removed, remaining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=100000)
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
            segment = row['video_id']
            if segment in metadata:
                raise ValueError('Duplicate trajectory metadata')
            metadata[segment] = {'kind': row['task'].split(':', 1)[0], 'task': row['task'],
                                 'video': row['video'], 'instruction': row['instruction']}

    def classify(row):
        if not row.get('images'):
            return 'qa', None
        match = FRAME.fullmatch(Path(row['images'][-1]).name)
        if match is None:
            raise ValueError('Invalid current training image')
        segment = match.group(1)
        kind = metadata[segment]['kind']
        if kind not in ('mine_block', 'kill_entity', 'craft_item'):
            raise ValueError(f'Unexpected VLA task: {kind}')
        return kind, segment

    initial_stat = SOURCE.stat()
    before, segment_counts = Counter(), Counter()
    source_hash = hashlib.sha256()
    protected_hashes = {kind: hashlib.sha256() for kind in ('craft_item', 'qa')}
    with SOURCE.open('rb') as handle:
        for line in handle:
            source_hash.update(line)
            kind, segment = classify(json.loads(line))
            before[kind] += 1
            if segment is not None:
                segment_counts[segment] += 1
            if kind in protected_hashes:
                protected_hashes[kind].update(line)
    if before != EXPECTED or len(segment_counts) != 5622:
        raise ValueError(f'Unexpected input composition: {before}, {len(segment_counts)} segments')
    removals, expected_remaining = set(), {}
    for kind in ('mine_block', 'kill_entity'):
        chosen, remaining = choose_removals(segment_counts, metadata, kind, args.target, args.seed)
        removals.update(chosen)
        expected_remaining[kind] = remaining
        print(f'{kind}: remove {len(chosen)} complete segments, retain {remaining} training rows', flush=True)
    after, deleted_counts = Counter(), Counter()
    retained_segment_counts = Counter()
    retained_hash, second_source_hash = hashlib.sha256(), hashlib.sha256()
    try:
        with SOURCE.open('rb') as handle, temporary.open('xb') as destination:
            for line in handle:
                second_source_hash.update(line)
                kind, segment = classify(json.loads(line))
                if segment in removals:
                    deleted_counts[segment] += 1
                    continue
                destination.write(line)
                retained_hash.update(line)
                after[kind] += 1
                if segment is not None:
                    retained_segment_counts[segment] += 1
        if second_source_hash.digest() != source_hash.digest():
            raise ValueError('Source content changed between passes')
        if deleted_counts != Counter({segment: segment_counts[segment] for segment in removals}):
            raise ValueError('Incomplete removal of selected segments')
        expected_retained = Counter({segment: count for segment, count in segment_counts.items() if segment not in removals})
        if retained_segment_counts != expected_retained:
            raise ValueError('Kept segment samples were partially changed')
        if any(after[kind] != expected_remaining[kind] for kind in expected_remaining):
            raise ValueError('Unexpected balanced task counts')
        actual, actual_segments = Counter(), Counter()
        actual_hash = hashlib.sha256()
        actual_protected_hashes = {kind: hashlib.sha256() for kind in ('craft_item', 'qa')}
        with temporary.open('rb') as handle:
            for line in handle:
                kind, segment = classify(json.loads(line))
                if segment in removals:
                    raise ValueError('Removed segment still present')
                actual[kind] += 1
                actual_hash.update(line)
                if segment is not None:
                    actual_segments[segment] += 1
                if kind in actual_protected_hashes:
                    actual_protected_hashes[kind].update(line)
        if actual != after or actual_segments != expected_retained or actual_hash.digest() != retained_hash.digest():
            raise ValueError('Output verification failed')
        if any(actual_protected_hashes[kind].digest() != protected_hashes[kind].digest() or after[kind] != before[kind]
               for kind in protected_hashes):
            raise ValueError('Craft or QA changed')
        final_stat = SOURCE.stat()
        if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError('Source changed during filtering')
        before_segment_types = Counter(metadata[segment]['kind'] for segment in segment_counts)
        after_segment_types = Counter(metadata[segment]['kind'] for segment in actual_segments)
        stats = {
            'source': str(SOURCE), 'output': str(OUTPUT), 'seed': args.seed, 'target_rows_per_task': args.target,
            'sampling': 'For each of mine_block and kill_entity, independently shuffle sorted segment IDs using Random(seed), then remove a prefix of complete segments until the retained training-row count is closest to target at the crossing boundary. No per-frame sampling, no partial segments, no input-line shuffling.',
            'before': dict(before), 'after': dict(after),
            'segments_before': dict(before_segment_types), 'segments_after': dict(after_segment_types),
            'total_before': sum(before.values()), 'total_after': sum(after.values()),
            'vla_after': sum(after.values()) - after['qa'],
            'removed_segments': len(removals), 'removed_training_rows': sum(deleted_counts.values()),
            'removed_by_task': {
                kind: {'segments': before_segment_types[kind] - after_segment_types[kind],
                       'training_rows': before[kind] - after[kind], 'retained_rows_minus_target': after[kind] - args.target}
                for kind in expected_remaining
            },
            'source_sha256': source_hash.hexdigest(), 'output_sha256': actual_hash.hexdigest(),
            'qa_sha256': actual_protected_hashes['qa'].hexdigest(),
            'craft_sha256': actual_protected_hashes['craft_item'].hexdigest(), 'bytes': temporary.stat().st_size,
            'verification': {'all_selected_segments_completely_removed': True,
                             'all_kept_segment_sample_counts_unchanged': True,
                             'retained_lines_byte_exact_in_original_order': True,
                             'craft_and_qa_byte_exact_in_original_order': True, 'source_file_unchanged': True},
            'removal_list': [{'segment_id': segment, **metadata[segment], 'removed_rows': deleted_counts[segment]}
                             for segment in sorted(removals)],
        }
        temporary.rename(OUTPUT)
        with stats_path.open('x', encoding='utf-8') as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in stats.items() if k != 'removal_list'}, ensure_ascii=False, indent=2), flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()

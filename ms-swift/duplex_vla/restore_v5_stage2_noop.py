#!/usr/bin/env python3
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from build_flat3_ablation_full_vla import read_rows, signature

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
BASELINE = SHARE / 'dataset/duplex_vla_v5/stage2.jsonl'
ORIGINAL = SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3.jsonl'
OUTPUT = SHARE / 'dataset/duplex_vla_v5/stage2_with_noop.jsonl'
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')
NOP = '<|reserved_special_token_178|><|reserved_special_token_219|><|reserved_special_token_240|><|reserved_special_token_179|>'


def frame_key(row):
    match = FRAME.fullmatch(Path(row['images'][-1]).name)
    if match is None:
        raise ValueError('Unrecognized current-target image')
    return match.group(1), int(match.group(2))


def main():
    stats_path = Path(str(OUTPUT) + '.stats.json')
    temporary = Path(str(OUTPUT) + '.building')
    if any(path.exists() for path in (OUTPUT, stats_path, temporary)):
        raise FileExistsError('Output or partial output already exists')
    initial_stats = {path: path.stat() for path in (BASELINE, ORIGINAL)}
    rows, current_frames = [], {}
    expected_qa, expected_existing, expected_added = Counter(), Counter(), Counter()
    before_noop, added_noop = Counter(), Counter()
    before_counts = Counter()
    source_hash = hashlib.sha256()
    metadata = {}
    with (SHARE / 'dataset/mcvla_video_sft/train.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['video_id'] in metadata:
                raise ValueError('Duplicate original trajectory metadata')
            metadata[row['video_id']] = {'kind': row['task'].split(':', 1)[0], 'n_frames': row['n_frames']}
    for line, is_vla in read_rows(BASELINE):
        rows.append(line)
        source_hash.update(line)
        expected_existing[signature(line)] += 1
        row = json.loads(line)
        if not is_vla:
            before_counts['qa'] += 1
            expected_qa[signature(line)] += 1
            continue
        key = frame_key(row)
        if key in current_frames:
            raise ValueError(f'Duplicate baseline target: {key}')
        current_frames[key] = signature(line)
        kind = metadata[key[0]]['kind']
        before_counts[kind] += 1
        if row['messages'][-1]['content'].strip() == NOP:
            before_noop[kind] += 1
    if before_counts != {'mine_block': 99981, 'kill_entity': 99959, 'craft_item': 97330, 'qa': 195370}:
        raise ValueError(f'Unexpected v5 baseline: {before_counts}')
    if source_hash.hexdigest() != json.loads(Path(str(BASELINE) + '.stats.json').read_text())['sha256']:
        raise ValueError('Published v5 input changed')
    retained_segments = {segment for segment, _ in current_frames}
    if len(retained_segments) != 4706:
        raise ValueError('Unexpected current v5 segment set')
    original_frames, additions = set(), set()
    for line, is_vla in read_rows(ORIGINAL):
        if not is_vla:
            continue
        row = json.loads(line)
        key = frame_key(row)
        if key[0] not in retained_segments:
            continue
        if key in original_frames:
            raise ValueError(f'Duplicate original target: {key}')
        original_frames.add(key)
        if key in current_frames:
            if signature(line) != current_frames[key]:
                raise ValueError(f'Existing sample differs from original flat3 format: {key}')
            continue
        if row['messages'][-1]['content'].strip() != NOP:
            raise ValueError(f'Missing frame is not a no-op; refusing to restore unrelated sample: {key}')
        for image in row['images']:
            if not Path(image).is_file():
                raise FileNotFoundError(image)
        additions.add(key)
        rows.append(line)
        expected_added[signature(line)] += 1
        added_noop[metadata[key[0]]['kind']] += 1
    expected_frames = {(segment, step) for segment in retained_segments for step in range(metadata[segment]['n_frames'])}
    if original_frames != expected_frames or set(current_frames) | additions != expected_frames:
        raise ValueError('Restored samples do not exactly cover complete current video segments')
    random.Random(42).shuffle(rows)
    try:
        with temporary.open('xb') as handle:
            handle.writelines(rows)
        del rows
        actual_lines, actual_qa, actual_counts, actual_noop = Counter(), Counter(), Counter(), Counter()
        actual_frames = set()
        output_hash = hashlib.sha256()
        for line, is_vla in read_rows(temporary):
            output_hash.update(line)
            actual_lines[signature(line)] += 1
            row = json.loads(line)
            if not is_vla:
                actual_counts['qa'] += 1
                actual_qa[signature(line)] += 1
                continue
            key = frame_key(row)
            if key in actual_frames:
                raise ValueError('Duplicate restored output target')
            actual_frames.add(key)
            kind = metadata[key[0]]['kind']
            actual_counts[kind] += 1
            if row['messages'][-1]['content'].strip() == NOP:
                actual_noop[kind] += 1
        if actual_lines != expected_existing + expected_added or actual_qa != expected_qa:
            raise ValueError('Existing samples or LLM QA content changed')
        if actual_frames != expected_frames or actual_noop != before_noop + added_noop:
            raise ValueError('Restored no-op or target set mismatch')
        for path, initial in initial_stats.items():
            final = path.stat()
            if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                raise ValueError(f'Input file changed: {path}')
        stats = {
            'output': str(OUTPUT), 'baseline': str(BASELINE), 'original_noop_source': str(ORIGINAL),
            'baseline_sha256': source_hash.hexdigest(), 'seed': 42,
            'before': dict(before_counts), 'after': dict(actual_counts),
            'added_noop_by_task': dict(added_noop), 'added_noop_total': sum(added_noop.values()),
            'noop_before_by_task': dict(before_noop), 'noop_after_by_task': dict(actual_noop),
            'vla': sum(actual_counts.values()) - actual_counts['qa'], 'qa': actual_counts['qa'],
            'total': sum(actual_counts.values()), 'video_segments': len(retained_segments),
            'video_segments_by_task': dict(Counter(metadata[segment]['kind'] for segment in retained_segments)),
            'sha256': output_hash.hexdigest(), 'bytes': temporary.stat().st_size,
            'construction': 'Keep every current v5 sample byte-exact; restore only missing canonical no-op rows from original flat3 for the same retained 4706 segments; seed-42 shuffle all rows to mix restored no-ops. No deleted segment is reintroduced.',
            'qa_policy': 'Exact current v5 QA multiset: existing 145370 LLM factoid answers plus 50000 original Alpaca answers. Original flat3 QA is not used.',
            'verification': {'all_current_samples_byte_exact_multiset_preserved': True,
                             'qa_byte_exact_multiset_preserved': True, 'only_missing_noop_targets_added': True,
                             'complete_current_segment_frame_sets_restored': True, 'deleted_segments_not_restored': True,
                             'added_image_files_exist': True, 'input_files_unchanged': True,
                             'running_training_inputs_and_scripts_not_modified': True},
        }
        temporary.rename(OUTPUT)
        with stats_path.open('x') as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
        print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()

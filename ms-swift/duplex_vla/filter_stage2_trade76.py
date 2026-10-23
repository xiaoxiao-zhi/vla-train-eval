#!/usr/bin/env python3
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
SOURCE = SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm_drop147.jsonl'
OUTPUT = SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm_drop147_droptrade76.jsonl'
AUDIT = Path('/apdcephfs/private_chengzhixu/JarvisVLA/outputs/stage2_craft_villager_trade_audit/audit.json')
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')


def main():
    stats_path = Path(str(OUTPUT) + '.stats.json')
    temporary = Path(str(OUTPUT) + '.building')
    if any(path.exists() for path in (OUTPUT, stats_path, temporary)):
        raise FileExistsError('Output or partial output already exists')
    audit = json.loads(AUDIT.read_text())
    if Path(audit['dataset']).resolve() != SOURCE.resolve():
        raise ValueError('Audit source differs from filtering source')
    removals = {row['segment_id']: row for row in audit['samples'] if row['confirmed_trade_in_segment']}
    if len(removals) != 76 or sum(row['training_rows'] for row in removals.values()) != 6866:
        raise ValueError('Unexpected trade removal set')
    tasks = {}
    with (SHARE / 'dataset/mcvla_video_sft/train.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['video_id'] in tasks:
                raise ValueError('Duplicate video metadata')
            tasks[row['video_id']] = row['task'].split(':', 1)[0]
    for segment in removals:
        if tasks[segment] != 'craft_item':
            raise ValueError(f'Non-craft removal: {segment}')

    def classify(row):
        if not row.get('images'):
            return 'qa', None
        match = FRAME.fullmatch(Path(row['images'][-1]).name)
        if match is None:
            raise ValueError('Unrecognized current training image')
        segment = match.group(1)
        kind = tasks[segment]
        if kind not in ('mine_block', 'craft_item', 'kill_entity'):
            raise ValueError(f'Unexpected task: {kind}')
        return kind, segment

    before, after, deleted = Counter(), Counter(), Counter()
    before_segments, after_segments = set(), set()
    source_hash, kept_hash, qa_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    initial_stat = SOURCE.stat()
    try:
        with SOURCE.open('rb') as handle, temporary.open('xb') as destination:
            for line in handle:
                source_hash.update(line)
                row = json.loads(line)
                kind, segment = classify(row)
                before[kind] += 1
                if segment is not None:
                    before_segments.add(segment)
                if segment in removals:
                    instruction, separator, _ = row['messages'][0]['content'].partition('\nobservation:')
                    if not separator or instruction != removals[segment]['instruction']:
                        raise ValueError(f'Removal instruction mismatch: {segment}')
                    deleted[segment] += 1
                    continue
                destination.write(line)
                kept_hash.update(line)
                after[kind] += 1
                if segment is not None:
                    after_segments.add(segment)
                if kind == 'qa':
                    qa_hash.update(line)
        expected_before = {'mine_block': 129907, 'craft_item': 104196, 'kill_entity': 122111, 'qa': 195370}
        if before != expected_before:
            raise ValueError(f'Unexpected input counts: {before}')
        if deleted != Counter({segment: row['training_rows'] for segment, row in removals.items()}):
            raise ValueError('Removed frame counts differ from the audited 76 segments')
        actual, actual_segments = Counter(), set()
        actual_hash, actual_qa_hash = hashlib.sha256(), hashlib.sha256()
        with temporary.open('rb') as handle:
            for line in handle:
                kind, segment = classify(json.loads(line))
                if segment in removals:
                    raise ValueError(f'Removed segment still present: {segment}')
                actual[kind] += 1
                actual_hash.update(line)
                if segment is not None:
                    actual_segments.add(segment)
                if kind == 'qa':
                    actual_qa_hash.update(line)
        if actual != after or actual_hash.digest() != kept_hash.digest() or actual_segments != after_segments:
            raise ValueError('Output does not exactly preserve retained input lines')
        if actual_qa_hash.digest() != qa_hash.digest() or after['qa'] != before['qa']:
            raise ValueError('QA changed')
        final_stat = SOURCE.stat()
        if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise ValueError('Input changed during filtering')
        stats = {
            'source': str(SOURCE), 'output': str(OUTPUT), 'audit_source': str(AUDIT),
            'criterion': 'Remove all samples of the 76 craft segments with a confirmed in-segment increase in traded_with_villager. Keep nonzero history-only and uncertain first-observation segments.',
            'before': dict(before), 'after': dict(after),
            'total_before': sum(before.values()), 'total_after': sum(after.values()),
            'vla_before': sum(before.values()) - before['qa'], 'vla_after': sum(after.values()) - after['qa'],
            'removed_segments': len(deleted), 'removed_vla_rows': sum(deleted.values()),
            'vla_segments_before': len(before_segments), 'vla_segments_after': len(after_segments),
            'craft_segments_after': sum(tasks[segment] == 'craft_item' for segment in after_segments),
            'source_sha256': source_hash.hexdigest(), 'output_sha256': actual_hash.hexdigest(),
            'qa_sha256': actual_qa_hash.hexdigest(), 'bytes': temporary.stat().st_size,
            'verification': {'all_76_segments_removed': True, 'retained_lines_byte_exact_in_original_order': True,
                             'qa_byte_exact_in_original_order': True, 'source_file_unchanged': True,
                             'no_new_noop_filter_or_sampling': True},
            'removal_list': [{'segment_id': segment, 'task': row['task'], 'instruction': row['instruction'],
                              'video': row['video'], 'source_jsonl': row['source_jsonl'],
                              'removed_rows': deleted[segment], 'trade_events': row['trade_events']}
                             for segment, row in sorted(removals.items())],
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

#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
AUDIT_DIR = Path('/apdcephfs/private_chengzhixu/JarvisVLA/outputs')
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')
CATEGORY_COUNTS = {
    '按原版常规配方无法直接合成的目标': 22,
    '熔炼/烹饪（含目标属于熔炼但文字笼统的描述）': 105,
    '熔炼后再合成的混合描述': 1,
    '配方或产物表述疑点': 13,
    '附魔/锻造升级': 6,
}


def load_removals():
    audit = json.loads((AUDIT_DIR / 'stage2_craft_description_audit.json').read_text())
    samples = {row['segment_id']: row for row in audit['samples']}
    text = (AUDIT_DIR / 'stage2_craft_video_description_check.txt').read_text()
    removals = {}
    categories = Counter()
    for block in text.split('=' * 80)[1:]:
        lines = block.strip().splitlines()
        match = re.fullmatch(r'\d+\. (.+)', lines[0])
        if match is None:
            raise ValueError('Unrecognized audit entry')
        segment = match.group(1)
        category = next(line.removeprefix('分类：') for line in lines if line.startswith('分类：'))
        if category.startswith('其余合成/制作描述'):
            continue
        if category not in CATEGORY_COUNTS or segment in removals:
            raise ValueError(f'Invalid removal entry: {segment}, {category}')
        sample = samples[segment]
        if not sample['task'].startswith('craft_item:'):
            raise ValueError(f'Non-craft removal: {segment}')
        if len(sample['training_descriptions']) != 1:
            raise ValueError(f'Multiple descriptions: {segment}')
        instruction = next(iter(sample['training_descriptions']))
        if sample['metadata_instruction'] != instruction:
            raise ValueError(f'Audit instruction mismatch: {segment}')
        removals[segment] = {
            'category': category, 'task': sample['task'], 'instruction': instruction,
            'video': sample['video'],
        }
        categories[category] += 1
    if categories != CATEGORY_COUNTS or len(removals) != 147:
        raise ValueError(f'Unexpected removal set: {categories}')
    return removals


def row_kind(row, metadata):
    if not row.get('images'):
        return 'qa', None
    match = FRAME.fullmatch(Path(row['images'][-1]).name)
    if match is None:
        raise ValueError('Unrecognized current-target image')
    segment = match.group(1)
    task = metadata[segment]['task'].split(':', 1)[0]
    if task not in ('mine_block', 'craft_item', 'kill_entity'):
        raise ValueError(f'Unexpected task: {task}')
    return task, segment


def filter_dataset(source, output):
    stats_path = Path(str(output) + '.stats.json')
    temporary = Path(str(output) + '.building')
    if source.resolve() == output.resolve():
        raise ValueError('Output cannot overwrite source')
    if any(path.exists() for path in (output, stats_path, temporary)):
        raise FileExistsError('Output or partial output already exists')
    removals = load_removals()
    metadata = {}
    with (SHARE / 'dataset/mcvla_video_sft/train.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            segment = row['video_id']
            if segment in metadata:
                raise ValueError(f'Duplicate trajectory metadata: {segment}')
            metadata[segment] = {'task': row['task'], 'instruction': row['instruction']}
    for segment, removal in removals.items():
        if metadata[segment] != {'task': removal['task'], 'instruction': removal['instruction']}:
            raise ValueError(f'Removal metadata changed: {segment}')
    before = Counter()
    after = Counter()
    deleted = Counter()
    deleted_categories = Counter()
    before_segments = set()
    after_segments = set()
    source_digest = hashlib.sha256()
    kept_digest = hashlib.sha256()
    qa_digest = hashlib.sha256()
    source_stat = source.stat()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open('rb') as handle, temporary.open('xb') as destination:
            for line_number, line in enumerate(handle, 1):
                source_digest.update(line)
                row = json.loads(line)
                kind, segment = row_kind(row, metadata)
                before[kind] += 1
                if segment is not None:
                    before_segments.add(segment)
                if segment in removals:
                    instruction, separator, _ = row['messages'][0]['content'].partition('\nobservation:')
                    if not separator or instruction != removals[segment]['instruction']:
                        raise ValueError(f'Training instruction mismatch at {source}:{line_number}')
                    deleted[segment] += 1
                    deleted_categories[removals[segment]['category']] += 1
                    continue
                destination.write(line)
                kept_digest.update(line)
                after[kind] += 1
                if segment is not None:
                    after_segments.add(segment)
                if kind == 'qa':
                    qa_digest.update(line)
        if before['qa'] != 195370 or sum(before.values()) - before['qa'] != 370104:
            raise ValueError(f'Unexpected source composition: {before}')
        if set(deleted) != set(removals):
            raise ValueError(f'Removal segments absent from source: {sorted(set(removals) - set(deleted))}')
        actual = Counter()
        actual_digest = hashlib.sha256()
        actual_qa_digest = hashlib.sha256()
        with temporary.open('rb') as handle:
            for line in handle:
                row = json.loads(line)
                kind, segment = row_kind(row, metadata)
                if segment in removals:
                    raise ValueError(f'Removed segment still present: {segment}')
                actual[kind] += 1
                actual_digest.update(line)
                if kind == 'qa':
                    actual_qa_digest.update(line)
        if actual != after or actual_digest.digest() != kept_digest.digest():
            raise ValueError('Output differs from retained source lines')
        if actual_qa_digest.digest() != qa_digest.digest() or after['qa'] != before['qa']:
            raise ValueError('QA was changed')
        current_stat = source.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns):
            raise ValueError('Source changed during filtering')
        stats = {
            'source': str(source), 'output': str(output),
            'criterion': 'Remove all target samples from the 147 user-selected trajectory segments; no additional noop filtering, resampling, or shuffling.',
            'before': dict(before), 'after': dict(after),
            'total_before': sum(before.values()), 'total_after': sum(after.values()),
            'vla_before': sum(before.values()) - before['qa'],
            'vla_after': sum(after.values()) - after['qa'],
            'removed_vla_rows': sum(deleted.values()), 'removed_segments': len(deleted),
            'vla_segments_before': len(before_segments), 'vla_segments_after': len(after_segments),
            'removed_categories': {
                category: {'segments': CATEGORY_COUNTS[category], 'training_rows': deleted_categories[category]}
                for category in CATEGORY_COUNTS
            },
            'source_sha256': source_digest.hexdigest(), 'output_sha256': actual_digest.hexdigest(),
            'qa_sha256': actual_qa_digest.hexdigest(), 'bytes': temporary.stat().st_size,
            'verification': {'retained_lines_byte_exact_and_order_unchanged': True,
                             'qa_byte_exact_and_order_unchanged': True,
                             'all_147_segments_removed': True, 'source_file_unchanged': True},
            'removal_list': [{'segment_id': segment, **removals[segment], 'removed_rows': deleted[segment]}
                             for segment in sorted(removals)],
        }
        temporary.rename(output)
        with stats_path.open('x', encoding='utf-8') as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
        print(json.dumps({key: value for key, value in stats.items() if key != 'removal_list'},
                         ensure_ascii=False, indent=2), flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm.jsonl')
    parser.add_argument('--output', type=Path, default=SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm_drop147.jsonl')
    args = parser.parse_args()
    filter_dataset(args.source, args.output)


if __name__ == '__main__':
    main()

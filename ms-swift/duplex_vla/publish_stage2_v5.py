#!/usr/bin/env python3
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
SOURCE = SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm_clean_balanced100k.jsonl'
QA_SOURCE = SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm.jsonl'
OUTPUT = SHARE / 'dataset/duplex_vla_v5/stage2.jsonl'
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')


def main():
    stats_path = Path(str(OUTPUT) + '.stats.json')
    temporary = Path(str(OUTPUT) + '.building')
    if any(path.exists() for path in (OUTPUT, stats_path, temporary)):
        raise FileExistsError('V5 output or partial output already exists')
    baseline_stats = json.loads(Path(str(SOURCE) + '.stats.json').read_text())
    qa_source_stats = json.loads(Path(str(QA_SOURCE) + '.stats.json').read_text())
    if qa_source_stats['alpaca_kept_original'] != 50000 or qa_source_stats['factoid_kept_llm_answer'] != 145370:
        raise ValueError('Unexpected LLM QA source policy')
    qa_source_hash, qa_source_counts = hashlib.sha256(), Counter()
    qa_source_initial = QA_SOURCE.stat()
    with QA_SOURCE.open('rb') as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get('images'):
                qa_source_hash.update(line)
                qa_source_counts[row['datasource']] += 1
    expected_qa = {'alpaca': 50000, 'squad_v2': 61997, 'trivia_qa': 46298, 'natural_questions': 37075}
    if qa_source_counts != expected_qa:
        raise ValueError(f'Unexpected LLM QA sources: {qa_source_counts}')
    if qa_source_hash.hexdigest() != baseline_stats['qa_sha256']:
        raise ValueError('Balanced data QA hash does not match original LLM QA version')
    tasks = {}
    with (SHARE / 'dataset/mcvla_video_sft/train.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['video_id'] in tasks:
                raise ValueError('Duplicate video metadata')
            tasks[row['video_id']] = row['task'].split(':', 1)[0]
    source_initial = SOURCE.stat()
    counts, qa_counts, segments = Counter(), Counter(), {}
    source_hash, qa_hash = hashlib.sha256(), hashlib.sha256()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SOURCE.open('rb') as handle, temporary.open('xb') as destination:
            for line in handle:
                row = json.loads(line)
                messages = row['messages']
                if len(messages) != 2 or messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
                    raise ValueError('Invalid trained flat3 single-round format')
                if row.get('images'):
                    images = row['images']
                    if not 1 <= len(images) <= 3 or messages[0]['content'].count('<image>') != len(images):
                        raise ValueError('Invalid trained flat3 image format')
                    match = FRAME.fullmatch(Path(images[-1]).name)
                    if match is None:
                        raise ValueError('Invalid current-target image name')
                    segment = match.group(1)
                    kind = tasks[segment]
                    if kind not in ('mine_block', 'kill_entity', 'craft_item'):
                        raise ValueError(f'Unexpected VLA task: {kind}')
                    counts[kind] += 1
                    segments[segment] = kind
                else:
                    counts['qa'] += 1
                    qa_counts[row['datasource']] += 1
                    qa_hash.update(line)
                destination.write(line)
                source_hash.update(line)
        if counts != baseline_stats['after']:
            raise ValueError(f'V5 composition differs from balanced source: {counts}')
        if qa_counts != qa_source_counts or qa_hash.digest() != qa_source_hash.digest():
            raise ValueError('V5 QA differs from the existing LLM QA version')
        if source_hash.hexdigest() != baseline_stats['output_sha256']:
            raise ValueError('Balanced source was changed')
        output_hash = hashlib.sha256()
        with temporary.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                output_hash.update(chunk)
        if output_hash.digest() != source_hash.digest():
            raise ValueError('V5 output is not byte-identical to the approved mixed dataset')
        for path, initial in ((SOURCE, source_initial), (QA_SOURCE, qa_source_initial)):
            final = path.stat()
            if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                raise ValueError(f'Input changed during publishing: {path}')
        segment_counts = Counter(segments.values())
        stats = {
            'version': 'v5', 'output': str(OUTPUT), 'mixed_source': str(SOURCE), 'qa_source': str(QA_SOURCE),
            'format': 'Existing trained flat3 messages and images, unchanged; absolute image paths retained.',
            'composition': dict(counts), 'vla': sum(counts.values()) - counts['qa'],
            'qa': counts['qa'], 'total': sum(counts.values()), 'video_segments': dict(segment_counts),
            'qa_datasources': dict(qa_counts),
            'qa_policy': {'factoid_llm_answers': 145370, 'alpaca_original_answers': 50000,
                          'factoid_sources': ['squad_v2', 'trivia_qa', 'natural_questions'],
                          'new_answers_generated': False, 'original_300k_qa_used': False},
            'vla_policy': {'source_noop_keep_rate': 0.1, 'removed_nonordinary_craft_segments': 147,
                           'removed_confirmed_trade_segments': 76,
                           'mine_kill_sampling': 'Random removal of complete segment target samples to approximately 100000 rows per class.',
                           'sampling_seed': baseline_stats['seed']},
            'sha256': output_hash.hexdigest(), 'qa_sha256': qa_hash.hexdigest(), 'bytes': temporary.stat().st_size,
            'verification': {'output_byte_identical_to_approved_balanced_mixed_source': True,
                             'qa_byte_identical_to_llm_qa_source_in_original_order': True,
                             'no_qa_duplicate_append': True, 'trained_format_preserved': True,
                             'input_files_unchanged': True},
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

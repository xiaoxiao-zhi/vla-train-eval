#!/usr/bin/env python
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def read_rows(path):
    with path.open('rb') as file:
        for index, line in enumerate(file, 1):
            record = json.loads(line)
            messages = record['messages']
            if not messages or messages[-1]['role'] != 'assistant':
                raise ValueError('Missing assistant target at %s:%d' % (path, index))
            is_vla = bool(record.get('images'))
            if is_vla:
                placeholders = sum(message['content'].count('<image>') for message in messages)
                if not 1 <= len(record['images']) <= 3 or placeholders != len(record['images']):
                    raise ValueError('Invalid flat3 images at %s:%d' % (path, index))
            yield line.rstrip(b'\r\n') + b'\n', is_vla


def signature(line):
    return hashlib.sha256(line).digest()


def build(source, baseline, output, seed):
    stats_path = Path(str(output) + '.stats.json')
    temporary = Path(str(output) + '.building')
    if any(path.exists() for path in (output, stats_path, temporary)):
        raise FileExistsError('Output or partial output already exists')
    if output.resolve() in (source.resolve(), baseline.resolve()):
        raise ValueError('Output must not replace either input')
    rows = []
    expected_vla, expected_qa = Counter(), Counter()
    source_counts = Counter()
    baseline_counts = Counter()
    for line, is_vla in read_rows(source):
        source_counts['vla' if is_vla else 'qa'] += 1
        if is_vla:
            rows.append(line)
            expected_vla[signature(line)] += 1
    for line, is_vla in read_rows(baseline):
        baseline_counts['vla' if is_vla else 'qa'] += 1
        if not is_vla:
            rows.append(line)
            expected_qa[signature(line)] += 1
    if baseline_counts != {'vla': 300000, 'qa': 195370}:
        raise ValueError('Unexpected current v3 composition: %s' % baseline_counts)
    if not expected_vla or not expected_qa:
        raise ValueError('Empty source VLA or current QA')
    random.Random(seed).shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('xb') as file:
        file.writelines(rows)
    del rows
    actual_vla, actual_qa = Counter(), Counter()
    output_counts = Counter()
    digest = hashlib.sha256()
    for line, is_vla in read_rows(temporary):
        output_counts['vla' if is_vla else 'qa'] += 1
        digest.update(line)
        counts = actual_vla if is_vla else actual_qa
        counts[signature(line)] += 1
    if actual_vla != expected_vla or actual_qa != expected_qa:
        raise ValueError('Output does not exactly preserve source VLA and current QA')
    expected_counts = {'vla': source_counts['vla'], 'qa': baseline_counts['qa']}
    if output_counts != expected_counts:
        raise ValueError('Output partition counts mismatch')
    stats = {
        'vla_source': str(source), 'qa_source': str(baseline), 'output': str(output),
        'seed': seed, 'source_counts': dict(source_counts),
        'baseline_counts': dict(baseline_counts), **expected_counts,
        'total': sum(output_counts.values()), 'bytes': temporary.stat().st_size,
        'sha256': digest.hexdigest(),
        'verification': {'vla_multiset_equals_original': True, 'qa_multiset_equals_v3': True,
                         'sample_lines_preserved': True, 'input_files_unchanged': True},
        'construction': 'All original flat3 VLA, including its noop samples, plus current v3 QA; no resampling or new noop definition.',
        'comparison_note': 'This restores the entire original VLA partition, not only noop rows; it also undoes v3 VLA subsampling.',
    }
    temporary.rename(output)
    with stats_path.open('x', encoding='utf-8') as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build(args.source, args.baseline, args.out, args.seed)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from build_flat3_ablation_full_vla import read_rows, signature


def downsample(source, output, n, seed):
    stats_path = Path(str(output) + '.stats.json')
    temporary = Path(str(output) + '.building')
    if output.resolve() == source.resolve():
        raise ValueError('Output must not replace the full dataset')
    if any(path.exists() for path in (output, stats_path, temporary)):
        raise FileExistsError('Output or partial output already exists')
    vla, qa = [], []
    source_digest = hashlib.sha256()
    for line, is_vla in read_rows(source):
        source_digest.update(line)
        (vla if is_vla else qa).append(line)
    source_counts = {'vla': len(vla), 'qa': len(qa)}
    if source_counts != {'vla': 435249, 'qa': 195370}:
        raise ValueError('Unexpected full dataset counts: %s' % source_counts)
    if not 0 < n <= len(vla):
        raise ValueError('Invalid VLA target count: %d' % n)
    source_stats = json.loads(Path(str(source) + '.stats.json').read_text())
    if source_digest.hexdigest() != source_stats['sha256']:
        raise ValueError('Full dataset differs from its verified SHA256')
    rng = random.Random(seed)
    selected = rng.sample(vla, n)
    expected_vla = Counter(map(signature, selected))
    expected_qa = Counter(map(signature, qa))
    source_vla = Counter(map(signature, vla))
    if expected_vla - source_vla:
        raise ValueError('Selected VLA is not a subset of the input')
    rows = selected + qa
    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('xb') as file:
        file.writelines(rows)
    del rows, selected, vla, qa
    actual_vla, actual_qa = Counter(), Counter()
    output_counts = Counter()
    output_digest = hashlib.sha256()
    for line, is_vla in read_rows(temporary):
        output_digest.update(line)
        output_counts['vla' if is_vla else 'qa'] += 1
        (actual_vla if is_vla else actual_qa)[signature(line)] += 1
    if actual_vla != expected_vla or actual_qa != expected_qa:
        raise ValueError('Output differs from the selected VLA or full QA partition')
    if output_counts != {'vla': n, 'qa': source_counts['qa']}:
        raise ValueError('Output counts mismatch: %s' % output_counts)
    stats = {
        'source': str(source), 'output': str(output), 'seed': seed,
        'sampling': 'Uniform random.sample without replacement over all VLA rows; no noop filtering; all QA retained; mixed rows shuffled.',
        'source_counts': source_counts, 'vla': n, 'qa': source_counts['qa'],
        'total': sum(output_counts.values()), 'vla_removed': source_counts['vla'] - n,
        'bytes': temporary.stat().st_size, 'sha256': output_digest.hexdigest(),
        'source_sha256': source_digest.hexdigest(),
        'verification': {'vla_is_exact_selected_subset': True, 'qa_multiset_unchanged': True,
                         'sample_lines_preserved': True, 'source_file_unchanged': True},
    }
    temporary.rename(output)
    with stats_path.open('x', encoding='utf-8') as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--n', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    downsample(args.source, args.out, args.n, args.seed)


if __name__ == '__main__':
    main()

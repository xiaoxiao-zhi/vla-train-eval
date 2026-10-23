#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
DATASET = SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm_drop147_droptrade76.jsonl'
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')
EXPECTED = {'craft_item': (1156, 97330), 'mine_block': (2222, 129907), 'kill_entity': (2244, 122111)}
ALIASES = {'craft_item': 'craft', 'mine_block': 'mine', 'kill_entity': 'kill'}
LABELS = {
    'confirmed_increase': '目标计数有可确认的数值增加',
    'first_positive_appearance_after_start': '片段开始后首次出现正值，但没有可确认的数值增量',
    'positive_at_start_no_increase': '片段首帧已有正值，但未观察到增加',
    'no_positive_value_or_increase': '未观察到目标正值或增加',
}


def load_jobs(task_type):
    metadata = {}
    with (SHARE / 'dataset/mcvla_video_sft/train.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            segment = row['video_id']
            if segment in metadata:
                raise ValueError('Duplicate video metadata')
            metadata[segment] = {k: v for k, v in row.items() if k != 'actions'}
    counts = Counter()
    with DATASET.open() as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get('images'):
                continue
            match = FRAME.fullmatch(Path(row['images'][-1]).name)
            if match is None:
                raise ValueError('Invalid training target image')
            segment = match.group(1)
            if metadata[segment]['task'].startswith(task_type + ':'):
                counts[segment] += 1
    if (len(counts), sum(counts.values())) != EXPECTED[task_type]:
        raise ValueError(f'Unexpected current {task_type} data: {len(counts)}, {sum(counts.values())}')
    grouped = defaultdict(list)
    for segment, count in counts.items():
        row = dict(metadata[segment])
        row['training_rows'] = count
        grouped[re.sub(r'-\d+-\d+$', '', segment)].append(row)
    return sorted(grouped.items())


def check_source(job):
    base, segments = job
    root = SHARE / 'dataset/vpt'
    versions = [v for v in ('6.x', '7.x', '10.x') if (root / v / (base + '.mp4')).is_file()]
    if not versions:
        raise FileNotFoundError(base)
    path = root / versions[-1] / (base + '.jsonl')
    keys = {row['task']: 'minecraft.' + row['task'].split(':', 1)[0] + ':minecraft.' + row['task'].split(':', 1)[1] for row in segments}
    ranges = [(max(0, row['abs_start'] - 1), row['abs_end']) for row in segments]
    maximum = max(end for _, end in ranges)
    snapshots = {}
    last_frame = -1
    with path.open() as handle:
        for frame, line in enumerate(handle):
            last_frame = frame
            if any(start <= frame <= end for start, end in ranges):
                row = json.loads(line)
                stats = row['stats']
                values = {task: stats.get(key) for task, key in keys.items()}
                if any(value is not None and (type(value) is not int or value < 0) for value in values.values()):
                    raise ValueError(f'Invalid target stat: {path}:{frame + 1}')
                snapshots[frame] = {'targets': values, 'tick': row['tick'], 'gui': row['isGuiOpen'],
                                    'stats_key_count': len(stats),
                                    'trade': stats.get('minecraft.custom:minecraft.traded_with_villager')}
            if frame == maximum:
                break
    if last_frame < maximum:
        raise ValueError(f'JSONL shorter than segment: {path}')
    results = []
    for row in segments:
        start, end, task = row['abs_start'], row['abs_end'], row['task']
        increases, appearances, decreases = [], [], []
        for frame in range(start, end + 1):
            current, previous = snapshots[frame], snapshots.get(frame - 1)
            before = previous['targets'][task] if previous else None
            after = current['targets'][task]
            evidence = {'absolute_frame': frame, 'relative_frame': frame - start, 'jsonl_line': frame + 1,
                        'before': before, 'after': after, 'isGuiOpen': current['gui'],
                        'trade_before': previous['trade'] if previous else None, 'trade_after': current['trade'],
                        'stats_key_count_before': previous['stats_key_count'] if previous else None,
                        'stats_key_count_after': current['stats_key_count']}
            if before is not None and after is not None and after > before:
                increases.append({**evidence, 'increase': after - before})
            elif before is None and after is not None and after > 0:
                appearances.append(evidence)
            elif before is not None and (after is None or after < before):
                decreases.append(evidence)
        at_start = snapshots[start]['targets'][task]
        if increases:
            category = 'confirmed_increase'
        elif any(event['after'] > 0 and event['absolute_frame'] > start for event in appearances):
            category = 'first_positive_appearance_after_start'
        elif at_start is not None and at_start > 0:
            category = 'positive_at_start_no_increase'
        else:
            category = 'no_positive_value_or_increase'
        baseline = snapshots.get(start - 1)
        before_segment = baseline['targets'][task] if baseline else None
        at_end = snapshots[end]['targets'][task]
        results.append({
            'segment_id': row['video_id'], 'task': task, 'instruction': row['instruction'],
            'video': str(SHARE / 'dataset/mcvla_video_sft' / row['video']), 'source_jsonl': str(path),
            'abs_start': start, 'abs_end': end, 'training_rows': row['training_rows'],
            'target_stat_key': keys[task], 'category': category, 'category_label': LABELS[category],
            'target_before_segment': before_segment, 'target_at_first_frame': at_start, 'target_at_last_frame': at_end,
            'net_increase_from_before_segment': at_end - before_segment if before_segment is not None and at_end is not None else None,
            'confirmed_increase_sum': sum(event['increase'] for event in increases),
            'confirmed_increase_events': increases, 'first_positive_appearances': appearances,
            'decreases_or_disappearances': decreases,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-type', choices=tuple(EXPECTED), default='craft_item')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    task_alias = ALIASES[args.task_type]
    if args.output is None:
        args.output = Path('/apdcephfs/private_chengzhixu/JarvisVLA/outputs') / f'stage2_{task_alias}_target_stat_audit'
    if args.output.exists():
        raise FileExistsError(args.output)
    expected_segments, expected_rows = EXPECTED[args.task_type]
    jobs = load_jobs(args.task_type)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, batch in enumerate(pool.map(check_source, jobs), 1):
            rows.extend(batch)
            if index % 100 == 0:
                print(f'Checked {index}/{len(jobs)} source files; {len(rows)}/{expected_segments} segments', flush=True)
    rows.sort(key=lambda row: row['segment_id'])
    if len(rows) != expected_segments or sum(row['training_rows'] for row in rows) != expected_rows:
        raise ValueError('Incomplete audit')
    summary = {}
    for category, label in LABELS.items():
        matching = [row for row in rows if row['category'] == category]
        summary[category] = {'label': label, 'segments': len(matching),
                             'training_rows': sum(row['training_rows'] for row in matching),
                             'percent_of_segments': round(len(matching) / len(rows) * 100, 4)}
    anomalies = [row['segment_id'] for row in rows if row['decreases_or_disappearances']]
    report = {'dataset': str(DATASET), 'task_type': args.task_type, 'segments': len(rows), 'training_rows': sum(row['training_rows'] for row in rows),
              'method': 'Inspect each complete segment and the preceding frame when available. A confirmed target increase requires consecutive observed integer values with after > before. Missing values are not assumed to be zero. No video playback or acquisition-method certification.',
              'summary': summary, 'decrease_or_disappearance_segments': anomalies, 'samples': rows}
    args.output.mkdir(parents=True)
    (args.output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [f'当前处理后{task_alias}数据的目标统计增量检查', str(DATASET),
             f'范围：{expected_segments}个片段、{expected_rows}条训练样本。没有修改训练数据，没有播放视频验证获取方式。',
             '缺失字段不视为零；首次出现正值与可确认数值增加严格区分。',
             json.dumps(summary, ensure_ascii=False, indent=2), '']
    for row in sorted(rows, key=lambda row: (row['category'] == 'confirmed_increase', row['category'], row['segment_id'])):
        lines.extend(['=' * 80, row['segment_id'], row['category_label'], row['task'], row['instruction'],
                      f"训练样本：{row['training_rows']}；前帧值：{row['target_before_segment']}；首帧值：{row['target_at_first_frame']}；末帧值：{row['target_at_last_frame']}",
                      row['video'], row['source_jsonl'],
                      '明确增加：' + json.dumps(row['confirmed_increase_events'], ensure_ascii=False),
                      '首次出现：' + json.dumps(row['first_positive_appearances'], ensure_ascii=False), ''])
    (args.output / 'segments.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, ensure_ascii=False, indent=2), flush=True)
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()

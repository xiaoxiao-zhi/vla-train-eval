#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
DATASET = SHARE / 'dataset/duplex_vla_v4/stage2_train_flat3_noop10_qallm_drop147.jsonl'
VIDEO_SFT = SHARE / 'dataset/mcvla_video_sft'
VPT = SHARE / 'dataset/vpt'
TRADE = 'minecraft.custom:minecraft.traded_with_villager'
FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')
KNOWN_SEGMENT = 'gimpy-jade-panda-f8ec19661079-20220129-101927-4647-4843'


def load_jobs():
    metadata = {}
    with (VIDEO_SFT / 'train.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            if row['video_id'] in metadata:
                raise ValueError('Duplicate video metadata')
            metadata[row['video_id']] = {k: v for k, v in row.items() if k != 'actions'}
    counts = Counter()
    with DATASET.open() as source:
        for line in source:
            row = json.loads(line)
            if not row.get('images'):
                continue
            match = FRAME.fullmatch(Path(row['images'][-1]).name)
            if match is None:
                raise ValueError('Invalid training target image')
            segment = match.group(1)
            if metadata[segment]['task'].startswith('craft_item:'):
                counts[segment] += 1
    if len(counts) != 1232 or sum(counts.values()) != 104196:
        raise ValueError(f'Unexpected current craft composition: {len(counts)}, {sum(counts.values())}')
    grouped = defaultdict(list)
    for segment, count in counts.items():
        row = dict(metadata[segment])
        row['training_rows'] = count
        base = re.sub(r'-\d+-\d+$', '', segment)
        grouped[base].append(row)
    return sorted(grouped.items())


def snapshot(row, targets):
    stats = row['stats']
    trade = stats.get(TRADE)
    if trade is not None and (type(trade) is not int or trade < 0):
        raise ValueError(f'Unexpected trade count: {trade!r}')
    return {'trade': trade, 'gui': row['isGuiOpen'], 'tick': row['tick'],
            'targets': {target: stats.get('minecraft.craft_item:minecraft.' + target) for target in targets}}


def check_source(job):
    base, segments = job
    versions = [v for v in ('6.x', '7.x', '10.x') if (VPT / v / (base + '.mp4')).is_file()]
    if not versions:
        raise FileNotFoundError(base)
    path = VPT / versions[-1] / (base + '.jsonl')
    targets = {row['task'].split(':', 1)[1] for row in segments}
    ranges = [(max(0, row['abs_start'] - 1), row['abs_end']) for row in segments]
    maximum = max(end for _, end in ranges)
    snapshots = {}
    last_frame = -1
    with path.open() as source:
        for frame, line in enumerate(source):
            last_frame = frame
            if any(start <= frame <= end for start, end in ranges):
                snapshots[frame] = snapshot(json.loads(line), targets)
            if frame == maximum:
                break
    if last_frame < maximum:
        raise ValueError(f'JSONL shorter than segment: {path}')
    results = []
    for segment in segments:
        start, end = segment['abs_start'], segment['abs_end']
        target = segment['task'].split(':', 1)[1]
        events, first_observations, decreases = [], [], []
        simultaneous, target_first_observations = [], []
        for frame in range(start, end + 1):
            current = snapshots[frame]
            previous = snapshots.get(frame - 1)
            before = previous['trade'] if previous else None
            after = current['trade']
            if before is not None and after is not None and after > before:
                event = {'absolute_frame': frame, 'relative_frame': frame - start,
                         'jsonl_line': frame + 1, 'tick': current['tick'],
                         'before': before, 'after': after, 'increase': after - before,
                         'isGuiOpen': current['gui']}
                events.append(event)
            elif before is None and after is not None:
                first_observations.append({'absolute_frame': frame, 'value': after})
            elif before is not None and (after is None or after < before):
                decreases.append({'absolute_frame': frame, 'before': before, 'after': after})
            target_before = previous['targets'][target] if previous else None
            target_after = current['targets'][target]
            target_increase = target_before is not None and target_after is not None and target_after > target_before
            target_appearance = target_before is None and target_after is not None and target_after > 0
            if target_appearance:
                target_first_observations.append({'absolute_frame': frame, 'value': target_after})
            if before is not None and after is not None and after > before and (target_increase or target_appearance):
                simultaneous.append({'absolute_frame': frame, 'relative_frame': frame - start,
                                     'jsonl_line': frame + 1, 'trade_before': before, 'trade_after': after,
                                     'target_before': target_before, 'target_after': target_after,
                                     'target_first_appearance': target_appearance,
                                     'isGuiOpen': current['gui']})
        values = [snapshots[frame]['trade'] for frame in range(start, end + 1)]
        baseline = snapshots.get(start - 1)
        results.append({
            'segment_id': segment['video_id'], 'task': segment['task'], 'instruction': segment['instruction'],
            'video': str(VIDEO_SFT / segment['video']), 'source_jsonl': str(path),
            'abs_start': start, 'abs_end': end, 'training_rows': segment['training_rows'],
            'trade_key_present': any(v is not None for v in values),
            'positive_trade_count_observed': any(v is not None and v > 0 for v in values),
            'trade_before_segment': baseline['trade'] if baseline else None,
            'trade_at_first_frame': snapshots[start]['trade'], 'trade_at_last_frame': snapshots[end]['trade'],
            'confirmed_trade_in_segment': bool(events),
            'confirmed_trade_increase': sum(event['increase'] for event in events),
            'trade_events': events, 'trade_first_observations': first_observations,
            'trade_decreases_or_disappearances': decreases,
            'target_and_trade_same_frame': bool(simultaneous), 'simultaneous_events': simultaneous,
            'target_stat_first_observations': target_first_observations,
        })
    return results


def summarize(rows):
    groups = {
        'trade_key_present': lambda row: row['trade_key_present'],
        'positive_cumulative_trade_count': lambda row: row['positive_trade_count_observed'],
        'confirmed_trade_in_segment': lambda row: row['confirmed_trade_in_segment'],
        'positive_history_without_confirmed_segment_trade': lambda row: row['positive_trade_count_observed'] and not row['confirmed_trade_in_segment'],
        'target_stat_and_trade_changed_in_same_frame': lambda row: row['target_and_trade_same_frame'],
        'first_observation_without_confirmed_segment_trade': lambda row: bool(row['trade_first_observations']) and not row['confirmed_trade_in_segment'],
        'trade_decreases_or_disappearances': lambda row: bool(row['trade_decreases_or_disappearances']),
    }
    result = {'segments': len(rows), 'training_rows': sum(row['training_rows'] for row in rows)}
    for name, predicate in groups.items():
        matching = [row for row in rows if predicate(row)]
        result[name] = {'segments': len(matching), 'training_rows': sum(row['training_rows'] for row in matching)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/apdcephfs/private_chengzhixu/JarvisVLA/outputs/stage2_craft_villager_trade_audit'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--known-only', action='store_true')
    args = parser.parse_args()
    jobs = load_jobs()
    if args.known_only:
        jobs = [(base, [row for row in rows if row['video_id'] == KNOWN_SEGMENT])
                for base, rows in jobs if any(row['video_id'] == KNOWN_SEGMENT for row in rows)]
        if len(jobs) != 1:
            raise ValueError('Known segment not found')
        print(json.dumps(check_source(jobs[0]), ensure_ascii=False, indent=2), flush=True)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, batch in enumerate(pool.map(check_source, jobs), 1):
            rows.extend(batch)
            if index % 100 == 0:
                print(f'Checked {index}/{len(jobs)} source files; {len(rows)}/1232 segments', flush=True)
    rows.sort(key=lambda row: row['segment_id'])
    if len(rows) != 1232:
        raise ValueError('Incomplete segment coverage')
    summary = summarize(rows)
    summary['other_segments_excluding_known_11mp4'] = summarize([row for row in rows if row['segment_id'] != KNOWN_SEGMENT])
    report = {'dataset': str(DATASET), 'raw_stat_key': TRADE,
              'scope': 'All remaining craft segments, full original frame ranges, including the known 11.mp4; separate counts excluding it are provided.',
              'method': 'A confirmed trade requires two consecutive available integer counts with an increase within segment frames. Missing-to-present is reported separately, not silently treated as zero. Simultaneous target stat changes are evidence of a possible trading acquisition, not video verification.',
              'summary': summary, 'samples': rows}
    args.output.mkdir(parents=True)
    (args.output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['当前drop147数据集craft村民交易统计检查', str(DATASET),
             'traded_with_villager是累计次数，不是布尔值。历史非零与片段内实际增加分别统计；没有播放视频认证动作。',
             json.dumps(summary, ensure_ascii=False, indent=2), '', '片段内交易次数实际增加的片段：']
    for row in rows:
        if row['confirmed_trade_in_segment']:
            lines.extend(['=' * 80, row['segment_id'], row['task'], row['instruction'],
                          f"训练帧数：{row['training_rows']}；片段内交易增量：{row['confirmed_trade_increase']}；目标统计与交易同帧变化：{row['target_and_trade_same_frame']}",
                          row['video'], json.dumps(row['trade_events'], ensure_ascii=False),
                          '目标与交易同帧变化：' + json.dumps(row['simultaneous_events'], ensure_ascii=False)])
    (args.output / 'trade_segments.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()

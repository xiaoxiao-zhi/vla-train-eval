#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from summarize_vla_fit import parse


def read_records(path):
    with open(path) as f:
        records = [json.loads(line) for line in f]
    indexed = {record['idx']: record for record in records}
    if len(records) != len(indexed):
        raise ValueError('Duplicate sample indices: %s' % path)
    return indexed


def scores(metrics):
    result = {key: metrics[key] for key in ('action_exact_rate', 'keyboard_set_exact_rate')}
    camera = metrics['camera']
    gold_n = sum(axis['gold_n'] for axis in camera.values())
    valid_n = sum(axis['valid_prediction_n'] for axis in camera.values())
    error_n = sum(axis['error_n'] for axis in camera.values())
    result['camera_accuracy'] = sum(axis['accuracy'] * axis['gold_n'] for axis in camera.values()) / gold_n
    result['camera_mae'] = sum(axis['mae_valid_predictions'] * axis['valid_prediction_n'] for axis in camera.values() if axis['valid_prediction_n']) / valid_n if valid_n else None
    result['camera_error_mae'] = sum((axis['mae_errors_only'] or 0) * axis['error_n'] for axis in camera.values()) / error_n if error_n else None
    result['camera_error_ge5_fraction'] = sum((axis['error_ge5_fraction'] or 0) * axis['error_n'] for axis in camera.values()) / error_n if error_n else None
    result['camera_valid_prediction_n'] = valid_n
    for name, axis in camera.items():
        result['camera_%s_accuracy' % name] = axis['accuracy']
        result['camera_%s_mae' % name] = axis['mae_valid_predictions']
        rows = axis['by_gold_bin']
        zero = rows['0']
        result['camera_%s_zero_accuracy' % name] = zero['accuracy_given_gold']
        nonzero = [row for target, row in rows.items() if target != '0']
        count = sum(row['gold_n'] for row in nonzero)
        result['camera_%s_nonzero_accuracy' % name] = sum(row['hit_n'] for row in nonzero) / count if count else None
    return result


def compare(baseline, candidate, expected_n):
    original, weighted = read_records(baseline), read_records(candidate)
    if len(original) != expected_n or len(weighted) != expected_n or original.keys() != weighted.keys():
        raise ValueError('Models do not have the same complete sample index set')
    if any(original[idx]['gold'] != weighted[idx]['gold'] for idx in original):
        raise ValueError('Gold differs between model evaluations')
    baseline_metrics = json.loads(Path(str(baseline) + '.metrics.json').read_text())
    candidate_metrics = json.loads(Path(str(candidate) + '.metrics.json').read_text())
    for path, metrics in ((baseline, baseline_metrics), (candidate, candidate_metrics)):
        if metrics['n'] != expected_n or Path(metrics['input']).resolve() != Path(path).resolve():
            raise ValueError('Metrics do not match their sample results')
    baseline_scores, candidate_scores = scores(baseline_metrics), scores(candidate_metrics)
    deltas = {key: candidate_scores[key] - value if value is not None and candidate_scores[key] is not None else None
              for key, value in baseline_scores.items()}
    paired = {axis: {'both_valid_n': 0, 'candidate_closer_n': 0, 'equal_distance_n': 0,
                     'baseline_closer_n': 0, 'baseline_distance_sum': 0, 'candidate_distance_sum': 0}
              for axis in ('x', 'y')}
    action = {'both_correct_n': 0, 'baseline_only_correct_n': 0,
              'candidate_only_correct_n': 0, 'both_wrong_n': 0}
    for idx, original_record in original.items():
        weighted_record = weighted[idx]
        gold = parse(original_record['gold'])
        a = parse(original_record.get('output_unwrapped', original_record['output']))
        b = parse(weighted_record.get('output_unwrapped', weighted_record['output']))
        a_correct = a is not None and a['tokens'] == gold['tokens']
        b_correct = b is not None and b['tokens'] == gold['tokens']
        outcome = 'both_correct_n' if a_correct and b_correct else 'baseline_only_correct_n' if a_correct else 'candidate_only_correct_n' if b_correct else 'both_wrong_n'
        action[outcome] += 1
        for axis, counts in paired.items():
            if a is None or b is None or len(a[axis]) != 1 or len(b[axis]) != 1:
                continue
            a_distance = abs(a[axis][0] - gold[axis][0])
            b_distance = abs(b[axis][0] - gold[axis][0])
            counts['both_valid_n'] += 1
            counts['baseline_distance_sum'] += a_distance
            counts['candidate_distance_sum'] += b_distance
            outcome = 'candidate_closer_n' if b_distance < a_distance else 'baseline_closer_n' if a_distance < b_distance else 'equal_distance_n'
            counts[outcome] += 1
    for counts in paired.values():
        count = counts['both_valid_n']
        counts['baseline_mae_on_common_valid'] = counts['baseline_distance_sum'] / count if count else None
        counts['candidate_mae_on_common_valid'] = counts['candidate_distance_sum'] / count if count else None
    return {'baseline_results': str(baseline), 'candidate_results': str(candidate), 'n': expected_n,
            'sample_idx_gold_exact_match': True,
            'delta_policy': 'candidate minus baseline; higher accuracy and lower distance are better',
            'distance_policy': 'absolute bin difference; per-model valid predictions and paired common-valid predictions reported separately',
            'baseline': baseline_scores, 'candidate': candidate_scores, 'candidate_minus_baseline': deltas,
            'paired_action': action, 'paired_camera': paired}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--expected-n', type=int, default=10000)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.expected_n)
    with open(args.out, 'x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

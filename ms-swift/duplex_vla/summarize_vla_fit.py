#!/usr/bin/env python
import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

ACT = re.compile(r'<\|reserved_special_token_178\|>(.*?)<\|reserved_special_token_179\|>', re.S)
TOK = re.compile(r'<\|reserved_special_token_(\d+)\|>')


def parse(text):
    if text.startswith('['):
        try:
            value = ast.literal_eval(text)
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                text = value[0]
        except (ValueError, SyntaxError):
            pass
    match = ACT.search(text)
    if not match:
        return None
    tokens = [int(v) for v in TOK.findall(match.group(1))]
    return {'keyboard': {v for v in tokens if 180 <= v <= 208},
            'x': [v - 219 for v in tokens if 209 <= v <= 229],
            'y': [v - 240 for v in tokens if 230 <= v <= 250],
            'tokens': tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    total = invalid_output = keyboard_exact = action_exact = 0
    keyboard_gold, keyboard_hit, keyboard_fp = Counter(), Counter(), Counter()
    axes = {axis: {'gold': Counter(), 'hit': Counter(), 'missing': Counter(),
                   'errors': [], 'all_distances': [], 'confusion': Counter()}
            for axis in ('x', 'y')}
    with open(args.input, encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            gold = parse(record['gold'])
            output = parse(record.get('output_unwrapped', record['output']))
            if gold is None or any(len(gold[axis]) != 1 for axis in axes):
                raise ValueError('Invalid VLA gold at idx=%s' % record.get('idx'))
            total += 1
            if output is None:
                invalid_output += 1
            predicted_keyboard = output['keyboard'] if output else set()
            keyboard_exact += int(output is not None and gold['keyboard'] == predicted_keyboard)
            action_exact += int(output is not None and gold['tokens'] == output['tokens'])
            keyboard_gold.update(gold['keyboard'])
            keyboard_hit.update(gold['keyboard'] & predicted_keyboard)
            keyboard_fp.update(predicted_keyboard - gold['keyboard'])
            for axis, stats in axes.items():
                target = gold[axis][0]
                stats['gold'][target] += 1
                if output is None or len(output[axis]) != 1:
                    stats['missing'][target] += 1
                    continue
                predicted = output[axis][0]
                stats['confusion'][(target, predicted)] += 1
                stats['hit'][target] += int(target == predicted)
                distance = abs(predicted - target)
                stats['all_distances'].append(distance)
                if distance:
                    stats['errors'].append(distance)

    keyboard = {}
    for token in sorted(set(keyboard_gold) | set(keyboard_fp)):
        gold_count, hits, fp = keyboard_gold[token], keyboard_hit[token], keyboard_fp[token]
        keyboard[str(token)] = {'gold_n': gold_count, 'hit_n': hits, 'false_positive_n': fp,
                                'recall_given_gold': hits / gold_count if gold_count else None,
                                'false_positive_rate_given_absent': fp / (total - gold_count)
                                if total > gold_count else None}
    camera = {}
    for axis, stats in axes.items():
        distances, errors = stats['all_distances'], stats['errors']
        rows = {}
        for target in range(-10, 11):
            count = stats['gold'][target]
            rows[str(target)] = {'gold_n': count, 'hit_n': stats['hit'][target],
                                 'missing_prediction_n': stats['missing'][target],
                                 'accuracy_given_gold': stats['hit'][target] / count if count else None,
                                 'predicted_counts': {str(p): n for (g, p), n in
                                                      sorted(stats['confusion'].items()) if g == target}}
        denominator = sum(stats['gold'].values())
        camera[axis] = {'gold_n': denominator, 'valid_prediction_n': len(distances),
                        'accuracy': sum(stats['hit'].values()) / denominator if denominator else None,
                        'mae_valid_predictions': sum(distances) / len(distances) if distances else None,
                        'mae_errors_only': sum(errors) / len(errors) if errors else None,
                        'error_n': len(errors),
                        'error_ge5_fraction': sum(d >= 5 for d in errors) / len(errors) if errors else None,
                        'error_distance_counts': dict(sorted(Counter(errors).items())),
                        'by_gold_bin': rows}
    summary = {'input': args.input, 'n': total, 'invalid_action_output_n': invalid_output,
               'action_exact_rate': action_exact / total if total else None,
               'keyboard_set_exact_rate': keyboard_exact / total if total else None,
               'accuracy_policy': 'gold-conditioned; missing predictions count as incorrect',
               'distance_policy': 'absolute signed-bin difference; valid predictions only',
               'keyboard_tokens': keyboard, 'camera': camera}
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('n=%d action_exact=%s keyboard_exact=%s' %
          (total, summary['action_exact_rate'], summary['keyboard_set_exact_rate']), flush=True)
    for axis in camera:
        print('camera-%s: %s' % (axis, {k: v for k, v in camera[axis].items()
                                     if k not in ('by_gold_bin', 'error_distance_counts')}), flush=True)


if __name__ == '__main__':
    main()

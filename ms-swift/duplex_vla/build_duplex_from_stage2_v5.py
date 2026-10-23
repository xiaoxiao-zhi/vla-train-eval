#!/usr/bin/env python3
import argparse
from collections import Counter
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import random

import build_duplex_from_stage2_v3 as R

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
STAGE2 = SHARE / 'dataset/duplex_vla_v5/stage2.jsonl'
REFERENCE = SHARE / 'dataset/flat3_ablation/duplex_train_v3_500traj_filtered.jsonl'
OUTPUT = SHARE / 'dataset/duplex_vla_v5/duplex.jsonl'


def load_sources():
    titles, actions, qa = {}, {}, []
    counts, qa_sources = Counter(), Counter()
    digest = hashlib.sha256()
    with STAGE2.open('rb') as handle:
        for row_index, line in enumerate(handle):
            digest.update(line)
            row = json.loads(line)
            messages = row['messages']
            if [m['role'] for m in messages] != ['user', 'assistant']:
                raise ValueError('Unexpected stage2 messages')
            if not row.get('images'):
                counts['qa'] += 1
                datasource = row['datasource']
                qa_sources[datasource] += 1
                if datasource in R.LLM_SOURCES:
                    qa.append({'source_row_index': row_index, 'datasource': datasource,
                               'question': messages[0]['content'], 'answer': messages[1]['content']})
                elif datasource != 'alpaca':
                    raise ValueError('Unknown QA source')
                continue
            counts['vla'] += 1
            match = R.FRAME.fullmatch(Path(row['images'][-1]).name)
            if match is None:
                raise ValueError('Invalid stage2 target image')
            base, step = match.group(1), int(match.group(2))
            title = messages[0]['content'].split('\nobservation:')[0]
            if base in titles and titles[base] != title:
                raise ValueError('Conflicting trajectory instructions')
            if (base, step) in actions:
                raise ValueError('Duplicate training target')
            titles[base] = title
            actions[base, step] = messages[1]['content']
    if counts != {'vla': 297270, 'qa': 195370} or len(qa) != 145370 or len(titles) != 4706:
        raise ValueError(f'Unexpected v5 source composition: {counts}, {len(titles)}')
    source_stats = json.loads(Path(str(STAGE2) + '.stats.json').read_text())
    if digest.hexdigest() != source_stats['sha256']:
        raise ValueError('V5 source differs from published dataset')
    candidates, metadata = [], {}
    with Path(R.V4.VIDEO_SFT).open() as handle:
        for line in handle:
            row = json.loads(line)
            base = row['video_id']
            if base not in titles:
                continue
            if base in metadata:
                raise ValueError('Duplicate original video metadata')
            if row['instruction'] != titles[base] or len(row['actions']) != row['n_frames']:
                raise ValueError(f'Original trajectory mismatch: {base}')
            if row['abs_end'] - row['abs_start'] + 1 != row['n_frames'] or len(row['actions']) < 4:
                raise ValueError(f'Invalid original frame range: {base}')
            metadata[base] = {k: v for k, v in row.items() if k != 'actions'}
            candidates.append((base, row['instruction'], row['actions']))
    if set(metadata) != set(titles):
        raise ValueError('Missing complete video sources for v5 trajectories')
    originals = {base: acts for base, _, acts in candidates}
    for (base, step), action in actions.items():
        if step >= len(originals[base]) or originals[base][step].rstrip('\n') != action:
            raise ValueError(f'Stage2 action differs from original trajectory: {base}, {step}')
    return sorted(candidates), qa, metadata, digest.hexdigest(), dict(qa_sources)


def prepare_tasks(candidates, qa, tokenizer, seed):
    random.Random(seed).shuffle(candidates)
    random.Random(seed).shuffle(qa)
    used, cursor, skipped = set(), 0, Counter()
    tasks, manifest = [], []
    for base, instruction, actions in candidates:
        while cursor < len(qa):
            row = qa[cursor]
            cursor += 1
            question, answer = row['question'], row['answer']
            if not question.strip() or not answer.strip() or question in used:
                skipped['empty_or_duplicate_question'] += 1
                continue
            answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
            if not answer_ids or not R.answer_fits_original_insertion(base, actions, answer_ids):
                skipped['answer_cannot_be_completed'] += 1
                continue
            used.add(question)
            tasks.append((base, instruction, actions, (question, answer, answer_ids)))
            manifest.append({'base': base, 'frames': len(actions), 'qa_source_row_index': row['source_row_index'],
                             'qa_datasource': row['datasource'], 'question': question, 'answer': answer,
                             'answer_tokens': len(answer_ids)})
            break
        else:
            raise ValueError('Insufficient fitting unique LLM QA')
    return tasks, manifest, skipped


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-frames', type=int, default=256 * 500)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    OUTPUT = args.out
    if args.target_frames <= 0 or os.environ.get('PYTHONHASHSEED') != str(args.seed):
        raise ValueError('Set positive target and PYTHONHASHSEED equal to seed before launching Python')
    temporary = Path(str(OUTPUT) + '.building')
    stats_path = Path(str(OUTPUT) + '.stats.json')
    manifest_path = Path(str(OUTPUT) + '.trajectories.json')
    if any(path.exists() for path in (OUTPUT, temporary, stats_path, manifest_path)):
        raise FileExistsError('Duplex output or partial output already exists')
    candidates, qa, metadata, stage2_hash, qa_sources = load_sources()
    R.V4.TOK = R.V4.B.get_tok()
    tokenizer = R.V4.TOK
    tasks, manifest, skipped = prepare_tasks(candidates, qa, tokenizer, args.seed)
    with REFERENCE.open() as handle:
        reference_fields = set(json.loads(handle.readline()))
    smoke = R.process_task(tasks[0])
    if set(smoke[0]) != reference_fields:
        raise ValueError('Duplex schema differs from previous successful training data')
    for sample in smoke:
        if not Path(sample['images'][-1]).is_file():
            raise FileNotFoundError(sample['images'][-1])
    print(f'REAL_TRAJECTORY_SMOKE_OK frames={len(smoke)} QA_complete=True candidates={len(tasks)}', flush=True)
    if args.smoke_only:
        return
    selected, expected_keys, noop_lines = [], set(), []
    totals, retained, speaking, task_counts = Counter(), Counter(), Counter(), Counter()
    selected_answer_ids = {}
    next_index = {task[0]: index for index, task in enumerate(tasks)}
    crossing_reached = False
    image_checks = 0
    try:
        with temporary.open('x') as destination, Pool(args.workers, initializer=R.V4.init_worker) as pool:
            for index, samples in enumerate(pool.imap(R.process_task, tasks, chunksize=1)):
                eligible_noops = sum(R.is_noop_silence(sample) for sample in samples)
                regular = len(samples) - eligible_noops
                before_n = retained['frames'] + int(len(noop_lines) * 0.05)
                candidate_n = retained['frames'] + regular + int((len(noop_lines) + eligible_noops) * 0.05)
                accept = True
                if candidate_n >= args.target_frames:
                    crossing_reached = True
                    if selected and abs(before_n - args.target_frames) <= abs(candidate_n - args.target_frames):
                        accept = False
                if accept:
                    entry = dict(manifest[index])
                    entry['insert_step'] = samples[0]['qa_meta']['insert_step']
                    entry['task'] = metadata[entry['base']]['task']
                    entry['video'] = metadata[entry['base']]['video']
                    entry['abs_start'] = metadata[entry['base']]['abs_start']
                    entry['abs_end'] = metadata[entry['base']]['abs_end']
                    entry['raw_noop_silence_frames'] = eligible_noops
                    entry['non_noop_silence_frames'] = regular
                    selected.append(entry)
                    selected_answer_ids[entry['base']] = tasks[index][3][2]
                    task_kind = entry['task'].split(':', 1)[0]
                    task_counts[task_kind] += 1
                    for sample in samples:
                        if not Path(sample['images'][-1]).is_file():
                            raise FileNotFoundError(sample['images'][-1])
                        image_checks += 1
                        line = json.dumps(sample, ensure_ascii=False) + '\n'
                        totals['frames'] += 1
                        match = R.ACTION.match(sample['messages'][1]['content'])
                        silent = sample['messages'][1]['content'][match.end():].strip() == R.V4.SILENCE
                        totals['silence_frames' if silent else 'speaking_frames'] += 1
                        if not silent:
                            speaking[sample['base']] += 1
                        if R.is_noop_silence(sample):
                            noop_lines.append((line, (sample['base'], sample['t'])))
                        else:
                            destination.write(line)
                            expected_keys.add((sample['base'], sample['t']))
                            retained['frames'] += 1
                    if len(selected) % 100 == 0:
                        print(f'SELECTED {len(selected)} trajectories raw={totals["frames"]} estimated_final={candidate_n}', flush=True)
                if crossing_reached:
                    pool.terminate()
                    break
            if not crossing_reached:
                raise ValueError('Current complete trajectory pool cannot reach the requested final frame count')
            keep_n = int(len(noop_lines) * 0.05)
            keep_indices = set(random.Random(args.seed).sample(range(len(noop_lines)), keep_n))
            for index, (line, key) in enumerate(noop_lines):
                if index in keep_indices:
                    destination.write(line)
                    expected_keys.add(key)
                    retained['frames'] += 1
        if totals['frames'] != sum(entry['frames'] for entry in selected) or image_checks != totals['frames']:
            raise ValueError('Full trajectory construction or image checking mismatch')
        expected_answers = {entry['base']: (entry['question'], entry['answer']) for entry in selected}
        actual_keys, actual_noop = set(), 0
        actual_speaking, actual_classes = Counter(), Counter()
        recovered_ids = {entry['base']: [] for entry in selected}
        speaking_last_step = {}
        digest = hashlib.sha256()
        with temporary.open('rb') as handle:
            for line in handle:
                digest.update(line)
                sample = json.loads(line)
                R.validate_sample(sample)
                if set(sample) != reference_fields:
                    raise ValueError('Published sample schema mismatch')
                key = sample['base'], sample['t']
                if key in actual_keys:
                    raise ValueError('Duplicate final target frame')
                actual_keys.add(key)
                if (sample['qa_meta']['question'], sample['qa_meta']['answer']) != expected_answers[sample['base']]:
                    raise ValueError('Final paired QA metadata mismatch')
                original = tasks[next_index[sample['base']]][2][sample['t']]
                action_text = tokenizer.decode(R.V4.B.enc_action(original))
                target = sample['messages'][1]['content']
                if not target.startswith(action_text):
                    raise ValueError('Final action differs from original')
                text = target[len(action_text):]
                if text != R.V4.SILENCE:
                    if sample['t'] <= speaking_last_step.get(sample['base'], -1):
                        raise ValueError('Answer segments out of original time order')
                    speaking_last_step[sample['base']] = sample['t']
                    recovered_ids[sample['base']].extend(tokenizer(text, add_special_tokens=False)['input_ids'])
                    actual_speaking[sample['base']] += 1
                actual_noop += int(R.is_noop_silence(sample))
                actual_classes[metadata[sample['base']]['task'].split(':', 1)[0]] += 1
        if actual_keys != expected_keys or actual_speaking != speaking or actual_noop != keep_n:
            raise ValueError('Final filtering or answer-frame preservation mismatch')
        if recovered_ids != selected_answer_ids:
            raise ValueError('Final complete answer token reconstruction mismatch')
        if len(actual_keys) != retained['frames'] or abs(len(actual_keys) - args.target_frames) > max(entry['frames'] for entry in selected):
            raise ValueError('Final size differs excessively from whole-trajectory target')
        retained_by_base = Counter(base for base, _ in actual_keys)
        for entry in selected:
            entry['retained_frames'] = retained_by_base[entry['base']]
        stats = {
            'output': str(OUTPUT), 'stage2_source': str(STAGE2), 'stage2_sha256': stage2_hash,
            'full_trajectory_source': R.V4.VIDEO_SFT, 'format_reference': str(REFERENCE),
            'construction_reference': R.V4.__file__, 'validated_wrapper_reference': R.__file__,
            'seed': args.seed, 'target_frames': args.target_frames, 'actual_frames': retained['frames'],
            'difference_from_target': retained['frames'] - args.target_frames,
            'candidate_trajectories': len(tasks), 'selected_trajectories': len(selected),
            'trajectory_task_distribution': dict(task_counts), 'final_frame_task_distribution': dict(actual_classes),
            'raw': dict(totals), 'final_speaking_frames': sum(actual_speaking.values()),
            'noop_silence_before': len(noop_lines), 'noop_silence_kept': keep_n,
            'noop_silence_keep_rate': 0.05, 'qa_pairs': len(selected),
            'eligible_llm_qa_pool': len(qa), 'qa_pool_sources': qa_sources,
            'qa_pairing_exclusions': dict(skipped), 'sha256': digest.hexdigest(), 'bytes': temporary.stat().st_size,
            'format': 'Same previous successful v4 single flattened user/assistant pair: full text history, no historical actions, current plus two historical images, current original action plus answer fragment or silence.',
            'selection': 'Seeded random order of complete current-v5 trajectories; take a prefix closest to target final frames. Restore full original actions within existing truncated bounds before duplex construction; no tail extension or arbitrary frame cuts.',
            'qa_policy': 'One unique existing LLM-generated factoid QA per trajectory; original Alpaca excluded; no new QA generated; complete answer token sequence validated.',
            'verification': {'all_original_action_targets_preserved_before_noop_silence_filter': True,
                             'all_original_target_images_exist': True, 'complete_answer_tokens_all_pairs': True,
                             'all_speaking_frames_retained': True, 'full_text_history_unchanged': True,
                             'exact_global_noop_silence_5percent_selection': True,
                             'final_target_set_exact': True, 'previous_schema_identical': True},
        }
        temporary.rename(OUTPUT)
        with manifest_path.open('x') as handle:
            json.dump(selected, handle, ensure_ascii=False, indent=2)
        with stats_path.open('x') as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
        print('BUILD_COMPLETE', json.dumps(stats, ensure_ascii=False), flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()

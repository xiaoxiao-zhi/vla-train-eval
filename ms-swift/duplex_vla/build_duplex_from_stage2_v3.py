#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import build_duplex_v4 as V4

FRAME = re.compile(r'(.+)_f([0-9]+)\.jpg$')
ACTION = re.compile(r'<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>')
NOP = '<|reserved_special_token_178|><|reserved_special_token_219|><|reserved_special_token_240|><|reserved_special_token_179|>'
LLM_SOURCES = {'squad_v2', 'trivia_qa', 'natural_questions'}


def load_stage2(path):
    titles, qa, selected_actions = {}, [], {}
    counts, qa_sources = Counter(), Counter()
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for row_index, line in enumerate(file):
            digest.update(line)
            row = json.loads(line)
            messages = row['messages']
            if [message['role'] for message in messages] != ['user', 'assistant']:
                raise ValueError('Unexpected stage2 messages format')
            if not row.get('images'):
                counts['qa'] += 1
                source = row['datasource']
                qa_sources[source] += 1
                if source in LLM_SOURCES:
                    qa.append({'source_row_index': row_index, 'datasource': source,
                               'question': messages[0]['content'], 'answer': messages[1]['content']})
                elif source != 'alpaca':
                    raise ValueError('Unknown QA provenance: %s' % source)
                continue
            counts['vla'] += 1
            match = FRAME.fullmatch(Path(row['images'][-1]).name)
            if match is None:
                raise ValueError('Invalid stage2 frame filename')
            base, step = match.group(1), int(match.group(2))
            title = messages[0]['content'].split('\nobservation:')[0]
            if base in titles and titles[base] != title:
                raise ValueError('Multiple instructions for a trajectory')
            titles[base] = title
            if (base, step) in selected_actions:
                raise ValueError('Duplicate stage2 target frame')
            selected_actions[base, step] = messages[1]['content']
    if counts != {'vla': 300000, 'qa': 195370} or len(qa) != 145370:
        raise ValueError('Unexpected current stage2 data composition')
    return titles, qa, selected_actions, digest.hexdigest(), dict(qa_sources)


def select_trajectories(args, titles, selected_actions):
    with args.candidates.open() as file:
        wanted = {json.loads(line)['video_id'] for line in file}
    candidates = []
    seen = set()
    with args.original.open() as file:
        for line in file:
            row = json.loads(line)
            base = row['video_id']
            if base not in wanted:
                continue
            if base in seen:
                raise ValueError('Duplicate original trajectory: %s' % base)
            seen.add(base)
            if base not in titles or row['instruction'] != titles[base]:
                raise ValueError('Candidate not represented correctly in current stage2')
            if len(row['actions']) < 4:
                raise ValueError('Trajectory too short for original QA insertion rule')
            candidates.append((base, row['instruction'], row['actions']))
    if seen != wanted:
        raise ValueError('Candidate list has missing original trajectories')
    if not 0 < args.trajectories <= len(candidates):
        raise ValueError('Requested trajectory count exceeds original candidate pool')
    chosen = random.Random(args.seed).sample(sorted(candidates), args.trajectories)
    originals = {base: actions for base, _, actions in chosen}
    for (base, step), action in selected_actions.items():
        if base in originals and (step >= len(originals[base]) or originals[base][step].rstrip('\n') != action):
            raise ValueError('Current stage2 action differs from original source')
    return chosen, len(candidates)


def answer_fits_original_insertion(base, actions, answer_ids):
    lens = [len(V4.B.enc_action(action)) for action in actions]
    valid = []
    for start in range(3, len(actions)):
        if sum(lens[start:]) >= len(answer_ids):
            valid.append(start)
    if not valid:
        return False
    start = random.Random(hash(base) & 0xffffffff).choice(valid)
    position = 0
    for budget in lens[start:]:
        if position == len(answer_ids):
            break
        segment, _ = V4.safe_seg(answer_ids[position:], budget)
        position += len(segment)
    return position == len(answer_ids)


def validate_sample(sample):
    if set(sample) != {'base', 't', 'messages', 'images', 'qa_meta'}:
        raise ValueError('Unexpected v4 sample schema')
    messages = sample['messages']
    if [message['role'] for message in messages] != ['user', 'assistant']:
        raise ValueError('Invalid duplex messages')
    prompt = messages[0]['content']
    if not prompt.startswith(V4.OBS_USER) or not prompt.endswith('\nobservation: \n<image>'):
        raise ValueError('Invalid duplex prompt boundaries')
    if '<|reserved_special_token_' in prompt:
        raise ValueError('Historical actions are forbidden')
    expected_images = [os.path.join(V4.FRAMES_DIR, '%s_f%06d.jpg' % (sample['base'], step))
                       for step in range(max(0, sample['t'] - 2), sample['t'] + 1)]
    if sample['images'] != expected_images or prompt.count('<image>') != len(expected_images):
        raise ValueError('Invalid original image window')
    target = messages[1]['content']
    match = ACTION.match(target)
    if match is None or not target[match.end():]:
        raise ValueError('Action or text channel is missing')


def process_task(task):
    base, instruction, actions, (question, answer, answer_ids) = task
    samples = V4.process_traj(task)
    if len(samples) != len(actions):
        raise ValueError('Original trajectory frames were lost')
    recovered, turns = [], []
    for step, (sample, action) in enumerate(zip(samples, actions)):
        if sample['base'] != base or sample['t'] != step:
            raise ValueError('Full trajectory steps are not continuous')
        action_text = V4.TOK.decode(V4.B.enc_action(action))
        target = sample['messages'][1]['content']
        if not target.startswith(action_text):
            raise ValueError('Generated target action differs from original')
        text = target[len(action_text):]
        if text != V4.SILENCE:
            ids = V4.TOK(text, add_special_tokens=False)['input_ids']
            if len(ids) > len(V4.B.enc_action(action)):
                raise ValueError('Answer segment exceeds action token budget')
            recovered.extend(ids)
        sample['qa_meta']['question'] = question
        sample['qa_meta']['answer'] = answer
        insert = sample['qa_meta']['insert_step']
        user = instruction if step == 0 else question if step == insert else V4.SILENCE
        history = ''.join(V4.OBS_USER + old_user + ('\nobservation: \n<image>' if i >= step - 2 else '')
                          + V4.OBS_ASST + old_text for i, (old_user, old_text) in enumerate(turns))
        expected_prompt = history + V4.OBS_USER + user + '\nobservation: \n<image>'
        if sample['messages'][0]['content'] != expected_prompt:
            raise ValueError('Full history or QA insertion differs from trained v4 format')
        turns.append((user, text))
        validate_sample(sample)
    if recovered != answer_ids:
        raise ValueError('Paired answer was not fully emitted in original trajectory')
    return samples


def is_noop_silence(sample):
    target = sample['messages'][1]['content']
    action = ACTION.match(target)
    return action.group(0) == NOP and target[action.end():].strip() == V4.SILENCE


def build(args):
    if os.environ.get('PYTHONHASHSEED') != str(args.seed):
        raise ValueError('Set PYTHONHASHSEED=%d before Python starts' % args.seed)
    output = args.out
    temporary = Path(str(output) + '.building')
    stats_path = Path(str(output) + '.stats.json')
    manifest_path = Path(str(output) + '.trajectories.json')
    if any(path.exists() for path in (output, temporary, stats_path, manifest_path)):
        raise FileExistsError('Output or partial output already exists')
    titles, qa, selected_actions, source_sha, qa_sources = load_stage2(args.stage2)
    trajectories, candidate_n = select_trajectories(args, titles, selected_actions)
    tokenizer = V4.B.get_tok()
    V4.TOK = tokenizer
    random.Random(args.seed).shuffle(qa)
    used_questions, cursor, skipped = set(), 0, Counter()
    tasks, manifest = [], []
    for base, instruction, actions in trajectories:
        while cursor < len(qa):
            row = qa[cursor]
            cursor += 1
            question, answer = row['question'], row['answer']
            if not question.strip() or not answer.strip() or question in used_questions:
                skipped['empty_or_duplicate_question'] += 1
                continue
            answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
            if not answer_ids or not answer_fits_original_insertion(base, actions, answer_ids):
                skipped['answer_not_complete_with_original_insertion'] += 1
                continue
            used_questions.add(question)
            break
        else:
            raise ValueError('Insufficient unique current LLM QA that fit complete trajectories')
        tasks.append((base, instruction, actions, (question, answer, answer_ids)))
        manifest.append({'base': base, 'frames': len(actions), 'qa_source_row_index': row['source_row_index'],
                         'qa_datasource': row['datasource'], 'question': question, 'answer': answer,
                         'answer_tokens': len(answer_ids)})
    with args.format_reference.open() as file:
        reference_fields = set(json.loads(file.readline()))
    smoke = process_task(tasks[0])
    if set(smoke[0]) != reference_fields:
        raise ValueError('Schema differs from trained v4 reference')
    for sample in (smoke[0], smoke[-1]):
        for image in sample['images']:
            if not Path(image).is_file():
                raise FileNotFoundError(image)
    print('REAL_FULL_TRAJECTORY_SMOKE_OK frames=%d QA_complete=True' % len(smoke), flush=True)
    print('SOURCE_READY trajectories=%d original_frames=%d LLM_QA_pool=%d' %
          (len(tasks), sum(len(task[2]) for task in tasks), len(qa)), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    totals, retained = Counter(), Counter()
    noop_lines, expected_keys = [], set()
    expected_answers = {entry['base']: (entry['question'], entry['answer']) for entry in manifest}
    speaking_counts = Counter()
    with temporary.open('x') as file, Pool(args.workers, initializer=V4.init_worker) as pool:
        for index, samples in enumerate(pool.imap(process_task, tasks, chunksize=2)):
            manifest[index]['insert_step'] = samples[0]['qa_meta']['insert_step']
            for sample in samples:
                line = json.dumps(sample, ensure_ascii=False) + '\n'
                totals['frames'] += 1
                action = ACTION.match(sample['messages'][1]['content'])
                silent = sample['messages'][1]['content'][action.end():].strip() == V4.SILENCE
                totals['silence_frames' if silent else 'speaking_frames'] += 1
                if not silent:
                    speaking_counts[sample['base']] += 1
                if is_noop_silence(sample):
                    noop_lines.append((line, (sample['base'], sample['t'])))
                    continue
                file.write(line)
                retained['frames'] += 1
                retained['image_references'] += len(sample['images'])
                expected_keys.add((sample['base'], sample['t']))
            if (index + 1) % 100 == 0:
                print('BUILD %d/%d trajectories raw_frames=%d' % (index + 1, len(tasks), totals['frames']), flush=True)
        keep_n = int(len(noop_lines) * 0.05)
        keep_indices = set(random.Random(args.seed).sample(range(len(noop_lines)), keep_n))
        for index, (line, key) in enumerate(noop_lines):
            if index in keep_indices:
                file.write(line)
                expected_keys.add(key)
                retained['frames'] += 1
                retained['image_references'] += min(key[1] + 1, 3)
    if totals['frames'] != sum(entry['frames'] for entry in manifest):
        raise ValueError('Raw constructed frames differ from full trajectories')
    if any(speaking_counts[entry['base']] == 0 for entry in manifest):
        raise ValueError('A trajectory has no supervised answer segments')
    actual_keys, actual_speaking, actual_noop = set(), Counter(), 0
    digest = hashlib.sha256()
    with temporary.open('rb') as file:
        for line in file:
            digest.update(line)
            sample = json.loads(line)
            validate_sample(sample)
            key = sample['base'], sample['t']
            if key in actual_keys:
                raise ValueError('Duplicate filtered target')
            actual_keys.add(key)
            meta = sample['qa_meta']
            if (meta['question'], meta['answer']) != expected_answers[sample['base']]:
                raise ValueError('Wrong QA metadata')
            action = ACTION.match(sample['messages'][1]['content'])
            if sample['messages'][1]['content'][action.end():].strip() != V4.SILENCE:
                actual_speaking[sample['base']] += 1
            actual_noop += int(is_noop_silence(sample))
    if actual_keys != expected_keys or actual_speaking != speaking_counts or actual_noop != keep_n:
        raise ValueError('Filtering or speaking-frame preservation failed')
    stats = {'stage2_qa_source': str(args.stage2), 'stage2_sha256': source_sha,
             'full_trajectory_source': str(args.original), 'trajectory_candidate_source': str(args.candidates),
             'format_reference': str(args.format_reference), 'construction_reference': V4.__file__,
             'seed': args.seed, 'candidate_trajectories': candidate_n, 'selected_trajectories': len(tasks),
             'qa_pairs': len(tasks), 'qa_pool_sources': qa_sources, 'eligible_llm_qa_pool': len(qa),
             'qa_selection_exclusions': dict(skipped), 'raw': dict(totals), 'filtered': dict(retained),
             'noop_silence_before': len(noop_lines), 'noop_silence_kept': keep_n,
             'noop_silence_dropped': len(noop_lines) - keep_n, 'noop_silence_keep_rate': 0.05,
             'bytes': temporary.stat().st_size, 'sha256': digest.hexdigest(),
             'format': 'Original v4 single flattened user/assistant pair, image history=2, historical actions=0; target action + answer segment or silence.',
             'selection': '%d complete trajectories sampled from original duplex candidate pool; no frame subsampling or tail extension. Only current LLM-generated factoid QA; original Alpaca excluded.' % len(tasks),
             'filter': 'Same global seed-42 sample of floor(5% * noop-and-silence targets); all other targets kept; full original text history unchanged.',
             'qa_meta_correction': 'Full paired answer preserved, correcting original builder variable shadowing in metadata only.',
             'verification': {'full_trajectories_built': True, 'complete_answer_tokens_all_pairs': True,
                              'all_speaking_frames_retained': True, 'original_prompt_history_unchanged': True,
                              'filtered_target_set_exact': True, 'reference_schema_identical': True,
                              'input_files_unchanged': True}}
    temporary.rename(output)
    with manifest_path.open('x') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    with stats_path.open('x') as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
    print('BUILD_COMPLETE', json.dumps(stats, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2', type=Path, required=True)
    parser.add_argument('--original', type=Path, default=Path(V4.VIDEO_SFT))
    parser.add_argument('--candidates', type=Path, default=Path(V4.DUP20W))
    parser.add_argument('--format-reference', type=Path, default=Path(V4.OUT_DIR) / 'duplex_train_filtered.jsonl')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trajectories', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    build(args)


if __name__ == '__main__':
    main()

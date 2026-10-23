#!/usr/bin/env python
import argparse
import hashlib
import json
import random
import re
import string
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests

ROOT = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset')
BLOCKED = re.compile(r'president|prime minister|politic|government|election|parliament|senat|congress|communist|democra|republican|dictator|war\b|military|terror|weapon|nazi|hitler|stalin|mao\b|xi jinping|putin|trump|biden|obama|clinton|churchill|mussolini|imperial|colonial|racis|sex\b|porn|nude|erotic|习近平|政治|政府|战争', re.I)


def canonical(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(ch for ch in text if ch.isalnum())


def fingerprint(text):
    return hashlib.sha256(canonical(text).encode()).hexdigest()


def question_part(text):
    return text.rsplit('\nQuestion:', 1)[-1].strip()


def context_part(text):
    if text.startswith('Context:') and '\nQuestion:' in text:
        return text[len('Context:'):].rsplit('\nQuestion:', 1)[0].strip()
    return None


def normalization(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def token_f1(prediction, answer):
    a, b = normalization(prediction).split(), normalization(answer).split()
    if not a or not b:
        return float(a == b)
    common = sum((Counter(a) & Counter(b)).values())
    return 2 * common / (len(a) + len(b))


def prepare(out, seed, selected_sources, n):
    audit_path = out / 'sample_audit.json'
    samples_path = out / 'samples.jsonl'
    if audit_path.exists() or samples_path.exists():
        raise FileExistsError('Holdout samples already exist; refusing to overwrite')
    sources = {
        'trivia_qa': ROOT / 'trivia_qa/rc.nocontext/validation-00000-of-00001.parquet',
        'squad_v2': ROOT / 'squad_v2/squad_v2/validation-00000-of-00001.parquet',
    }
    if n < 1 or n % len(selected_sources):
        raise ValueError('Sample count must be positive and divisible by source count')
    per_source = n // len(selected_sources)
    candidates = {}
    for name in selected_sources:
        path = sources[name]
        pool = []
        columns = ['question_id', 'question', 'answer'] if name == 'trivia_qa' else ['id', 'question', 'context', 'answers']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                q = row['question'].strip()
                if name == 'trivia_qa':
                    answer = row['answer'] or {}
                    gold = answer.get('value') or ''
                    aliases = [gold] + (answer.get('aliases') or []) + (answer.get('normalized_aliases') or [])
                    prompt, context, source_id = q, None, row['question_id']
                else:
                    aliases = (row['answers'] or {}).get('text') or []
                    gold = aliases[0] if aliases else ''
                    context = row['context'].strip()
                    prompt = 'Context:%s\nQuestion:%s' % (context, q)
                    source_id = row['id']
                aliases = list(dict.fromkeys(a.strip() for a in aliases if a and a.strip() and a.strip() != '<unk>'))
                if not q or not gold or gold == '<unk>' or not aliases or BLOCKED.search(prompt + '\n' + '\n'.join(aliases)):
                    continue
                pool.append({'source': name, 'datasource': name, 'source_split': 'validation', 'source_file': str(path),
                             'source_id': source_id, 'question': q, 'prompt': prompt, 'gold': gold,
                             'aliases': aliases, 'context_hash': fingerprint(context) if context else None,
                             'messages': [{'role': 'user', 'content': prompt},
                                          {'role': 'assistant', 'content': gold}]})
        candidates[name] = pool
    candidate_questions = {fingerprint(r['question']) for pool in candidates.values() for r in pool}
    candidate_prompts = {fingerprint(r['prompt']) for pool in candidates.values() for r in pool}
    candidate_contexts = {r['context_hash'] for pool in candidates.values() for r in pool if r['context_hash']}
    blocked_questions, blocked_prompts, blocked_contexts = set(), set(), set()
    train_paths = [ROOT / 'general_qa/train.jsonl', ROOT / 'general_qa_300k/train.jsonl',
                   ROOT / 'duplex_vla_v4/stage2_train_flat3.jsonl',
                   ROOT / 'duplex_vla_v4/stage2_train_flat3_v3.jsonl']
    scans = []
    for path in train_paths:
        count = 0
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row.get('images'):
                    continue
                for message in row['messages']:
                    if message['role'] != 'user':
                        continue
                    text = message['content']
                    count += 1
                    question_hash = fingerprint(question_part(text))
                    prompt_hash = fingerprint(text)
                    if question_hash in candidate_questions:
                        blocked_questions.add(question_hash)
                    if prompt_hash in candidate_prompts:
                        blocked_prompts.add(prompt_hash)
                    context = context_part(text)
                    if context:
                        context_hash = fingerprint(context)
                        if context_hash in candidate_contexts:
                            blocked_contexts.add(context_hash)
        scans.append({'path': str(path), 'qa_user_messages_checked': count})
        print('TRAIN_CHECK', path, count, flush=True)
    rng = random.Random(seed)
    selected, seen, pools = [], set(), {}
    for source, rows in candidates.items():
        eligible = []
        for row in rows:
            question_hash = fingerprint(row['question'])
            if question_hash in seen or question_hash in blocked_questions or fingerprint(row['prompt']) in blocked_prompts or row['context_hash'] in blocked_contexts:
                continue
            seen.add(question_hash)
            eligible.append(row)
        if len(eligible) < per_source:
            raise ValueError('Not enough clean heldout samples for %s: %d' % (source, len(eligible)))
        pools[source] = {'eligible_before_training_exclusion': len(rows), 'eligible_after_training_exclusion': len(eligible), 'selected': per_source}
        selected.extend(rng.sample(eligible, per_source))
    rng.shuffle(selected)
    for index, row in enumerate(selected):
        row['idx'] = index
    assert len(selected) == n
    assert not any(fingerprint(row['question']) in blocked_questions or fingerprint(row['prompt']) in blocked_prompts or row['context_hash'] in blocked_contexts for row in selected)
    serialized = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected)
    with open(samples_path, 'x') as f:
        f.write(serialized)
    audit = {'n': n, 'seed': seed, 'source_split': 'validation',
             'split_note': 'Held-out validation from QA training sources; not an official test split or an independently synthesized test split.',
             'excluded_test_sources': {'trivia_qa': 'official test answers are hidden (<unk>)', 'web_question': 'official test was included in general_qa training'},
             'sources': pools, 'training_checks': scans, 'selected_question_or_prompt_overlap_n': 0,
             'selected_squad_context_overlap_n': 0,
             'dedup_policy': 'NFKC, casefold, alphanumeric-only hashes; full user prompt, extracted question and SQuAD context checked',
             'limitation': 'Does not certify absence from foundation-model pretraining or semantic paraphrases.',
             'samples_sha256': hashlib.sha256(serialized.encode()).hexdigest()}
    with open(audit_path, 'x') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print('SAMPLES_READY', json.dumps(audit, ensure_ascii=False), flush=True)


def evaluate(out, url, expected_model):
    result_path = out / 'results.jsonl'
    summary_path = out / 'summary.json'
    if result_path.exists() or summary_path.exists():
        raise FileExistsError('Existing QA results will not be overwritten')
    samples_path = out / 'samples.jsonl'
    audit = json.loads((out / 'sample_audit.json').read_text())
    if hashlib.sha256(samples_path.read_bytes()).hexdigest() != audit['samples_sha256']:
        raise ValueError('Holdout samples have changed')
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
    if len(samples) != audit['n']:
        raise ValueError('Sample count differs from audit')
    for row in samples:
        if row['messages'] != [{'role': 'user', 'content': row['prompt']},
                               {'role': 'assistant', 'content': row['gold']}]:
            raise ValueError('Unexpected single-turn QA sample format')
        if row['source'] == 'trivia_qa' and row['prompt'] != row['question']:
            raise ValueError('TriviaQA input must contain only the original question')
    session = requests.Session()
    session.trust_env = False
    response = session.get(url + '/models', timeout=10)
    response.raise_for_status()
    models = [model['id'] for model in response.json()['data']]
    if expected_model not in models:
        raise ValueError('Unexpected vLLM models: %s' % models)
    params = {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'max_tokens': 256, 'skip_special_tokens': False}

    def ask(row):
        with requests.Session() as client:
            client.trust_env = False
            response = client.post(url + '/chat/completions',
                                   json={'model': expected_model, 'messages': row['messages'][:-1], **params}, timeout=300)
            response.raise_for_status()
            content = response.json()
        choice = content['choices'][0]
        prediction = choice['message']['content']
        if not isinstance(prediction, str):
            raise ValueError('Invalid QA response: %s' % content)
        normalized = normalization(prediction)
        answers = [normalization(answer) for answer in row['aliases']]
        exact = normalized in answers
        contains = any(answer and (' ' + answer + ' ') in (' ' + normalized + ' ') for answer in answers)
        return {**row, 'model': expected_model, 'prediction': prediction,
                'exact_match': exact, 'answer_contains': contains,
                'token_f1': max(token_f1(prediction, answer) for answer in row['aliases']),
                'finish_reason': choice.get('finish_reason')}

    started = time.time()
    first_result = ask(samples[0])
    print('QA_SMOKE_OK idx=%s' % first_result['idx'], flush=True)
    results = [first_result]
    with open(result_path, 'x') as f, ThreadPoolExecutor(max_workers=8) as executor:
        f.write(json.dumps(first_result, ensure_ascii=False) + '\n')
        f.flush()
        futures = [executor.submit(ask, row) for row in samples[1:]]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            if len(results) % 20 == 0:
                print('QA_PROGRESS %d/%d' % (len(results), len(samples)), flush=True)
    if len(results) != len(samples):
        raise ValueError('Incomplete QA evaluation')
    summary = {'n': len(results), 'model': expected_model, 'base_url': url, 'source_split': 'validation',
               'sample_audit': str(out / 'sample_audit.json'), 'samples_sha256': audit['samples_sha256'],
               'generation': params, 'seconds': round(time.time() - started, 1)}
    for name, rows in [('overall', results)] + [(source, [row for row in results if row['source'] == source]) for source in sorted({row['source'] for row in results})]:
        summary[name] = {'n': len(rows), 'exact_match': sum(row['exact_match'] for row in rows) / len(rows),
                         'answer_contains': sum(row['answer_contains'] for row in rows) / len(rows),
                         'token_f1': sum(row['token_f1'] for row in rows) / len(rows),
                         'empty_output_n': sum(not row['prediction'].strip() for row in rows),
                         'truncated_n': sum(row['finish_reason'] == 'length' for row in rows)}
    with open(summary_path, 'x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('QA_COMPLETE', json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'evaluate', 'all'], default='all')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--sources', nargs='+', choices=['trivia_qa', 'squad_v2'], default=['trivia_qa', 'squad_v2'])
    parser.add_argument('--base-url', default='http://127.0.0.1:8101/v1')
    parser.add_argument('--model')
    args = parser.parse_args()
    if args.mode != 'prepare' and not args.model:
        parser.error('--model is required for evaluation')
    if len(set(args.sources)) != len(args.sources):
        parser.error('--sources must not contain duplicates')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode in ('prepare', 'all'):
        prepare(out, args.seed, args.sources, args.n)
    if args.mode in ('evaluate', 'all'):
        evaluate(out, args.base_url, args.model)


if __name__ == '__main__':
    main()

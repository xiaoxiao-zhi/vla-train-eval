#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM-as-judge: 判定采样答案与参考答案(真值)是否事实一致.

贪心解码(temperature=0)保证确定性判定; 输出YES/NO.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

TLS = threading.local()
MODEL_NAME = '/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/Qwen2.5-7B-Instruct'

SYS = 'You are a strict fact checker.'
TEMPLATE = (
    'Question: {q}\n'
    'Reference (correct) answer: {ref}\n'
    'Candidate answer: {cand}\n\n'
    'Does the candidate answer correctly answer the question with the same fact '
    'as the reference answer? Synonyms and rephrasings of the same fact count as YES '
    '(e.g. "Holland" vs "Netherlands"). Reply NO if the candidate gives a different fact, '
    'refuses to answer, hedges without stating the answer, or rambles off-topic.\n'
    'Reply with exactly one word: YES or NO.'
)


def get_session():
    if not hasattr(TLS, 's'):
        TLS.s = requests.Session()
        TLS.s.proxies = {'http': None, 'https': None}
    return TLS.s


def one_judge(session, url, q, ref, cand, retries=3):
    payload = {
        'model': MODEL_NAME,
        'messages': [
            {'role': 'system', 'content': SYS},
            {'role': 'user', 'content': TEMPLATE.format(q=q, ref=ref, cand=cand)},
        ],
        'max_tokens': 5, 'temperature': 0,
    }
    last = None
    for att in range(retries):
        try:
            r = session.post(url + '/chat/completions', json=payload, timeout=300)
            if r.status_code != 200:
                raise RuntimeError('HTTP %d: %s' % (r.status_code, r.text[:200]))
            out = r.json()['choices'][0]['message']['content'].strip().upper()
            return 'YES' if out.startswith('YES') else 'NO'
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (att + 1))
    raise RuntimeError('重试%d次仍失败: %r' % (retries, last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-file', required=True, help='含 q/orig_a/new_a 的jsonl')
    ap.add_argument('--url', default='http://localhost:8100/v1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--concurrency', type=int, default=64)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.in_file)]
    print('判定 %d 条' % len(recs), flush=True)
    with open(args.out, 'w') as fo, \
            tqdm(total=len(recs), dynamic_ncols=True) as pbar, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(one_judge, get_session(), args.url,
                          r['q'], r['orig_a'], r['new_a']): r for r in recs}
        for fut in as_completed(futs):
            r = futs[fut]
            r['judge'] = fut.result()
            fo.write(json.dumps(r, ensure_ascii=False) + '\n')
            pbar.update(1)
    yes = sum(1 for l in open(args.out) if json.loads(l)['judge'] == 'YES')
    print('判定YES: %d/%d (%.0f%%)' % (yes, len(recs), yes * 100.0 / len(recs)), flush=True)


if __name__ == '__main__':
    main()

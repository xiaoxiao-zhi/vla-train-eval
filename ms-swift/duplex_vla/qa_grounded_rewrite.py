#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""factoid QA 接地改写: 原答案进prompt, 只改风格为LLM自然回答, 事实保真."""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

TLS = threading.local()
MODEL_NAME = '/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/Qwen2.5-7B-Instruct'

SYS = 'You are a knowledgeable assistant. Answer directly and naturally.'
TEMPLATE = (
    'Answer the question below in one natural sentence (or short phrase), as an assistant would. '
    'The known correct answer is provided — your response must state exactly this fact, '
    'phrased naturally. Do not mention that an answer was provided. '
    'Do not add any other facts.\n\nQuestion: {q}\nKnown correct answer: {a}'
)


def get_session():
    if not hasattr(TLS, 's'):
        TLS.s = requests.Session()
        TLS.s.proxies = {'http': None, 'https': None}
    return TLS.s


def one_req(session, url, q, a, retries=3):
    payload = {
        'model': MODEL_NAME,
        'messages': [
            {'role': 'system', 'content': SYS},
            {'role': 'user', 'content': TEMPLATE.format(q=q, a=a)},
        ],
        'max_tokens': 160, 'temperature': 0.7, 'top_p': 0.8,
        'top_k': 20, 'repetition_penalty': 1.05,
    }
    last = None
    for att in range(retries):
        try:
            r = session.post(url + '/chat/completions', json=payload, timeout=300)
            if r.status_code != 200:
                raise RuntimeError('HTTP %d: %s' % (r.status_code, r.text[:200]))
            return r.json()['choices'][0]['message']['content']
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (att + 1))
    raise RuntimeError('重试%d次仍失败: %r' % (retries, last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-file', required=True, help='qa_resample_test100.jsonl')
    ap.add_argument('--url', default='http://localhost:8100/v1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--concurrency', type=int, default=64)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.test_file)]
    recs = [r for r in recs if len(r['orig_a']) < 60]  # factoid
    print('factoid 样例: %d 条' % len(recs), flush=True)

    with open(args.out, 'w') as fo, \
            tqdm(total=len(recs), dynamic_ncols=True) as pbar, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(one_req, get_session(), args.url, r['q'], r['orig_a']): r
                for r in recs}
        for fut in as_completed(futs):
            r = futs[fut]
            r['rewritten'] = fut.result()
            fo.write(json.dumps(r, ensure_ascii=False) + '\n')
            pbar.update(1)
    print('完成', flush=True)


if __name__ == '__main__':
    main()

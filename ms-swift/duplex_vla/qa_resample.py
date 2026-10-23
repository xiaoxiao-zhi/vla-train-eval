#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QA答案重采样: 用Qwen2.5-7B-Instruct给纯文本QA重新生成答案.

从 stage2_train_flat3.jsonl 抽纯文本QA, 固定seed抽样, 每题采样1个答案,
官方推荐参数 (temp=0.7, top_p=0.8, top_k=20, rep=1.05) 显式传参.
并发: ThreadPoolExecutor (线程发HTTP请求), 输出含原答案便于对比.
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

TLS = threading.local()
MODEL_NAME = '/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/Qwen2.5-7B-Instruct'


def get_session():
    if not hasattr(TLS, 's'):
        TLS.s = requests.Session()
        TLS.s.proxies = {'http': None, 'https': None}
    return TLS.s


def one_req(session, url, q, retries=3):
    payload = {
        'model': MODEL_NAME,
        'messages': [{'role': 'user', 'content': q}],
        'max_tokens': 640, 'temperature': 0.7, 'top_p': 0.8,
        'top_k': 20, 'repetition_penalty': 1.05,
    }
    last = None
    for att in range(retries):
        try:
            r = session.post(url + '/chat/completions', json=payload, timeout=300)
            if r.status_code != 200:
                raise RuntimeError('HTTP %d: %s' % (r.status_code, r.text[:200]))
            d = r.json()
            return d['choices'][0]['message']['content']
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (att + 1))
    raise RuntimeError('重试%d次仍失败: %r' % (retries, last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--url', default='http://localhost:8100/v1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--concurrency', type=int, default=64)
    ap.add_argument('--start-idx', type=int, default=0, help='续跑起始行号')
    ap.add_argument('--factoid-only', action='store_true',
                    help='只采非alpaca的factoid QA (答案唯一的短答案型)')
    ap.add_argument('--resume', action='store_true', help='跳过输出文件中已完成的idx续跑')
    args = ap.parse_args()

    done_idx = set()
    if args.resume:
        try:
            for line in open(args.out):
                try:
                    done_idx.add(json.loads(line)['idx'])
                except json.JSONDecodeError:
                    continue
            print('续跑: 已完成 %d 条' % len(done_idx), flush=True)
        except FileNotFoundError:
            pass

    # 蓄水池抽n条QA (固定seed); n<=0且factoid-only时全量
    rng = random.Random(args.seed)
    reservoir = []
    total = 0
    for line in open(args.data):
        d = json.loads(line)
        if d.get('images'):
            continue
        q = d['messages'][0]['content']
        if args.factoid_only and ('Below is an instruction' in q or '### Instruction' in q):
            continue
        total += 1
        if total - 1 in done_idx:
            continue
        rec = {'idx': total - 1, 'q': q,
               'orig_a': d['messages'][1]['content']}
        if args.n <= 0:
            reservoir.append(rec)
        elif len(reservoir) < args.n:
            reservoir.append(rec)
        else:
            j = rng.randrange(total)
            if j < args.n:
                reservoir[j] = rec
    print('factoid总数: %d, 待采样: %d (seed=%d)' % (total, len(reservoir), args.seed), flush=True)

    t0 = time.time()
    done = 0
    with open(args.out, 'a' if args.resume else 'w') as fo, \
            tqdm(total=len(reservoir), dynamic_ncols=True) as pbar, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(one_req, get_session(), args.url, rec['q']): rec
                for rec in reservoir}
        for fut in as_completed(futs):
            rec = futs[fut]
            new_a = fut.result()
            rec['new_a'] = new_a
            fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
            done += 1
            pbar.update(1)
    print('完成 %d 条, 耗时 %.1fs' % (done, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()

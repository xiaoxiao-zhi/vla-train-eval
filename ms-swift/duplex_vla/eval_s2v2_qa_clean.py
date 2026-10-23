#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s2v2 general QA 能力评测 (原始数据集 validation split, 未见过).

题源: squad_v2 validation + trivia_qa rc.nocontext validation (训练只用了
这两个数据集的 train split), 各抽 50 道, 发 8100, 参考答案任一匹配即对.
"""
import glob
import json
import random
import re

import pandas as pd
import requests

SQUAD_VAL = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
             'chengzhixu/dataset/squad/plain_text/validation-*.parquet')
TRIVIA_VAL = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
              'chengzhixu/dataset/trivia_qa/rc.nocontext/validation-*.parquet')
URL = 'http://localhost:8100/v1/chat/completions'
N_PER = 50
OUT = '/apdcephfs/private_chengzhixu/s2v2_qa_eval_clean.json'


def norm(s):
    return re.sub(r'[^a-z0-9 ]', '', str(s).lower()).strip()


def main():
    random.seed(7)
    items = []  # (src, question, [参考答案...])

    df = pd.read_parquet(glob.glob(SQUAD_VAL)[0])
    idx = random.sample(range(len(df)), N_PER)
    for i in idx:
        r = df.iloc[i]
        refs = [a for a in list(r['answers']['text']) if str(a).strip()]
        if refs:
            items.append(('squad_v2', r['question'], refs))

    df = pd.read_parquet(glob.glob(TRIVIA_VAL)[0])
    idx = random.sample(range(len(df)), N_PER)
    for i in idx:
        r = df.iloc[i]
        ans = r['answer']
        refs = [ans['value']] + [a for a in list(ans.get('aliases', []))][:8]
        refs = [a for a in refs if str(a).strip()]
        if refs:
            items.append(('trivia_qa', r['question'], refs))

    print('题数: %d (squad %d, trivia %d)' % (
        len(items), sum(1 for x in items if x[0] == 'squad_v2'),
        sum(1 for x in items if x[0] == 'trivia_qa')), flush=True)

    results, correct = [], 0
    by_src = {'squad_v2': [0, 0], 'trivia_qa': [0, 0]}
    for i, (src, q, refs) in enumerate(items):
        try:
            r = requests.post(URL, json={
                'model': 'jarvisvla_s2v2',
                'messages': [{'role': 'user', 'content': q}],
                'max_tokens': 128, 'temperature': 0,
            }, proxies={'http': None, 'https': None}, timeout=60)
            pred = r.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            pred = '<REQ_FAIL: %s>' % e
        ok = any(norm(ref) in norm(pred) or norm(pred) == norm(ref)
                 for ref in refs)
        correct += ok
        by_src[src][0] += ok
        by_src[src][1] += 1
        results.append({'src': src, 'q': q[:150], 'refs': refs[:3],
                        'pred': pred[:200], 'ok': ok})
        if (i + 1) % 20 == 0:
            print('  %d/%d  acc=%d' % (i + 1, len(items), correct), flush=True)

    n = len(items)
    print('\n== s2v2 QA (原始数据集 validation, 未见过) ==')
    for src, (c, t) in by_src.items():
        print('  %-10s %d/%d = %.1f%%' % (src, c, t, 100 * c / max(t, 1)))
    print('  合计       %d/%d = %.1f%%' % (correct, n, 100 * correct / max(n, 1)))
    json.dump({'n': n, 'acc': correct / max(n, 1), 'results': results},
              open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('明细: %s' % OUT)


if __name__ == '__main__':
    main()

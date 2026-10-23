#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s2v2 general QA 能力批量评测.

从 general_qa_300k 池抽题 (排除 s2v2 训练用过的 100k), 发到 8100 chat API,
统计准确率 (归一化精确匹配 + 包含匹配), 保存错题样例.
"""
import json
import random
import re
import requests
import sys

QA_POOL = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/general_qa_300k/train.jsonl')
# s2v2 训练集 (mcvla_stage2_v2) 的 24.7 万个 QA 问题, 全部排除
TRAIN_QA = '/apdcephfs/private_chengzhixu/s2v2_train_qa_questions.json'
URL = 'http://localhost:8100/v1/chat/completions'
N = 100
OUT = '/apdcephfs/private_chengzhixu/s2v2_qa_eval_v2.json'


def norm(s):
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()


def main():
    random.seed(123)
    train_qs = set(norm(q) for q in json.load(open(TRAIN_QA)))
    pool = []
    with open(QA_POOL) as f:
        for line in f:
            try:
                d = json.loads(line)
                if norm(d['messages'][0]['content']) not in train_qs:
                    pool.append(d)
            except Exception:
                pass
    print('池 %d 题, 排除训练集后 %d 题' % (pool and len(pool) + len(train_qs) or 0, len(pool)))
    samples = random.sample(pool, N)

    exact = contain = 0
    results = []
    for i, d in enumerate(samples):
        q = d['messages'][0]['content']
        ref = d['messages'][1]['content']
        try:
            r = requests.post(URL, json={
                'model': 'jarvisvla_s2v2',
                'messages': [{'role': 'user', 'content': q}],
                'max_tokens': 256, 'temperature': 0,
            }, proxies={'http': None, 'https': None}, timeout=60)
            pred = r.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            pred = '<REQ_FAIL: %s>' % e
        e_ok = norm(pred) == norm(ref)
        c_ok = norm(ref) in norm(pred)
        exact += e_ok
        contain += c_ok
        results.append({'q': q[:150], 'ref': ref[:200], 'pred': pred[:200],
                        'exact': e_ok, 'contain': c_ok})
        if (i + 1) % 10 == 0:
            print('  %d/%d  exact=%d contain=%d' % (i + 1, N, exact, contain), flush=True)

    acc_e, acc_c = exact / N, contain / N
    print('\n== s2v2 QA (未见题 %d 道) ==' % N)
    print('精确匹配: %d/%d = %.1f%%' % (exact, N, acc_e * 100))
    print('包含匹配: %d/%d = %.1f%%' % (contain, N, acc_c * 100))
    json.dump({'n': N, 'exact_acc': acc_e, 'contain_acc': acc_c, 'results': results},
              open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('明细: %s' % OUT)


if __name__ == '__main__':
    main()

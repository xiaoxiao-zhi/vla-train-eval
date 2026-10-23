#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""非 no-op 帧的按键逐类预测准确率 (三个模型对比).

口径:
  召回   = P(pred 含该键 | gold 含该键)          —— 单键"认出来"能力
  集合全对 = gold 含该键 且 整组按键集合完全一致   —— 组合复现能力
  幻觉   = gold 不含该键 但 pred 含
hotbar(180-189) 聚合为一类; 样本为记忆度检查的 5000 条抽样 (seed=42).
"""
import json
import re
from collections import defaultdict

TOK_RE = re.compile(r'<\|reserved_special_token_(\d+)\|>')
ACT_RE = re.compile(r'(<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>)(.*)', re.S)
BTN = {191: 'forward', 192: 'back', 194: 'left', 195: 'right', 197: 'sprint',
       198: 'sneak', 200: 'use', 202: 'drop', 204: 'attack', 206: 'jump',
       189: 'hotbar'}


def parse(s):
    m = ACT_RE.match(s)
    if not m:
        return None, None
    nums = [int(n) for n in TOK_RE.findall(m.group(1))][1:-1]
    return (sorted(n for n in nums if 180 <= n <= 206),
            sorted(n for n in nums if 209 <= n <= 250))


def is_noop(b, c):
    return (not b) and 219 in c and 240 in c and len(c) == 2


def analyze(path, name):
    stat = defaultdict(lambda: [0, 0, 0, 0])   # [gold出现, 召回, 集合全对, pred出现]
    n_active = n_multikey = 0
    for line in open(path):
        d = json.loads(line)
        gb, gc = parse(d['gold'])
        ob, oc = parse(d['output'])
        if gb is None or ob is None:
            continue
        if is_noop(gb, gc):
            continue
        n_active += 1
        gset, oset = set(gb), set(ob)
        if len(gset) > 1:
            n_multikey += 1
        exact = (gset == oset)
        for k in {189 if 180 <= n <= 189 else n for n in gset | oset}:
            s = stat[k]
            gk = any((n == k) or (k == 189 and 180 <= n <= 189) for n in gset)
            ok = any((n == k) or (k == 189 and 180 <= n <= 189) for n in oset)
            if gk:
                s[0] += 1
                if ok:
                    s[1] += 1
                if exact:
                    s[2] += 1
            if ok:
                s[3] += 1
    print('===== %s: 非no-op帧 n=%d (其中多键组合帧 %d, %.0f%%) ====='
          % (name, n_active, n_multikey, n_multikey * 100.0 / max(n_active, 1)))
    print('  %-9s %8s %10s %12s %10s' % ('按键', 'gold出现', '召回', '集合全对', '幻觉次数'))
    for k in sorted(stat, key=lambda x: -stat[x][0]):
        g, r, e, p = stat[k]
        halluc = max(p - r, 0)
        print('  %-9s %8d %9.1f%% %11.1f%% %10d'
              % (BTN.get(k, str(k)), g, r * 100.0 / max(g, 1), e * 100.0 / max(g, 1), halluc))
    print()


if __name__ == '__main__':
    base = '/apdcephfs/private_chengzhixu/mem_check/'
    analyze(base + 'mem_histact.jsonl', 'histact (动作+文本)')
    analyze(base + 'mem_actonly.jsonl', 'actonly (仅动作)')
    analyze(base + 'mem_s2hist2.jsonl', 's2v3_hist2 (stage2)')

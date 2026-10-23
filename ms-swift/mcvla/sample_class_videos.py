#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""每大类(mine_block/craft_item/kill_entity)随机抽 30 段, 三来源(6.x/7.x/10.x)各 10 段,
按绝对帧号截取原始视频, 保存到统一目录.

输出: /apdcephfs/private_chengzhixu/class_samples/{类别}_{来源}_{序号}.mp4
"""
import json
import os
import random
import re
import subprocess
from multiprocessing import Pool
from collections import defaultdict

VPT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/vpt')
TASKS_JSONL = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
               'chengzhixu/dataset/minecraft-vla-sft/video_tasks_nodrop.jsonl')
OUT = '/apdcephfs/private_chengzhixu/class_samples'
N_PER_SRC = 10
SEG_MIN, SEG_MAX = 60, 300   # 段长限制 (帧), 控制单个文件大小
random.seed(42)


def main():
    loc = {}
    for d in ('6.x', '7.x', '10.x'):
        for f in os.listdir(os.path.join(VPT, d)):
            if f.endswith('.mp4'):
                loc[f[:-4]] = d

    by = defaultdict(list)   # (task大类, 来源) -> [row]
    with open(TASKS_JSONL) as f:
        for line in f:
            r = json.loads(line)
            cat = r['task'].split(':')[0]
            if cat not in ('mine_block', 'craft_item', 'kill_entity'):
                continue
            if not (SEG_MIN <= r['abs_end'] - r['abs_start'] + 1 <= SEG_MAX):
                continue
            base = re.sub(r'-\d+-\d+$', '', r['video_id'])
            src = loc.get(base)
            if src:
                by[(cat, src)].append(r)

    jobs = []
    for cat in ('mine_block', 'craft_item', 'kill_entity'):
        for src in ('6.x', '7.x', '10.x'):
            pool = by[(cat, src)]
            picks = random.sample(pool, min(N_PER_SRC, len(pool)))
            print('%s/%s: 候选 %d 段 -> 抽 %d' % (cat, src, len(pool), len(picks)))
            for i, r in enumerate(picks, 1):
                r['_src'] = src
                r['_base'] = re.sub(r'-\d+-\d+$', '', r['video_id'])
                jobs.append((r, '%s_%s_%02d.mp4' % (cat, src.replace('.', ''), i)))
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(OUT, 'index.jsonl'), 'w') as idx:
        for r, name in jobs:
            idx.write(json.dumps({'file': name, **r}, ensure_ascii=False) + '\n')

    with Pool(8) as p:
        results = p.starmap(cut, jobs)
    ok = sum(1 for x in results if x)
    print('完成: %d/%d 个视频 -> %s' % (ok, len(jobs), OUT))


def cut(r, name):
    src = os.path.join(VPT, r['_src'], r['_base'] + '.mp4')
    dst = os.path.join(OUT, name)
    try:
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', src,
            '-vf', "select='between(n,%d,%d)'" % (r['abs_start'], r['abs_end']),
            '-fps_mode', 'passthrough', dst,
        ], check=True, timeout=120)
        return True
    except Exception as e:
        print('FAIL %s: %s' % (name, e), flush=True)
        return False


if __name__ == '__main__':
    main()

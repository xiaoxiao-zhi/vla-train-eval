#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""双模型同批采样对比: flat3_r2(无加权) vs flat3_top5weighted(加权).

从 stage2_train_flat3.jsonl 的 VLA 样本中固定 seed 蓄水池抽样,
同一 prompt 发给两个模型, 非贪婪采样, 逐条保存 gold + 两模型输出.
并发: ThreadPoolExecutor(N) + requests + as_completed, 与 memorization_check 同逻辑.
启动: 先 wait_service 双服务, 全部就绪才开跑 (防 8101 未就绪崩溃).
"""
import argparse
import base64
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

CAM_X = re.compile(r'<\|reserved_special_token_(209|21[0-9]|22[0-9])\|>')
CAM_Y = re.compile(r'<\|reserved_special_token_(23[0-9]|24[0-9]|250)\|>')
ACT_RE = re.compile(r'(<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>)(.*)', re.S)
TLS = threading.local()


def get_session():
    """线程本地 requests.Session (连接复用, 线程安全)."""
    if not hasattr(TLS, 's'):
        TLS.s = requests.Session()
        TLS.s.proxies = {'http': None, 'https': None}
    return TLS.s


def img_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def build_content(user_text, b64s):
    """flat3: 指令 + N个 observation 直排, 按 '\nobservation: \n<image>' 分隔."""
    parts = user_text.split('\nobservation: \n<image>')
    content = []
    ii = 0
    for i, seg in enumerate(parts):
        if i > 0:
            content.append({'type': 'image_url',
                            'image_url': {'url': 'data:image/jpeg;base64,' + b64s[ii]}})
            ii += 1
        if seg:
            content.append({'type': 'text', 'text': seg})
    return content


def parse_action(text):
    m = ACT_RE.search(text)
    if not m:
        return None, text.strip()
    act, txt = m.group(1), m.group(2).strip()
    mx, my = CAM_X.search(act), CAM_Y.search(act)
    cx = int(mx.group(1)) - 219 if mx else None
    cy = int(my.group(1)) - 240 if my else None
    kb = re.findall(r'<\|reserved_special_token_(1[8-9][0-9]|20[0-6])\|>', act)
    return {'camera_x': cx, 'camera_y': cy, 'keyboard': kb,
            'raw': act, 'text': txt}, txt


def wait_service(url, model, timeout_s=1800):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url + '/models', timeout=5, proxies={'http': None, 'https': None})
            if r.status_code == 200 and model in r.text:
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError('服务 %s 未就绪 (等了 %ds)' % (url, timeout_s))


def one_req(session, url, model, content, retries=3):
    """单次请求: 网络错重试, 配置错直抛."""
    payload = {'model': model, 'messages': [{'role': 'user', 'content': content}],
               'max_tokens': 120, 'temperature': 0.9,
               'top_p': 0.99, 'skip_special_tokens': False, 'top_k': -1}
    last_err = None
    for att in range(retries):
        try:
            r = session.post(url + '/chat/completions', json=payload, timeout=300)
            if r.status_code != 200:
                raise RuntimeError('HTTP %d: %s' % (r.status_code, r.text[:200]))
            d = r.json()
            if 'choices' not in d or not d['choices']:
                raise RuntimeError('响应缺 choices: %s' % str(d)[:200])
            return d['choices'][0]['message']['content']
        except RuntimeError:
            raise
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.5 * (att + 1))
    raise RuntimeError('重试 %d 次仍失败: %r' % (retries, last_err))


def do_req(rec, url_a, model_a, url_b, model_b, retries=3):
    """单条任务: 同一 content 发给两个模型, 返回 (rec, out_a, out_b)."""
    session = get_session()
    b64s = [img_b64(p) for p in rec['images']]
    content = build_content(rec['user'], b64s)
    out_a = one_req(session, url_a, model_a, content, retries)
    out_b = one_req(session, url_b, model_b, content, retries)
    return rec, out_a, out_b


def run(args):
    # 双服务预检: 全部就绪才开跑
    wait_service(args.url_a, args.model_a)
    print('服务A就绪: %s' % args.model_a, flush=True)
    wait_service(args.url_b, args.model_b)
    print('服务B就绪: %s, 开始采样 %d 条' % (args.model_b, args.n), flush=True)

    # 蓄水池抽样 (只留有 images 的 VLA 样本)
    rng = random.Random(args.seed)
    reservoir = []
    total = 0
    for line in open(args.data):
        d = json.loads(line)
        if not d.get('images'):
            continue
        rec = {'idx': total, 'user': d['messages'][0]['content'],
               'gold': d['messages'][1]['content'], 'images': d['images']}
        total += 1
        if len(reservoir) < args.n:
            reservoir.append(rec)
        else:
            j = rng.randrange(total)
            if j < args.n:
                reservoir[j] = rec
    print('VLA总量: %d, 抽样: %d (seed=%d)' % (total, len(reservoir), args.seed), flush=True)

    stats = {'n': 0, 'a_act': 0, 'b_act': 0, 'a_full': 0, 'b_full': 0}
    t0 = time.time()
    n_since_flush = 0
    with open(args.out, 'w') as fo, \
            tqdm(total=len(reservoir), dynamic_ncols=True, unit='req',
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(do_req, rec, args.url_a, args.model_a,
                          args.url_b, args.model_b, 3) for rec in reservoir]
        for fut in as_completed(futs):
            rec, out_a, out_b = fut.result()  # 异常直接抛出(含重试耗尽)
            gold_act, gold_txt = parse_action(rec['gold'])
            a_act, a_txt = parse_action(out_a)
            b_act, b_txt = parse_action(out_b)
            stats['n'] += 1
            if a_act:
                stats['a_act'] += 1
                if a_act['raw'] == gold_act['raw'] and a_txt == gold_txt:
                    stats['a_full'] += 1
            if b_act:
                stats['b_act'] += 1
                if b_act['raw'] == gold_act['raw'] and b_txt == gold_txt:
                    stats['b_full'] += 1
            fo.write(json.dumps({
                'idx': rec['idx'],
                'gold': {'action': gold_act, 'text': gold_txt},
                'model_a': {'action': a_act, 'text': a_txt, 'raw': out_a},
                'model_b': {'action': b_act, 'text': b_txt, 'raw': out_b},
            }, ensure_ascii=False) + '\n')
            n_since_flush += 1
            if n_since_flush >= 200:
                fo.flush()
                n_since_flush = 0
            pbar.update(1)
            n = max(stats['n'], 1)
            pbar.set_postfix({'A_act': '%.3f' % (stats['a_act'] / n),
                              'B_act': '%.3f' % (stats['b_act'] / n)})
        fo.flush()

    summary = {
        'data': args.data, 'n': stats['n'],
        'model_a': args.model_a, 'model_b': args.model_b,
        'a_act_rate': stats['a_act'] / max(stats['n'], 1),
        'b_act_rate': stats['b_act'] / max(stats['n'], 1),
        'a_full_rate': stats['a_full'] / max(stats['n'], 1),
        'b_full_rate': stats['b_full'] / max(stats['n'], 1),
        'secs': round(time.time() - t0, 1),
    }
    with open(args.out + '.summary.json', 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('汇总: %s' % json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--url-a', required=True, help='模型A (无加权)')
    ap.add_argument('--model-a', required=True)
    ap.add_argument('--url-b', required=True, help='模型B (加权)')
    ap.add_argument('--model-b', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--concurrency', type=int, default=256)
    args = ap.parse_args()
    run(args)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆度检查: 训练数据同输入 -> vLLM 输出 vs 训练 target.

用途: 检查训练集生成匹配率，不单独据此判断欠拟合或过拟合。
方法: 固定 seed 蓄水池采样；默认 temperature=0.9/top_p=0.99/top_k=-1。
并发: ThreadPoolExecutor(256) + requests + as_completed, tqdm 实时进度条.
落盘: 每条 {idx, gold, output, act_match, txt_match, full_match}; 结尾汇总匹配率.
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

SEP = '\nobservation: \n<image>'
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


def build_messages(msgs, img_b64s):
    """按 <image> 逐一插图，保留 observation、换行及其他输入文本。"""
    out, ii = [], 0
    for m in msgs:
        if m['role'] == 'user':
            parts = m['content'].split('<image>')
            content = []
            for k, seg in enumerate(parts):
                if k > 0:
                    if ii >= len(img_b64s):
                        raise ValueError('Image placeholders exceed supplied images')
                    content.append({'type': 'image_url',
                                    'image_url': {'url': 'data:image/jpeg;base64,' + img_b64s[ii]}})
                    ii += 1
                if seg:
                    content.append({'type': 'text', 'text': seg})
            out.append({'role': 'user', 'content': content})
        else:
            out.append({'role': m['role'], 'content': m['content']})
    if ii != len(img_b64s):
        raise ValueError('Supplied images do not match image placeholders')
    return out


def split_target(s):
    m = ACT_RE.match(s)
    if m:
        return m.group(1), m.group(2)          # 动作段, 尾部文本
    return '', s


def wait_service(url, model, timeout_s=900):
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


def do_req(rec, url, model, retries=3, temperature=0.9, top_p=0.99, top_k=-1):
    """单条请求: 构建多轮messages(图片b64) -> POST -> (rec, output). 网络错重试, 配置错直抛.
    注意: top_p/top_k 必须显式传! 否则 vLLM 用 generation_config.json 默认值
    (temperature=0.01, top_k=1, top_p=0.001) => 等效贪婪, camera 分布会被压塌 (10-10 事故)."""
    b64s = [img_b64(p) for p in rec['images']]
    payload = {'model': model,
               'messages': build_messages(rec['msgs'], b64s),
               'max_tokens': 120, 'temperature': temperature,
               'top_p': top_p, 'top_k': top_k,
               'skip_special_tokens': False}
    last_err = None
    for att in range(retries):
        try:
            r = get_session().post(url + '/chat/completions', json=payload, timeout=300)
            if r.status_code != 200:
                raise RuntimeError('HTTP %d: %s' % (r.status_code, r.text[:200]))
            d = r.json()
            if 'choices' not in d or not d['choices']:
                raise RuntimeError('响应缺 choices: %s' % str(d)[:200])
            return rec, d['choices'][0]['message']['content']
        except RuntimeError:
            raise  # 配置类错误不重试
        except requests.RequestException as e:  # 网络瞬断重试
            last_err = e
            time.sleep(1.5 * (att + 1))
    raise RuntimeError('重试 %d 次仍失败: %r' % (retries, last_err))


def record(rec, out, stats):
    g_act, g_txt = split_target(rec['gold'])
    o_act, o_txt = split_target(out)
    full = (out == rec['gold'])
    act_m = (g_act != '' and o_act == g_act)
    txt_m = (o_txt.strip() == g_txt.strip())
    stats['n'] += 1
    stats['full'] += int(full)
    stats['act'] += int(act_m)
    stats['txt'] += int(txt_m)
    if not out:
        stats['no_output'] += 1
    return {'idx': rec['idx'], 'gold': rec['gold'], 'output': out,
            'act_match': act_m, 'txt_match': txt_m, 'full_match': full}


def load_samples(data, n, seed, vla_only, reference_results=None):
    rng = random.Random(seed)
    kept, seen = [], 0
    with open(data) as f:
        for line in f:
            d = json.loads(line)
            if vla_only and not d.get('images'):
                continue
            rec = {'idx': seen,
                   'msgs': d['messages'][:-1],
                   'gold': d['messages'][-1]['content'],
                   'images': d.get('images', [])}
            seen += 1
            if len(kept) < n:
                kept.append(rec)
            else:
                j = rng.randrange(seen)
                if j < n:
                    kept[j] = rec
    if len(kept) != n:
        raise ValueError('Requested %d samples, found %d' % (n, len(kept)))
    if reference_results:
        with open(reference_results) as f:
            reference = [json.loads(line) for line in f]
        expected = {r['idx']: r['gold'] for r in reference}
        actual = {r['idx']: r['gold'] for r in kept}
        if len(reference) != n or len(expected) != n or actual != expected:
            raise ValueError('Sample indices/gold do not exactly match reference results')
        print('样本逐条核验通过: %d 条 idx/gold 与 %s 完全一致' %
              (n, reference_results), flush=True)
    print('全量 %d 行, 抽样 %d 条 (seed=%d)' % (seen, len(kept), seed), flush=True)
    return kept


def run(args):
    kept = load_samples(args.data, args.n, args.seed, args.vla_only, args.reference_results)
    wait_service(args.base_url, args.model_name)
    print('服务就绪, 开始拟合 %d 条' % len(kept), flush=True)

    stats = {'n': 0, 'full': 0, 'act': 0, 'txt': 0, 'no_output': 0}
    t0 = time.time()
    n_since_flush = 0
    with open(args.out, 'w') as fo, \
            tqdm(total=len(kept), dynamic_ncols=True, unit='req',
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(do_req, rec, args.base_url, args.model_name, 3,
                          args.temperature, args.top_p, args.top_k) for rec in kept]
        for fut in as_completed(futs):
            rec, out = fut.result()          # 异常直接抛出(含重试耗尽)
            fo.write(json.dumps(record(rec, out, stats), ensure_ascii=False) + '\n')
            n_since_flush += 1
            if n_since_flush >= 200:
                fo.flush()
                n_since_flush = 0
            pbar.update(1)
            n = max(stats['n'], 1)
            pbar.set_postfix({'full': '%.3f' % (stats['full'] / n),
                              'act': '%.3f' % (stats['act'] / n),
                              'txt': '%.3f' % (stats['txt'] / n)})
        fo.flush()

    summary = {'data': args.data, 'model': args.model_name, 'n': stats['n'],
               'full_match': stats['full'], 'act_match': stats['act'],
               'txt_match': stats['txt'], 'no_output': stats['no_output'],
               'full_rate': stats['full'] / max(stats['n'], 1),
               'act_rate': stats['act'] / max(stats['n'], 1),
               'txt_rate': stats['txt'] / max(stats['n'], 1),
               'secs': round(time.time() - t0, 1)}
    with open(args.out + '.summary.json', 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print('汇总: %s' % json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='训练 jsonl')
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--out', required=True, help='结果 jsonl (每条 gold+output+match)')
    ap.add_argument('--n', type=int, default=30000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--concurrency', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-p', type=float, default=0.99)
    ap.add_argument('--top-k', type=int, default=-1)
    ap.add_argument('--vla-only', action='store_true',
                    help='只抽带帧图的VLA样本 (跳过general-qa纯文本样本)')
    ap.add_argument('--reference-results',
                    help='校验抽样 idx/gold 与已有拟合结果完全一致，否则报错')
    args = ap.parse_args()
    run(args)

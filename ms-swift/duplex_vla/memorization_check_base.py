#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JarvisVLA 原始基座模型记忆度检查: minecraft-vla-sft 原始数据同输入对比.

数据: parquet 分片 (conversations + image_bytes), 单轮/多轮通用.
采样: 随机抽 30 个分片 (约50万行) 做蓄水池采样 n 条 (seed 固定).
说明: 分片按视频组织, 样本会聚集在这30个分片内 (对分布/记忆度统计无偏影响可忽略).
"""
import argparse
import base64
import glob
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
import requests
from tqdm import tqdm

ACT_RE = re.compile(r'(<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>)(.*)', re.S)
WRAP_RE = re.compile(r"^\['(.*)'\]\s*$", re.S)
TLS = threading.local()
URLS = []          # 多端点负载均衡 (random.choice 分发)


def unwrap(out):
    """剥掉官方模型的 "['...']" 列表包装."""
    m = WRAP_RE.match(out)
    return m.group(1) if m else out


def get_session():
    if not hasattr(TLS, 's'):
        TLS.s = requests.Session()
        TLS.s.proxies = {'http': None, 'https': None}
    return TLS.s


def split_target(s):
    m = ACT_RE.match(s)
    if m:
        return m.group(1), m.group(2)
    return '', s


def conv_parts(conv):
    """conversations -> (输入messages结构, gold末轮assistant文本, 图片bytes列表)."""
    user_msgs, imgs = [], []
    for m in conv:
        if m['role'] == 'user':
            items = []          # [{'t': text} 或 {'i': True}]
            for it in m['content']:
                if it['type'] == 'text' and it.get('text'):
                    items.append({'t': it['text']})
                else:
                    items.append({'i': True})
            user_msgs.append(items)
        elif m['role'] == 'assistant':
            txt = ''.join(it['text'] for it in m['content'] if it['type'] == 'text')
            user_msgs.append({'a': txt})
    gold = user_msgs.pop()            # 末轮 assistant 是 target
    assert 'a' in gold, '末轮非 assistant: %r' % conv
    return user_msgs, gold['a'], imgs


def attach_images(user_msgs, img_b64s):
    out, ii = [], 0
    for m in user_msgs:
        if isinstance(m, dict) and 'a' in m:
            out.append({'role': 'assistant', 'content': m['a']})
        else:
            content = []
            for it in m:
                if 't' in it:
                    content.append({'type': 'text', 'text': it['t']})
                else:
                    content.append({'type': 'image_url',
                                    'image_url': {'url': 'data:image/jpeg;base64,' + img_b64s[ii]}})
                    ii += 1
            out.append({'role': 'user', 'content': content})
    return out


def do_req(rec, url, model, retries=3, temperature=0.9, top_p=0.99, top_k=-1):
    b64s = [base64.b64encode(b).decode() for b in rec['imgs']]
    payload = {'model': model,
               'messages': attach_images(rec['msgs'], b64s),
               'max_tokens': 120, 'temperature': temperature,
               'top_p': top_p, 'top_k': top_k,
               'skip_special_tokens': False}
    last_err = None
    for att in range(retries):
        try:
            u = random.choice(URLS) if URLS else url   # 多端点随机分发
            r = get_session().post(u + '/chat/completions', json=payload, timeout=300)
            if r.status_code != 200:
                raise RuntimeError('HTTP %d: %s' % (r.status_code, r.text[:200]))
            d = r.json()
            if 'choices' not in d or not d['choices']:
                raise RuntimeError('响应缺 choices: %s' % str(d)[:200])
            return rec, d['choices'][0]['message']['content']
        except RuntimeError:
            raise
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.5 * (att + 1))
    raise RuntimeError('重试 %d 次仍失败: %r' % (retries, last_err))


def record(rec, out, stats):
    raw = out
    out = unwrap(out)                       # 匹配用剥壳后的内容
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
    return {'idx': rec['idx'], 'id': rec['id'], 'gold': rec['gold'],
            'output': raw, 'output_unwrapped': out,
            'act_match': act_m, 'txt_match': txt_m, 'full_match': full}


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


def run(args):
    global URLS
    if args.base_urls:
        URLS = ['http://localhost:%s/v1' % p.strip() for p in args.base_urls.split(',')]
        print('多端点负载均衡: %d 个实例' % len(URLS), flush=True)
    wait_service(args.base_url, args.model_name)
    print('服务就绪, 开始采样', flush=True)

    files = sorted(glob.glob(args.data_dir + '/train-*.parquet'))
    random.Random(args.seed).shuffle(files)
    picked = files[:args.n_shards]
    print('分片总数 %d, 随机抽 %d 个: %s ...' % (len(files), len(picked), picked[0].split('/')[-1]), flush=True)

    rng = random.Random(args.seed)
    kept, seen = [], 0
    for f in picked:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=['id', 'conversations', 'image_bytes'])
            ids = t['id'].to_pylist()
            convs = t['conversations'].to_pylist()
            imgs = t['image_bytes'].to_pylist()
            for i in range(len(ids)):
                try:
                    msgs, gold, _ = conv_parts(convs[i])
                    rec = {'idx': seen, 'id': ids[i], 'msgs': msgs, 'gold': gold,
                           'imgs': imgs[i] or []}
                except Exception:
                    continue          # 结构异常样本跳过 (计数在日志里)
                seen += 1
                if len(kept) < args.n:
                    kept.append(rec)
                else:
                    j = rng.randrange(seen)
                    if j < args.n:
                        kept[j] = rec
    print('扫描 %d 行, 抽样 %d 条 (seed=%d)' % (seen, len(kept), args.seed), flush=True)

    stats = {'n': 0, 'full': 0, 'act': 0, 'txt': 0, 'no_output': 0}
    t0 = time.time()
    n_flush = 0
    with open(args.out, 'w') as fo, \
            tqdm(total=len(kept), dynamic_ncols=True, unit='req',
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(do_req, rec, args.base_url, args.model_name,
                          3, args.temperature, args.top_p, args.top_k) for rec in kept]
        for fut in as_completed(futs):
            rec, out = fut.result()
            fo.write(json.dumps(record(rec, out, stats), ensure_ascii=False) + '\n')
            n_flush += 1
            if n_flush >= 200:
                fo.flush()
                n_flush = 0
            pbar.update(1)
            n = max(stats['n'], 1)
            pbar.set_postfix({'full': '%.3f' % (stats['full'] / n),
                              'act': '%.3f' % (stats['act'] / n),
                              'txt': '%.3f' % (stats['txt'] / n)})
        fo.flush()

    summary = {'data': args.data_dir, 'model': args.model_name, 'n': stats['n'],
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
    ap.add_argument('--data-dir', required=True, help='minecraft-vla-sft/data 目录')
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--base-urls', default=None,
                    help='逗号分隔多端点, 随机分发负载均衡 (覆盖 --base-url)')
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--n-shards', type=int, default=30, help='随机抽多少个分片做蓄水池')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--concurrency', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.9,
                    help='解码温度 (0=贪婪; 官方评测协议为0.9)')
    ap.add_argument('--top-p', type=float, default=0.99,
                    help='必须显式传! 否则vLLM用generation_config默认0.001=准贪婪')
    ap.add_argument('--top-k', type=int, default=-1,
                    help='必须显式传! 否则vLLM用generation_config默认1=argmax')
    args = ap.parse_args()
    run(args)

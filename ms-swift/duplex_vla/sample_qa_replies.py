#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 Qwen2.5-7B-Instruct (vLLM) 给 general_qa_300k 的每题采样 3 条自然回复.

设计:
  - 异步流式: OpenAI 兼容 API, 并发 128, 主进程单点写 jsonl (无死锁风险)
  - 每题 3 次采样 (temperature 0.7, max_tokens 2048), 全部保留
  - 元数据: 数据来源 (datasource), 问题 id (qa_000001 格式), 采样轮次 (sample 1/2/3)
  - 断点续跑: 启动时扫描已完成的 (qid, sample), 跳过
输出: /apdcephfs/share_chengzhixu/.../dataset/qa_sampled/qwen25_32b_replies.jsonl
  每行: {"qid": ..., "datasource": ..., "question": ..., "gold": ...,
         "sample": 1, "reply": "..."}
"""
import argparse
import asyncio
import json
import os

import aiohttp

QA_PATH = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
           'chengzhixu/dataset/general_qa_300k/train.jsonl')
OUT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/qa_sampled/qwen25_32b_replies.jsonl')
MODEL = '/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/Qwen2.5-7B-Instruct'
SYSTEM = ('You are a helpful assistant. Answer the question or follow the '
          'instruction accurately and concisely.')

N_SAMPLES = 3
CONCURRENCY = 128


async def one_req(session, url, model, messages, max_tokens, sem, retries=3):
    async with sem:
        last_err = None
        for att in range(retries):
            try:
                payload = {
                    'model': model,
                    'messages': messages,
                    'max_tokens': max_tokens,
                    'temperature': 0.7,
                    'top_p': 0.9,
                }
                async with session.post(url + '/chat/completions', json=payload,
                                        proxy=None, timeout=aiohttp.ClientTimeout(total=600)) as r:
                    if r.status == 200:
                        d = await r.json()
                        return d['choices'][0]['message']['content']
                    body = (await r.text())[:200]
                    # 404/400 是配置错误, 重试无意义, 直接失败
                    if r.status in (400, 404):
                        raise RuntimeError('HTTP %d (model=%s): %s' % (r.status, model, body))
                    last_err = 'HTTP %d: %s' % (r.status, body)
            except RuntimeError:
                raise
            except Exception as e:
                last_err = str(e)
            await asyncio.sleep(2 * (att + 1))
        raise RuntimeError('重试 %d 次仍失败: %s' % (retries, last_err))


async def main_async(args):
    url = args.base_url
    tasks_meta = []  # (qid, datasource, question, gold, sample_idx)
    for i, line in enumerate(open(args.qa_path)):
        d = json.loads(line)
        qid = 'qa_%06d' % i
        for s in range(1, N_SAMPLES + 1):
            tasks_meta.append((qid, d.get('datasource', '?'),
                               d['messages'][0]['content'],
                               d['messages'][1]['content'], s))

    # 断点续跑: 已完成的 (qid, sample) 跳过
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                d = json.loads(line)
                done.add((d['qid'], d['sample']))
            except Exception:
                pass
    todo = [t for t in tasks_meta if (t[0], t[4]) not in done]
    print('总任务 %d, 已完成 %d, 待跑 %d' % (len(tasks_meta), len(done), len(todo)), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    out_f = open(args.out, 'a')          # ★ 主进程单点写, 无死锁
    n_done = 0
    CHUNK = 4096

    try:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
        prog = Progress(TextColumn('[progress.description]{task.description}'),
                        BarColumn(), TextColumn('{task.completed}/{task.total}'),
                        TextColumn('({task.percentage:.1f}%)'),
                        TimeElapsedColumn(), TimeRemainingColumn())
        ptask = prog.add_task('采样', total=len(todo))
        prog.start()
    except ImportError:
        prog = None

    async with aiohttp.ClientSession() as session:
        # 分批 await: 控制内存 + 周期性落盘 (meta 与协程绑定, 避免索引错位)
        batch = []   # [(meta_tuple, coroutine)]
        for t in todo:
            qid, src, q, gold, s = t
            msgs = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': q}]
            batch.append((t, one_req(session, url, args.model_name, msgs,
                                     args.max_tokens, sem)))
            if len(batch) >= CHUNK:
                metas = [b[0] for b in batch]
                replies = await asyncio.gather(*[b[1] for b in batch])
                for (qid, src, q, gold, s), rep in zip(metas, replies):
                    out_f.write(json.dumps(
                        {'qid': qid, 'datasource': src, 'question': q,
                         'gold': gold, 'sample': s, 'reply': rep},
                        ensure_ascii=False) + '\n')
                out_f.flush()
                n_done += len(batch)
                batch = []
                if prog is not None:
                    prog.update(ptask, completed=n_done)
                else:
                    print('  %d/%d' % (n_done, len(todo)), flush=True)
        if batch:
            metas = [b[0] for b in batch]
            replies = await asyncio.gather(*[b[1] for b in batch])
            for (qid, src, q, gold, s), rep in zip(metas, replies):
                out_f.write(json.dumps(
                    {'qid': qid, 'datasource': src, 'question': q,
                     'gold': gold, 'sample': s, 'reply': rep},
                    ensure_ascii=False) + '\n')
            out_f.flush()
            n_done += len(batch)
            if prog is not None:
                prog.update(ptask, completed=n_done)
    out_f.close()
    if prog is not None:
        prog.stop()
    print('完成: 累计写入到 %s' % args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://localhost:8300/v1')
    ap.add_argument('--model-name', default='qwen25_7b')
    ap.add_argument('--qa-path', default=QA_PATH)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--concurrency', type=int, default=128)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()

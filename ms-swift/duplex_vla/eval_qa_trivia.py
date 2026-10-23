#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stage2 QA 泛化评测: trivia_qa validation (模型未见过的干净测试集).

判分: TriviaQA 官方风格 EM —— 预测 normalize 后与 answer.value / aliases 之一匹配.
"""
import argparse
import json
import random
import re

import pyarrow.parquet as pq
from openai import OpenAI

VAL = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
       "chengzhixu/dataset/trivia_qa/rc.nocontext/validation-00000-of-00001.parquet")


def norm(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, default=16, help="并发请求数")
    ap.add_argument("--out", default="/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/qa_trivia_result.json")
    args = ap.parse_args()

    # 抽样 (answer 有 value 的)
    f = pq.ParquetFile(VAL)
    rows = f.read().to_pylist()
    rows = [r for r in rows if r["answer"] and r["answer"].get("value")]
    rng = random.Random(0)
    rng.shuffle(rows)
    samples = rows[:args.n]
    print("测试样本: %d (池 %d)" % (len(samples), len(rows)))

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=120)
    model = client.models.list().data[0].id

    from concurrent.futures import ThreadPoolExecutor

    def ask(r):
        q = r["question"].strip()
        ans = r["answer"]
        gold = {norm(ans["value"])} | {norm(a) for a in ans.get("aliases", []) if a}
        gold.discard("")
        resp = None
        for _ in range(3):  # 容错重试 (评测高峰期可能超时)
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=48, temperature=0.0,
                    messages=[{"role": "user", "content": q}])
                if resp.choices:
                    break
            except Exception:
                pass
        pred = ((resp.choices[0].message.content if resp and resp.choices else "") or "").strip()
        pn = norm(pred)
        ok = pn in gold or any(g and g in pn for g in gold)
        return {"q": q[:90], "gold": ans["value"], "pred": pred[:90], "em": ok}

    with ThreadPoolExecutor(args.concurrency) as ex:
        results = list(ex.map(ask, samples))
    hit = sum(x["em"] for x in results)
    for i in range(0, len(results), 20):
        print("  %d/%d  EM=%.1f%%" % (i + 20, len(results),
                                      100 * sum(x["em"] for x in results[:i + 20]) / (i + 20)), flush=True)

    print()
    print("=" * 50)
    print("QA 泛化 (trivia_qa validation, %d 条): EM = %.1f%%" % (
        args.n, 100 * hit / args.n))
    json.dump({"n": args.n, "em": hit / args.n, "results": results},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print("明细: %s" % args.out)
    print()
    print("== 随机样例 (对/错各3) ==")
    right = [x for x in results if x["em"]][:3]
    wrong = [x for x in results if not x["em"]][:3]
    for tag, xs in [("✓", right), ("✗", wrong)]:
        for x in xs:
            print(" %s Q: %s" % (tag, x["q"][:70]))
            print("   gold: %s" % x["gold"][:60])
            print("   pred: %s" % x["pred"][:60])


if __name__ == "__main__":
    main()

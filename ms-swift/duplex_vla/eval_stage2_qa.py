#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stage2 checkpoint QA 能力离线评测: general_qa 抽样生成 + 与参考答案比对."""
import argparse
import json
import random

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

CKPT = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
        "chengzhixu/checkpoint/jarvisvla_s2/v0-20260922-172519/checkpoint-2423")
QA = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
      "chengzhixu/dataset/general_qa/train.jsonl")


def norm(s):
    return " ".join(s.strip().lower().rstrip(".!?).").lstrip("(?.").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/qa_eval_result.json")
    args = ap.parse_args()

    rng = random.Random(0)
    pool = []
    with open(QA) as f:
        for line in f:
            d = json.loads(line)
            pool.append((d["messages"][0]["content"], d["messages"][1]["content"]))
    rng.shuffle(pool)
    samples = pool[:args.n]
    print("QA 评测样本: %d (池 %d)" % (len(samples), len(pool)))

    tok = AutoTokenizer.from_pretrained(CKPT)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    exact = contain = 0
    results = []
    B = 8
    for i in range(0, len(samples), B):
        batch = samples[i:i + B]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True) for q, _ in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     padding_side="left").to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j, (q, ref) in enumerate(batch):
            new = gen[j, inputs.input_ids.shape[1]:]
            pred = tok.decode(new, skip_special_tokens=True)
            e = norm(pred) == norm(ref)
            c = norm(ref) in norm(pred)
            exact += e
            contain += c
            results.append({"q": q[:80], "ref": ref, "pred": pred,
                            "exact": e, "contain": c})
        print("  %d/%d  exact=%.3f contain=%.3f" % (
            min(i + B, len(samples)), len(samples),
            exact / min(i + B, len(samples)), contain / min(i + B, len(samples))),
            flush=True)

    n = len(samples)
    summary = {"n": n, "exact_match": exact / n, "contain_match": contain / n}
    json.dump({"summary": summary, "results": results},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print()
    print("=" * 50)
    print("QA 能力: exact match = %d/%d = %.1f%% | contain = %.1f%%" % (
        exact, n, 100 * exact / n, 100 * contain / n))
    print("明细: %s" % args.out)


if __name__ == "__main__":
    main()

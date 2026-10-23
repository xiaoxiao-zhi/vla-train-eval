#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 probe 的 178后乱码问题: 词表对齐检查 + 教师强制逐位置概率."""
import argparse
import json

CKPT = ("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/"
         "chengzhixu/checkpoint/v2-20260920-214927/checkpoint-1225")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelCls

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    tok = processor.tokenizer
    model = ModelCls.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    print("== 1) 词表对齐检查 ==")
    print("tokenizer len:", len(tok))
    print("config.vocab_size:", model.config.vocab_size)
    print("embedding shape:", model.get_input_embeddings().weight.shape)
    cfg = json.load(open(args.checkpoint + "/config.json"))
    print("config.json vocab_size:", cfg.get("vocab_size"))
    gen_cfg = json.load(open(args.checkpoint + "/generation_config.json"))
    print("generation_config:", gen_cfg)
    for n in (178, 179, 191, 204, 219, 240):
        t = "<|reserved_special_token_%d|>" % n
        print("  %s -> id %s" % (t, tok.convert_tokens_to_ids(t)))
    print("eos:", tok.eos_token, tok.eos_token_id)

    # testset 第0条: 教师强制
    ts = json.loads(open("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/"
                         "share_303556863/chengzhixu/dataset/mcvla_eval/"
                         "testset.jsonl").readline())
    print("\n== 2) 教师强制 (testset[0]) ==")
    print("title:", ts["title"])
    print("gt_action:", ts["gt_action"] if "gt_action" in ts else ts.get("prev_action"))

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "%s\nobservation: \n" % ts["title"]},
            {"type": "image"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": ts["prev_action"]}]},
        {"role": "user", "content": [
            {"type": "text", "text": "observation: \n"},
            {"type": "image"}]},
    ]
    text = processor.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    gt = ts.get("gt_action") or ts["prev_action"]
    full = text + gt + "\n<|im_end|>"
    pil = [Image.open(ts["prev_frame"]).convert("RGB"),
           Image.open(ts["cur_frame"]).convert("RGB")]
    inputs = processor(text=[full], images=pil, return_tensors="pt").to("cuda:0")
    ids = inputs.input_ids[0]
    n_prompt = len(processor(text=[text], images=pil, return_tensors="pt").input_ids[0])
    print("prompt tokens: %d, total: %d" % (n_prompt, len(ids)))
    print("prompt 末尾 decode:", repr(tok.decode(ids[n_prompt-8:n_prompt])))

    with torch.no_grad():
        logits = model(**inputs).logits[0]  # [T, V]
    # 逐位置: 预测位置 i 的分布 -> 目标 token ids[i]
    for i in range(n_prompt - 1, len(ids) - 1):
        top = torch.topk(logits[i].float(), 4)
        tgt = ids[i + 1]
        p_tgt = torch.softmax(logits[i].float(), -1)[tgt]
        print("pos %2d tgt=%-18s p(tgt)=%.4f | top: %s" % (
            i, repr(tok.decode([tgt.item()]))[:18], float(p_tgt),
            [(repr(tok.decode([int(j)]))[:18], round(float(v), 3))
             for j, v in zip(top.indices, top.values)]))


if __name__ == "__main__":
    main()

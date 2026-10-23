#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stage2 checkpoint vLLM 服务快速测试: 纯文本 / 图像理解 / 动作格式 三组."""
import base64
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
FRAMES = Path("/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames")


def img_url(name):
    b = base64.b64encode((FRAMES / name).read_bytes()).decode()
    return "data:image/jpeg;base64," + b


def pretty(txt):
    return (txt.replace("<|reserved_special_token_", "R").replace("|>", "")
               .replace("<|im_end|>", "").replace("\n", " ")[:150])


def query(client, model, msgs, tag, max_tokens=128, timeout_note=True):
    t0 = time.time()
    r = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=0.0,
        extra_body={"skip_special_tokens": False})
    dt = time.time() - t0
    txt = r.choices[0].message.content or "<content=None>"
    finish = r.choices[0].finish_reason
    print("[%s] %.1fs finish=%s" % (tag, dt, finish))
    print("  输出:", repr(pretty(txt)))
    print()
    return txt


def main():
    client = OpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=600)
    model = client.models.list().data[0].id
    print("model:", model)
    print("=" * 60)

    # ---- 1. 纯文本 ----
    print("== 1. 纯文本 ==")
    query(client, model,
          [{"role": "user", "content": "What is 1+1? Answer with just the number."}],
          "文本-数学")
    query(client, model,
          [{"role": "user", "content":
            "Edit the sentence so that it begins with an adverb.\nShe sang a beautiful song."}],
          "文本-QA风格(general_qa)")

    # ---- 2. 图像理解 ----
    print("== 2. 图像理解 ==")
    frame = "jumpy-denim-lion-f153ac423f61-1603-1615_f000001.jpg"
    if not (FRAMES / frame).exists():
        frame = sorted(FRAMES.glob("*.jpg"))[0].name
    query(client, model,
          [{"role": "user", "content": [
              {"type": "text", "text": "What do you see in this image? Describe briefly."},
              {"type": "image_url", "image_url": {"url": img_url(frame)}}]}],
          "图像-描述", max_tokens=96)

    # ---- 3. 动作格式 (评测同款 prompt) ----
    print("== 3. 动作格式 (kill_zombie 评测同款) ==")
    query(client, model,
          [{"role": "user", "content": [
              {"type": "text", "text": "Hunt the zombie.\nobservation: \n"},
              {"type": "image_url", "image_url": {"url": img_url(frame)}}]}],
          "动作-僵尸任务", max_tokens=64)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对拍: 手拼 chat 外壳 vs 官方 apply_chat_template (逐 token)."""
import sys

sys.path.insert(0, "/apdcephfs/private_chengzhixu/ms-swift/duplex_vla")
from duplex_dataset import DuplexRenderer

R = DuplexRenderer(
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/frames",
    "/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B")
tok = R.tok

# 官方模板: system 默认 + 单 user turn + generation prompt
ref = tok.apply_chat_template(
    [{"role": "user", "content": "CONTENT"}],
    tokenize=True, add_generation_prompt=True)["input_ids"]
print("官方:", ref)
print("官方可读:", repr(tok.decode(ref)))

# 我们渲染器的外壳拼法 (与 render() 中一致)
ours = ([R.im_start]
        + tok("system\n", add_special_tokens=False)["input_ids"]
        + R.pfx_sys
        + [R.im_end] + R.nl
        + [R.im_start] + R.pfx_user
        + tok("CONTENT", add_special_tokens=False)["input_ids"]
        + [R.im_end] + R.nl
        + [R.im_start] + R.pfx_ast)
print("手拼:", ours)
print("逐 token 一致:", ref == ours)
if ref != ours:
    n = max(len(ref), len(ours))
    for i in range(n):
        a = ref[i] if i < len(ref) else None
        b = ours[i] if i < len(ours) else None
        if a != b:
            print("首个差异 pos %d: 官方=%r 我们=%r" % (
                i,
                tok.convert_ids_to_tokens(a) if a is not None else "<无>",
                tok.convert_ids_to_tokens(b) if b is not None else "<无>"))
            print("官方上下文:", ref[max(0, i - 3):i + 3])
            print("我们上下文:", ours[max(0, i - 3):i + 3])
            break

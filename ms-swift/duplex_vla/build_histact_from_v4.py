#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""histact 数据 v2: 从 v4 数据直接转换 (继承全部字段, 只注入历史动作).

背景: 独立构建因 hash(vid) 跨进程随机化导致 QA 插入点与 v4 不同, 无法满足
     "与 v4 唯一差异 = 历史动作". 本脚本改为确定性转换:
  - 逐行读 v4/duplex_train.jsonl, 继承 base/t/qa_meta/images/target 全部字段
  - user 拍平串中, 窗口内历史帧 (最后 n_img-1 个 <|obs_asst|> 段) 的
    assistant 文本前注入该帧动作串 (video_sft actions[i], 自带\n, 零额外字符)
  - 窗口外帧 / 当前帧不动
  - 纯字符串操作, 无 tokenizer / 无随机数, 逐字段确定性

输出: duplex_vla_v4_histact/duplex_train.jsonl (覆盖独立构建版)
之后链式: downsample_noop_histact.py -> build_histact_actonly.py
(输入保序 -> downsample seed=42 确定性 -> filtered 的保留集合与 v4 filtered 完全一致)
"""
import json
import os

V4 = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
      'chengzhixu/dataset/duplex_vla_v4/duplex_train.jsonl')
VIDEO_SFT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
             'chengzhixu/dataset/mcvla_video_sft/train.jsonl')
OUT = ('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/'
       'chengzhixu/dataset/duplex_vla_v4_histact/duplex_train.jsonl')

OBS_ASST = '<|obs_asst|>'
IMG_WINDOW = 2

# 1. 轨迹动作表 (video_id -> actions 列表, 动作串自带\n)
acts = {}
for line in open(VIDEO_SFT):
    d = json.loads(line)
    acts[d['video_id']] = d['actions']
print('轨迹动作表: %d 段' % len(acts), flush=True)

# 2. 逐行转换
n_in = n_out = 0
skipped = 0
with open(OUT, 'w') as fo:
    for line in open(V4):
        d = json.loads(line)
        a_list = acts.get(d['base'])
        if a_list is None:
            skipped += 1
            continue
        t = d['t']
        n_img = len(d['images'])
        n_hist_in_win = n_img - 1        # 窗口内历史帧数 (当前帧1 + 窗口内历史)
        user = d['messages'][0]['content']
        if n_hist_in_win > 0:
            parts = user.split(OBS_ASST)
            # 段 j (j>=1) 开头 = 帧 j-1 的 assistant 文本; 注入窗口内帧 i = t-n_hist_in_win..t-1
            # 窗口内帧是最近的 n_hist_in_win 帧 -> 对应最后 n_hist_in_win 个段 (索引 t-n_hist_in_win+1 .. t)
            for i in range(t - n_hist_in_win, t):
                seg_idx = i + 1
                act = a_list[i]
                if not act.endswith('\n'):
                    act += '\n'
                parts[seg_idx] = act + parts[seg_idx]
            d['messages'][0]['content'] = OBS_ASST.join(parts)
        fo.write(json.dumps(d, ensure_ascii=False) + '\n')
        n_out += 1
        n_in += 1

print('输入 %d 行, 输出 %d 行, 跳过(无轨迹) %d' % (n_in, n_out, skipped))
assert skipped == 0, '存在找不到轨迹的样本!'

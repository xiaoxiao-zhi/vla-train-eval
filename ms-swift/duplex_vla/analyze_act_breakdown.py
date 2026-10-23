#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""动作段拆解统计: 起始/结束标记 vs 真实动作内容(按键/camera) 的匹配率.

动作段结构: 178(起始) + [按键token + camera_x + camera_y] + 179(结束)
拆解维度:
  struct : 输出是否 178开头 179结尾 且无杂散 token
  inner  : 中间真实动作 token 是否全对
  buttons: 按键 token 集合是否一致 (180-206, 不含camera)
  cam    : camera 双轴 token (209-229=x, 230-250=y) 是否一致
  noop   : inner == [219,240] 即 no-op 帧, 单独统计
"""
import json
import re
import sys

TOK_RE = re.compile(r'<\|reserved_special_token_(\d+)\|>')
ACT_RE = re.compile(r'(<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>)(.*)', re.S)


def parse_act(s):
    m = ACT_RE.match(s)
    if not m:
        return None
    nums = [int(n) for n in TOK_RE.findall(m.group(1))]
    if not nums or nums[0] != 178 or nums[-1] != 179:
        return None, nums, []
    inner = nums[1:-1]
    buttons = [n for n in inner if 180 <= n <= 206]
    cams = [n for n in inner if 209 <= n <= 250]
    struct_ok = (nums[0] == 178 and nums[-1] == 179
                 and all(180 <= n <= 250 for n in inner))
    return struct_ok, buttons, cams


def analyze(path, name):
    st = dict(n=0, act=0, struct=0, inner=0, buttons=0, cam=0,
              n_noop=0, noop_inner=0, n_active=0, active_inner=0,
              active_buttons=0, active_cam=0, no_act=0)
    for line in open(path):
        d = json.loads(line)
        g, o = d['gold'], d['output']
        st['n'] += 1
        gp = parse_act(g)
        op = parse_act(o)
        if d['act_match']:
            st['act'] += 1
        if op is None or op[0] is None:
            # 模型输出动作段结构异常 (无178..179包裹)
            continue
        g_struct, g_btn, g_cam = gp if gp else (None, [], [])
        o_struct, o_btn, o_cam = op
        st['struct'] += int(o_struct and gp is not None and g_struct)
        inner_eq = (gp is not None and (g_btn + g_cam) == (o_btn + o_cam))
        # 按键和camera分别比较 (忽略顺序)
        btn_eq = (gp is not None and sorted(g_btn) == sorted(o_btn))
        cam_eq = (gp is not None and sorted(g_cam) == sorted(o_cam))
        st['inner'] += int(inner_eq)
        st['buttons'] += int(btn_eq)
        st['cam'] += int(cam_eq)
        # gold 是否 no-op (无按键 + 双轴中心 219/240)
        g_noop = (gp is not None and not g_btn
                  and 219 in g_cam and 240 in g_cam and len(g_cam) == 2)
        if g_noop:
            st['n_noop'] += 1
            st['noop_inner'] += int(inner_eq)
        else:
            st['n_active'] += 1
            st['active_inner'] += int(inner_eq)
            st['active_buttons'] += int(btn_eq)
            st['active_cam'] += int(cam_eq)
    n = max(st['n'], 1)
    print('== %s (n=%d) ==' % (name, st['n']))
    print('  整串动作段一致 act        : %5d  %.1f%%' % (st['act'], st['act'] * 100.0 / n))
    print('  标记结构正确 struct(178/179): %5d  %.1f%%' % (st['struct'], st['struct'] * 100.0 / n))
    print('  真实内容全对 inner        : %5d  %.1f%%' % (st['inner'], st['inner'] * 100.0 / n))
    print('    - 按键部分 buttons      : %5d  %.1f%%' % (st['buttons'], st['buttons'] * 100.0 / n))
    print('    - camera部分 cam        : %5d  %.1f%%' % (st['cam'], st['cam'] * 100.0 / n))
    nn = max(st['n_noop'], 1)
    na = max(st['n_active'], 1)
    print('  --- 按 gold 是否 no-op 拆分 ---')
    print('  no-op帧 (%d, %.0f%%): 内容量一致 %.1f%%' % (
        st['n_noop'], st['n_noop'] * 100.0 / n, st['noop_inner'] * 100.0 / nn))
    print('  有动作帧 (%d, %.0f%%): 内容量一致 %.1f%% | 按键 %.1f%% | camera %.1f%%' % (
        st['n_active'], st['n_active'] * 100.0 / n,
        st['active_inner'] * 100.0 / na, st['active_buttons'] * 100.0 / na,
        st['active_cam'] * 100.0 / na))
    print()


if __name__ == '__main__':
    base = '/apdcephfs/private_chengzhixu/mem_check/'
    analyze(base + 'mem_histact.jsonl', 'histact (动作+文本)')
    analyze(base + 'mem_actonly.jsonl', 'actonly (仅动作)')
    analyze(base + 'mem_s2hist2.jsonl', 's2v3_hist2 (stage2)')
    analyze(base + 'mem_s2hist1.jsonl', 's2v3_hist1 (stage2)')

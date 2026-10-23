import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全双工 v4 交互测试 (单轮拍平格式: <|obs_user|>/<|obs_asst|> + 动作窗口0).

与 v3 (duplex_interact_v2.py) 的差异 —— 只在 prompt 构造与滚动状态:
  - prompt = 单条 user content 字符串 (拍平), 每帧用 <|obs_user|>/<|obs_asst|> 分隔
  - 动作窗口 0: 历史帧 assistant 只留文本 (silence/回答片段), 无动作
  - 图片窗口 2: 窗口内帧 "文本\nobservation: \n<image>", 窗口外帧纯文本
  - target = 当前帧 动作 + 文本 (模型生成后回填历史)
其余 (环境/QA注入/动作解析/输出四件套) 与 v3 相同.

用法 (GPU 机, 先起 vLLM):
  vllm serve <v4_ckpt> --port 8100 --served-model-name duplex_v4 \
      --runner generate --limit-mm-per-prompt '{"image": 8}'
  DISPLAY=:99 python duplex_interact_v4.py --task kill/kill_zombie --max-frames 300
"""
import argparse
import base64
import io
import json
import os
import random
import re
import sys
import time
import math

import numpy as np
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

sys.path.insert(0, '/apdcephfs/private_chengzhixu/JarvisVLA')
from duplex.minecraft_eval_protocol import (
    TASKS as BASELINE_TASKS, DEFAULT_SEED, REPEATS, load_task_cfg,
    build_env as build_shared_env, prepare_episode, finish_episode_recording,
    build_task_instruction,
)

OUT_DIR = '/apdcephfs/private_chengzhixu/JarvisVLA/outputs/rollout/duplex_v4'
QA_POOL_JSON = '/apdcephfs/private_chengzhixu/JarvisVLA/outputs/qa/trivia_validation200/samples.jsonl'
TASKS = {task: BASELINE_TASKS[task][2] for task in (
    'kill/kill_zombie', 'mine/mine_stone', 'craft/craft_crafting_table',
)}
OBS_USER = '<|obs_user|>'
OBS_ASST = '<|obs_asst|>'
IMG_WINDOW = 2
ACT_RE = re.compile(r'(<\|reserved_special_token_178\|>.*?<\|reserved_special_token_179\|>)(.*)', re.S)
# camera token 区段: x轴 209-229, y轴 230-250 (编号范围无歧义, 不依赖位置)
CAM_X_RE = re.compile(r'<\|reserved_special_token_(209|21[0-9]|22[0-9])\|>')
CAM_Y_RE = re.compile(r'<\|reserved_special_token_(23[0-9]|24[0-9]|250)\|>')


def load_qa_pool(path):
    from pathlib import Path
    path = Path(path)
    with path.open() as file:
        records = [json.loads(line) for line in file] if path.suffix == '.jsonl' else json.load(file)
    if not records:
        raise ValueError('QA pool is empty')
    result = []
    for row in records:
        question = row['question'] if 'question' in row else row['q']
        answer = row['gold'] if 'gold' in row else row['a']
        if not question.strip() or not answer.strip():
            raise ValueError('QA pool contains empty question or answer')
        result.append({'q': question, 'a': answer,
                       'source_id': row.get('source_id'), 'source_split': row.get('source_split'),
                       'aliases': row.get('aliases', [answer])})
    return result


def choose_qa(pool, base_seed, episode):
    rng = random.Random(base_seed + episode)
    qa_frame = round(rng.uniform(1.0, 2.0) * 20)
    return rng.choice(pool), qa_frame


def img_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def vllm_chat(base_url, model_name, prompt, images_pil, session):
    """prompt: v4 拍平字符串 (含 N 个 <image> 占位); images_pil: 对应帧.
    返回 (回复文本, 完整请求体) — 请求体原样落盘, 发给 vLLM 的就是存下的."""
    content = []
    parts = prompt.split('<image>')
    if len(parts) - 1 != len(images_pil):
        raise ValueError('Image placeholders and supplied images do not match')
    for i, seg in enumerate(parts):
        if i > 0:
            content.append({'type': 'image_url',
                            'image_url': {'url': 'data:image/jpeg;base64,' + img_b64(images_pil[i - 1])}})
        if seg:
            content.append({'type': 'text', 'text': seg})
    req_body = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': content}],
        # 采样协议(对齐官方): temp=0.9, top_p=0.99, top_k=-1
        # 贪婪(temp=0)会把camera后验坍缩到0档, camera行为失真
        'max_tokens': 100, 'temperature': 0.9, 'top_p': 0.99, 'top_k': -1,
        'skip_special_tokens': False,
    }
    r = session.post('%s/chat/completions' % base_url, json=req_body,
                     proxies={'http': None, 'https': None}, timeout=120)
    r.raise_for_status()
    body = r.json()
    if 'choices' not in body or not body['choices']:
        # 不静默: 把原始响应体完整抛出, 便于定位 (HTTP状态/头/体)
        raise RuntimeError('vLLM 响应缺 choices (status=%s): %s' % (r.status_code, r.text[:800]))
    return body['choices'][0]['message']['content'], req_body


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:8100/v1')
    ap.add_argument('--base-urls', default=os.environ.get('EVAL_BASE_URLS', ''),
                    help='逗号分隔的独立vLLM服务地址，每个运行任务独占一个地址')
    ap.add_argument('--model-name', default='duplex_v4')
    ap.add_argument('--task', default='all', choices=['all'] + list(TASKS))
    ap.add_argument('--max-frames', type=int,
                    help='默认按任务使用非全双工帧预算300/500/400')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--repeats', type=int, default=REPEATS)
    ap.add_argument('--workers', type=int, default=3,
                    help='同时运行的任务进程数；每个任务内部各局串行')
    ap.add_argument('--stagger', type=float, default=10.0,
                    help='任务启动错峰秒数，与非全双工启动配置一致')
    ap.add_argument('--qa-pool', default=QA_POOL_JSON)
    ap.add_argument('--out-dir', default=OUT_DIR,
                    help='输出根目录，每任务每局独立子目录')
    ap.add_argument('--official-url', default='',
                    help='官方模型服务(h=0无历史); 设置后每帧双路采样: '
                         '全双工(带历史格式)+官方(指令+当前帧), 转角token替换为官方结果')
    ap.add_argument('--official-name', default='jarvisvla_official')
    ap.add_argument('--tokenizer', default=('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/'
                                            'share_303556863/chengzhixu/checkpoint/s2v3_hist2/'
                                            'v0-20260929-011248/checkpoint-2900-v4vocab'),
                    help='扩词表 tokenizer 目录 (含 reserved/silence token; 用于动作 re-encode)')
    ap.add_argument('--hist-mode', choices=['text', 'act', 'full'], default='text',
                    help='历史帧回填格式(须与训练数据对齐): '
                         'text=仅文本(v4无历史动作版); '
                         'act=动作段+\\n<|silence|>(actonly版, 模型无文本输出); '
                         'full=完整输出 动作\\n文本(histact版)')
    args = ap.parse_args(argv)
    if args.repeats < 1 or args.workers < 1 or (args.max_frames is not None and args.max_frames < 1):
        ap.error('repeats, workers and max-frames must be positive')
    if not math.isfinite(args.stagger) or args.stagger < 0:
        ap.error('stagger must be finite and nonnegative')
    if not 0 <= args.seed <= 2 ** 32 - args.repeats:
        ap.error('Episode seeds must fit within NumPy uint32 range')
    return args


def run_episode(args, env, record_callback, cfg, task, episode, ep_dir,
                max_frames, tkz, act_tok, session, console, pool):
    from PIL import Image
    out_dir = ep_dir
    run_stamp = str(time.time_ns())
    frames_dir = os.path.join(out_dir, 'frames_%s' % run_stamp)
    os.makedirs(frames_dir, exist_ok=True)
    qa, qa_frame = choose_qa(pool, args.seed, episode)
    qa_q, qa_a = qa['q'], qa['a']
    obs, info, _ = prepare_episode(env, record_callback, cfg, ep_dir, episode, args.seed)
    instruction = build_task_instruction(cfg.task_conf[0]['text'], episode, args.seed, ep_dir)
    console.print('[bold cyan]TASK:[/bold cyan] %s ep%d' % (instruction, episode))
    console.print('QA @frame %d: %s' % (qa_frame, qa_q[:60]))

    chat_log = [{'role': 'task', 'text': instruction}]
    act_log, raw_log, inputs_log = [], [], []
    # v4 滚动状态: hist 存 (user文本, 动作段, assistant文本); 拼接按 --hist-mode 决定回填格式
    hist = []          # [(user文本, 动作段, assistant文本)]
    hist_frames = []   # 与 hist 平行的帧 PIL
    success, t_end = False, max_frames
    t0 = time.time()

    def _dump_all(tag):
        """落盘已收集的全部日志 (正常结束/异常崩溃都调用, 异常不在此吞)."""
        chat_log.append({'role': 'result', 'success': success, 'frames': t_end,
                         'secs': round(time.time() - t0, 1), 'error': tag})
        payloads = {
            'chat': {'task': task, 'episode': episode,
                     'model': args.model_name + ('+official_cam' if args.official_url else ''),
                     'format': 'v4-flat', 'hist_mode': args.hist_mode,
                     'qa': {**qa, 'insert_frame': qa_frame}, 'chat': chat_log},
            'actions': {'frames': t_end, 'actions': act_log},
            'outputs': {'frames': t_end, 'outputs': raw_log},
            'inputs': {'frames': t_end, 'frames_dir': frames_dir, 'inputs': inputs_log},
        }
        for name, payload in payloads.items():
            with open(os.path.join(out_dir, '%s_%s.json' % (name, run_stamp)), 'w') as file:
                json.dump(payload, file, ensure_ascii=False, indent=1)
        result = {'ok': success, 'steps': t_end if tag is None else -1,
                  'episode': episode, 'max_frames': max_frames, 'rollout_frames': len(raw_log),
                  'seed': args.seed + episode, 'qa_inserted': len(raw_log) > qa_frame,
                  'qa_source_id': qa['source_id'], 'qa_pool': args.qa_pool,
                  'seconds': round(time.time() - t0, 1)}
        if tag is not None:
            result['err'] = tag
        with open(os.path.join(out_dir, 'result.json'), 'w') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print('输出: %s/{chat,actions,outputs,inputs}_%s.json + frames_%s/ (error=%s)' % (
            out_dir, run_stamp, run_stamp, tag or 'none'))

    crash_err = None
    try:
        for t in range(max_frames + 20):
            if (success and t > t_end + 20) or (not success and t >= max_frames):
                break
            cur_frame = Image.fromarray(info['pov'])
            cur_frame.save(os.path.join(frames_dir, 't%04d.jpg' % t), quality=85)

            if t == 0:
                user_txt = instruction
            elif t == qa_frame:
                user_txt = qa_q
                console.print('[bold cyan]USER(QA):[/bold cyan] %s' % qa_q)
                chat_log.append({'role': 'user_qa', 'text': qa_q, 'frame': t})
            else:
                user_txt = '<|silence|>'

            # ---- v4 prompt: 拍平单字符串 (历史回填按 --hist-mode 对齐训练格式) ----
            win_start = max(0, t - IMG_WINDOW)
            parts = []
            for i, (u_txt, h_act, h_ans) in enumerate(hist):
                in_img = i >= win_start
                u = u_txt + ('\nobservation: \n<image>' if in_img else '')
                if args.hist_mode == 'text':
                    # v4 无历史动作版: 历史帧 assistant = 仅文本 (训练分布同款)
                    a = h_ans
                elif args.hist_mode == 'act':
                    # actonly 版: 训练分布历史帧 = 动作段\n文本, 模型只生成动作 -> silence 顶文本位
                    a = (h_act + '\n<|silence|>') if in_img else '<|silence|>'
                else:  # full: histact 版, 历史帧 = 完整 动作段\n文本
                    a = (h_act + '\n' + h_ans) if in_img else h_ans
                parts.append(OBS_USER + u + OBS_ASST + a)
            parts.append(OBS_USER + user_txt + '\nobservation: \n<image>')
            prompt = ''.join(parts)
            images_pil = hist_frames[win_start:] + [cur_frame]   # 窗口内历史帧 + 当前帧

            out_text, req_body = vllm_chat(args.base_url, args.model_name, prompt, images_pil, session)

            # ---- 混合模式: 官方模型(h=0, 指令+当前帧)同帧采样, 全双工的转角token换成官方结果 ----
            off_out = None
            if args.official_url:
                off_prompt = instruction + '\nobservation: \n<image>'
                off_out, _ = vllm_chat(args.official_url, args.official_name,
                                       off_prompt, [cur_frame], session)
                mx, my = CAM_X_RE.search(off_out), CAM_Y_RE.search(off_out)
                if mx:
                    out_text = CAM_X_RE.sub(mx.group(0), out_text, count=1)
                if my:
                    out_text = CAM_Y_RE.sub(my.group(0), out_text, count=1)

            # 解析: 动作段 + 文本段
            m = ACT_RE.search(out_text)
            if m:
                act_text, answer_raw = m.group(1), m.group(2)
            else:
                act_text, answer_raw = '', out_text
            answer = answer_raw.strip()
            is_silence = answer in ('', '<|silence|>')
            if is_silence:
                answer = '<|silence|>'
            else:
                console.print('[bold green]ASSISTANT:[/bold green] %s' % answer)
            chat_log.append({'role': 'assistant', 'text': answer, 'text_raw': answer_raw,
                             'is_silence': is_silence, 'frame': t})

            raw_log.append({'frame': t, 'output': out_text, 'action_text': act_text,
                            'text': answer, 'text_raw': answer_raw, 'is_silence': is_silence,
                            **({'official_output': off_out} if off_out is not None else {})})
            # 完整保存发给 vLLM 的请求 (发啥存啥):
            #   prompt: 拍平完整字符串 (含 <image> 占位, 即 vllm_chat 的输入)
            #   request: 实际 POST 的完整请求体 (content 数组, 图片为 base64)
            #   images: 各 <image> 占位对应的帧图片文件名 (frames_*/ 目录)
            _img_names = ['t%04d.jpg' % f
                          for f in range(max(0, t - IMG_WINDOW), t + 1)]
            inputs_log.append({
                'frame': t,
                'user_text': user_txt,
                'prompt': prompt,
                'request': req_body,
                'images': _img_names[-len(images_pil):] if images_pil else [],
                'n_images': len(images_pil),
            })
            # 原则: 有 bug 直接崩溃, 绝不静默兜底替换默认动作
            if not act_text:
                raise RuntimeError('frame %d: 模型输出未解析出动作段: %r' % (t, out_text[:100]))
            env_action = act_tok.decode(
                tkz(act_text, add_special_tokens=False)['input_ids'])[0]

            def _il(v):
                try:
                    return [int(x) for x in v]
                except TypeError:
                    return [int(v)]
            act_log.append({'frame': t, 'act_text': act_text,
                            'action': {'buttons': _il(env_action['buttons']),
                                       'camera': _il(env_action['camera'])}})
            obs, reward, terminated, truncated, info = env.step(env_action)

            # 滚动更新 (历史存完整信息, 拼接格式由 --hist-mode 决定)
            hist.append((user_txt, act_text, answer))
            hist_frames.append(cur_frame)

            if t % 20 == 0:
                print('[frame %3d] act=%s txt=%s' % (
                    t, act_text[:40] or '(空)',
                    '(silence)' if answer == '<|silence|>' else answer[:30]), flush=True)
            if reward > 0 and not success:
                success, t_end = True, t
                console.print('[bold yellow]== 任务成功! frame %d ==[/bold yellow]' % t)
    except Exception as e:
        crash_err = e
        t_end = t
    finally:
        try:
            finish_episode_recording(record_callback)
        except Exception as e:
            if crash_err is None:
                crash_err = e

    dt = time.time() - t0
    if crash_err is None:
        console.print('[bold]== 结束: success=%s, %d 帧, %.0fs ==[/bold]' % (success, t_end, dt))
        console.print('[dim]QA 参考: %s[/dim]' % qa_a[:80])
    _dump_all(crash_err and ('%s: %s' % (type(crash_err).__name__, crash_err)))
    if crash_err is not None:
        raise crash_err
    return success, t_end


def main(argv=None):
    args = parse_args(argv)
    if args.task == 'all':
        from duplex.duplex_eval_group import run_tasks
        return run_tasks(args, TASKS)
    from pathlib import Path
    import requests
    from rich.console import Console
    from transformers import AutoTokenizer
    from minestudio.simulator.callbacks import RecordCallback
    sys.path.insert(0, '/apdcephfs/private_chengzhixu/JarvisVLA')
    from jarvisvla.inference.action_mapping import OneActionTokenizer

    tasks = [args.task]
    root = Path(args.out_dir)
    planned = [root / task.replace('/', '_') / ('ep%d' % episode)
               for task in tasks for episode in range(args.repeats)]
    if any(path.exists() and any(path.iterdir()) for path in planned):
        raise FileExistsError('Episode output already exists; use a new output directory')
    pool = load_qa_pool(args.qa_pool)
    tkz = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    checked = tkz('<|reserved_special_token_204|>', add_special_tokens=False)['input_ids']
    if len(checked) != 1:
        raise ValueError('Tokenizer must encode each action token as one id')
    act_tok = OneActionTokenizer(tokenizer_type='qwen2_vl')
    console = Console()
    root.mkdir(parents=True, exist_ok=True)
    summary = {'model': args.model_name, 'base_seed': args.seed, 'hist_mode': args.hist_mode,
               'repeats': args.repeats, 'qa_pool': args.qa_pool,
               'protocol': 'shared_minecraft_eval_v1', 'tasks': {}}
    with requests.Session() as session:
        session.trust_env = False
        for task in tasks:
            cfg = load_task_cfg(task)
            max_frames = args.max_frames if args.max_frames is not None else TASKS[task]
            task_dir = root / task.replace('/', '_')
            task_dir.mkdir(parents=True, exist_ok=True)
            env, callbacks = build_shared_env(cfg, task_dir / 'ep0', args.seed)
            record = next(callback for callback in callbacks if isinstance(callback, RecordCallback))
            episodes = []
            try:
                for episode in range(args.repeats):
                    ep_dir = task_dir / ('ep%d' % episode)
                    ep_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        ok, steps = run_episode(args, env, record, cfg, task, episode, str(ep_dir),
                                                max_frames, tkz, act_tok, session, console, pool)
                    except Exception as error:
                        if not (ep_dir / 'result.json').exists():
                            with (ep_dir / 'result.json').open('w') as file:
                                json.dump({'ok': False, 'steps': -1, 'episode': episode,
                                           'err': '%s: %s' % (type(error).__name__, error)}, file)
                        raise
                    episodes.append({'episode': episode, 'ok': bool(ok), 'steps': steps,
                                     'seed': args.seed + episode})
            finally:
                env.close()
            summary['tasks'][task] = {'successes': sum(item['ok'] for item in episodes),
                                     'total': len(episodes), 'max_frames': max_frames,
                                     'episodes': episodes}
            print('%s: %d/%d' % (task, summary['tasks'][task]['successes'], len(episodes)), flush=True)
    summary['successes'] = sum(item['successes'] for item in summary['tasks'].values())
    summary['total'] = sum(item['total'] for item in summary['tasks'].values())
    if summary['total'] != len(tasks) * args.repeats:
        raise ValueError('Incomplete duplex evaluation')
    with (task_dir / 'summary.json').open('x') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print('DUPLEX_EVAL_COMPLETE %d/%d' % (summary['successes'], summary['total']), flush=True)


if __name__ == '__main__':
    main()

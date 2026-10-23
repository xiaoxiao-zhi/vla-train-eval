import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


from duplex.eval_server_urls import URLSlots, parse_base_urls

ENTRY = Path(__file__).with_name('duplex_interact_v4.py')


def task_command(args, task, base_url=None):
    command = [sys.executable, str(ENTRY), '--task', task,
               '--base-url', base_url or args.base_url, '--base-urls', '', '--model-name', args.model_name,
               '--tokenizer', args.tokenizer, '--hist-mode', args.hist_mode,
               '--repeats', str(args.repeats), '--seed', str(args.seed),
               '--qa-pool', args.qa_pool, '--out-dir', args.out_dir,
               '--official-url', args.official_url, '--official-name', args.official_name]
    if args.max_frames is not None:
        command.extend(['--max-frames', str(args.max_frames)])
    return command


def collect_results(args, tasks):
    root = Path(args.out_dir)
    summary = {'model': args.model_name, 'base_seed': args.seed, 'hist_mode': args.hist_mode,
               'repeats': args.repeats, 'qa_pool': args.qa_pool,
               'protocol': 'shared_minecraft_eval_v1', 'workers': args.workers,
               'stagger_seconds': args.stagger,
               'scheduling': 'parallel task processes; serial episodes sharing one env per task',
               'tasks': {}}
    for task, budget in tasks.items():
        directory = root / task.replace('/', '_')
        expected_frames = args.max_frames if args.max_frames is not None else budget
        child = json.loads((directory / 'summary.json').read_text())
        for key in ('model', 'base_seed', 'hist_mode', 'repeats', 'qa_pool', 'protocol'):
            if child[key] != summary[key]:
                raise ValueError('Task summary configuration differs: %s %s' % (task, key))
        if set(child['tasks']) != {task} or child['total'] != args.repeats:
            raise ValueError('Incomplete single-task summary: %s' % task)
        episodes = []
        for episode in range(args.repeats):
            folder = directory / ('ep%d' % episode)
            result = json.loads((folder / 'result.json').read_text())
            environment = json.loads((folder / 'environment.json').read_text())
            if 'err' in result or result.get('steps', -1) < 0 or not isinstance(result.get('ok'), bool):
                raise ValueError('Invalid episode result: %s' % folder)
            if result['episode'] != episode or result['max_frames'] != expected_frames:
                raise ValueError('Episode index or budget differs: %s' % folder)
            seed = args.seed + episode
            if result['seed'] != seed or environment['python_seed'] != seed or environment['numpy_seed'] != seed:
                raise ValueError('Episode seed differs: %s' % folder)
            episodes.append({'episode': episode, 'ok': result['ok'], 'steps': result['steps'], 'seed': seed})
        row = {'successes': sum(item['ok'] for item in episodes), 'total': len(episodes),
               'max_frames': expected_frames, 'episodes': episodes}
        if child['tasks'][task] != row or child['successes'] != row['successes']:
            raise ValueError('Task summary does not match actual episode results: %s' % task)
        summary['tasks'][task] = row
    summary['successes'] = sum(row['successes'] for row in summary['tasks'].values())
    summary['total'] = sum(row['total'] for row in summary['tasks'].values())
    if summary['total'] != len(tasks) * args.repeats:
        raise ValueError('Incomplete duplex evaluation')
    with (root / 'summary.json').open('x') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def stop_owned_processes(running):
    for process, _, log in running:
        try:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=30)
        finally:
            log.close()


def run_tasks(args, tasks):
    root = Path(args.out_dir)
    if (root / 'summary.json').exists():
        raise FileExistsError('Existing duplex summary will not be overwritten')
    for task in tasks:
        directory = root / task.replace('/', '_')
        if (directory / 'summary.json').exists():
            raise FileExistsError('Existing task summary will not be overwritten')
        if any((directory / ('ep%d' % ep)).exists() and any((directory / ('ep%d' % ep)).iterdir())
               for ep in range(args.repeats)):
            raise FileExistsError('Existing episode results will not be overwritten')
    logs = root / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    running = []
    next_launch = 0.0
    urls = parse_base_urls(args.base_urls)
    if urls:
        args.workers = min(args.workers, len(urls))
    slots = URLSlots(urls, args.workers)
    assignments = {}
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        while pending or running:
            while pending and len(running) < args.workers and (not running or time.monotonic() >= next_launch):
                task = pending.pop(0)
                log = (logs / ('%s_group.log' % task.replace('/', '_'))).open('a')
                base_url = slots.acquire(task) or args.base_url
                assignments[task] = base_url
                command = task_command(args, task, base_url)
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               env=dict(os.environ, DISPLAY=os.environ.get('DISPLAY', ':99')),
                                               start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                running.append((process, task, log))
                next_launch = time.monotonic() + args.stagger
                (logs / 'task_servers.json').write_text(json.dumps(assignments, indent=2))
                print('DUPLEX_TASK_START %s pid=%d url=%s episodes=0..%d' %
                      (task, process.pid, base_url, args.repeats - 1), flush=True)
            for item in running[:]:
                process, task, log = item
                result = process.poll()
                if result is None:
                    continue
                running.remove(item)
                slots.release(task)
                log.close()
                print('DUPLEX_TASK_EXIT %s rc=%d' % (task, result), flush=True)
                if result != 0:
                    raise subprocess.CalledProcessError(result, process.args)
            if pending or running:
                time.sleep(1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        stop_owned_processes(running)
    summary = collect_results(args, tasks)
    print('DUPLEX_EVAL_COMPLETE %d/%d' % (summary['successes'], summary['total']), flush=True)
    return summary

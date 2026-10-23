#!/usr/bin/env python3
"""清理 GPU 机上的 vLLM 评测服务进程 (校验 cmdline 后按进程组 SIGTERM->SIGKILL)."""
import os
import signal
import sys
import time


def iter_vllm_pids():
    pids = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open('/proc/%s/cmdline' % entry, 'rb') as file:
                args = file.read().decode('utf-8', 'replace').split('\x00')
        except OSError:
            continue
        if any(a.endswith('vllm') or a.endswith('vllm.exe') for a in args[:2]) and 'serve' in args:
            pids.append((int(entry), args))
    return pids


def kill_all(pids, sig):
    for pid, args in pids:
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            os.kill(pid, sig)


def main():
    pids = iter_vllm_pids()
    if not pids:
        print('NO_VLLM_PROCESS_FOUND')
        return
    for pid, args in pids:
        print('TARGET', pid, ' '.join(a for a in args if a)[:160])
    kill_all(pids, signal.SIGTERM)
    for _ in range(10):
        time.sleep(2)
        alive = [(pid, args) for pid, args in pids if os.path.exists('/proc/%d' % pid)]
        if not alive:
            break
    alive = [(pid, args) for pid, args in pids if os.path.exists('/proc/%d' % pid)]
    if alive:
        for pid, args in alive:
            print('SIGKILL', pid)
        kill_all(alive, signal.SIGKILL)
        time.sleep(5)
    remaining = iter_vllm_pids()
    print('REMAINING_VLLM', [pid for pid, _ in remaining])
    sys.exit(1 if remaining else 0)


if __name__ == '__main__':
    main()

import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


def service_ready(url, name):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + '/models', timeout=3) as response:
            return any(model['id'] == name for model in json.load(response)['data'])
    except (OSError, ValueError, KeyError):
        return False


def terminate_owned(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)


def run_pool(args):
    for gpu in args.gpus:
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits', '-i', str(gpu)], text=True)
        if int(used.strip()) >= 2000:
            raise RuntimeError('GPU %d occupied; refusing to stop another task' % gpu)
    for port in args.ports:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    processes, handles, servers = [], [], []
    urls = ['http://127.0.0.1:%d/v1' % port for port in args.ports]
    try:
        for gpu, port, url in zip(args.gpus, args.ports, urls):
            command = [args.vllm, 'serve', args.model, '--runner', 'generate',
                       '--tensor-parallel-size', '1', '--data-parallel-size', '1',
                       '--host', '127.0.0.1', '--port', str(port),
                       '--served-model-name', args.model_name, '--max-model-len', '8192',
                       '--gpu-memory-utilization', '0.85', '--trust-remote-code',
                       '--limit-mm-per-prompt', '{"image": 8}']
            handle = (logs / ('vllm_gpu%d_port%d.log' % (gpu, port))).open('a')
            handles.append(handle)
            process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            servers.append({'gpu': gpu, 'url': url, 'pid': process.pid, 'tp': 1, 'dp': 1})
            print('INDEPENDENT_VLLM_START', servers[-1], flush=True)
        (logs / 'servers.json').write_text(json.dumps(servers, indent=2))
        deadline, ready = time.monotonic() + args.timeout, set()
        while len(ready) != len(urls):
            for process, url in zip(processes, urls):
                if process.poll() is not None:
                    raise RuntimeError('vLLM exited before readiness: %s' % url)
                if url not in ready and service_ready(url, args.model_name):
                    ready.add(url)
            if len(ready) == len(urls):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Some independent servers did not become ready')
            time.sleep(1)
        print('ALL_INDEPENDENT_VLLM_READY', urls, flush=True)
        (logs / 'base_urls.json').write_text(json.dumps(urls, indent=2))
        print('BASE_URLS=' + ','.join(urls), flush=True)
        return urls
    except BaseException:
        terminate_owned(processes)
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'model-name', 'gpus', 'ports', 'logs'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--vllm', default='/opt/conda/envs/torch-base/bin/vllm')
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    args.gpus = [int(value) for value in args.gpus.split(',')]
    args.ports = [int(value) for value in args.ports.split(',')]
    if (not args.gpus or len(args.gpus) != len(args.ports)
            or len(set(args.gpus)) != len(args.gpus) or len(set(args.ports)) != len(args.ports)
            or any(gpu < 0 for gpu in args.gpus) or any(not 1 <= port <= 65535 for port in args.ports)
            or args.timeout <= 0):
        parser.error('Invalid GPU/port mapping or timeout')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run_pool(args)


if __name__ == '__main__':
    main()

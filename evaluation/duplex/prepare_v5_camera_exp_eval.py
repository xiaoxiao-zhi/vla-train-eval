import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import json
from pathlib import Path

from safetensors import safe_open

from duplex.prepare_duplex500_eval import prepare_view

RUN = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/'
           'checkpoint/v5_ce_camera_exponential/run-20261016-224410-XIBAET/duplex_camera_exponential')
STEP = 500
# 动作token 编号178-250 (151835-151907) + v4vocab 扩展三token (silence/obs_user/obs_asst)
TOKENS = {f'<|reserved_special_token_{178 + i}|>': 151835 + i for i in range(73)}
TOKENS.update({'<|silence|>': 151908, '<|obs_user|>': 151909, '<|obs_asst|>': 151910})


def locate_final_checkpoint(run, step):
    candidates = list(run.glob(f'checkpoint-{step}')) + list(run.glob(f'v0-*/checkpoint-{step}'))
    if len(candidates) > 1:
        raise ValueError('Multiple final checkpoints found; refusing to guess')
    if not candidates:
        return None
    checkpoint = candidates[0]
    required = ('trainer_state.json', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
                'model.safetensors.index.json')
    if not all((checkpoint / name).is_file() for name in required):
        return None
    if not any((checkpoint / name).is_file() for name in ('preprocessor_config.json', 'processor_config.json')):
        return None
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if state['global_step'] != step or state['max_steps'] != step:
        raise ValueError('Checkpoint is not the completed final step')
    weight_map = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    shards = set(weight_map.values())
    if len(shards) != 4 or not all((checkpoint / name).is_file() for name in shards):
        raise ValueError('Final checkpoint is missing model shards')
    for name in shards:
        with safe_open(checkpoint / name, framework='pt', device='cpu') as file:
            keys = set(file.keys())
            expected = {key for key, shard in weight_map.items() if shard == name}
            if keys != expected:
                raise ValueError('Checkpoint shard contents do not match the index')
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--step', type=int, default=STEP)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    checkpoint = locate_final_checkpoint(args.run, args.step)
    if checkpoint is None:
        raise SystemExit(2)
    if args.out is None:
        print(checkpoint, flush=True)
    else:
        prepare_view(checkpoint, args.out, token_ids=TOKENS)


if __name__ == '__main__':
    main()

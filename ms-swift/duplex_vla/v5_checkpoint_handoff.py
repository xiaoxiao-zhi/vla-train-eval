#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

BASE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B')
EXPAND = Path(__file__).with_name('expand_ckpt_vocab_v4.py')
NEW_IDS = {'<|silence|>': 151908, '<|obs_user|>': 151909, '<|obs_asst|>': 151910}
CHECKPOINT = re.compile(r'checkpoint-([0-9]+)$')


def locate(root):
    root = root.resolve(strict=True)
    candidates = set()
    for pattern in ('checkpoint-*', 'v0-*/checkpoint-*', 'run-*/checkpoint-*', 'run-*/v0-*/checkpoint-*'):
        candidates.update(path for path in root.glob(pattern) if path.is_dir() and CHECKPOINT.fullmatch(path.name))
    if not candidates:
        raise FileNotFoundError(f'No numeric checkpoint directory within this training output: {root}')
    run_dirs = {path.parent for path in candidates}
    if len(run_dirs) != 1:
        raise ValueError(f'Ambiguous training outputs; supply the exact current stage directory: {sorted(map(str, run_dirs))}')
    checkpoint = max(candidates, key=lambda path: int(CHECKPOINT.fullmatch(path.name).group(1)))
    if not (checkpoint / 'config.json').is_file():
        raise FileNotFoundError(checkpoint / 'config.json')
    if not ((checkpoint / 'model.safetensors').is_file() or (checkpoint / 'model.safetensors.index.json').is_file()):
        raise FileNotFoundError(f'No model weights in selected checkpoint: {checkpoint}')
    return checkpoint


def token_map(directory):
    data = json.loads((directory / 'tokenizer.json').read_text())
    mapping = dict(data['model']['vocab'])
    for token in data['added_tokens']:
        content, token_id = token['content'], token['id']
        if content in mapping and mapping[content] != token_id:
            raise ValueError(f'Inconsistent tokenizer mapping: {content}')
        mapping[content] = token_id
    return mapping


def verify(directory, expanded):
    reference = token_map(BASE)
    actual = token_map(directory)
    if len(reference) != 151908 or set(NEW_IDS).intersection(reference):
        raise ValueError('Unexpected official base tokenizer')
    expected = {**reference, **NEW_IDS} if expanded else reference
    if actual != expected:
        raise ValueError(f'Tokenizer mapping differs from expected {"duplex" if expanded else "stage2"} vocabulary: {directory}')
    config = json.loads((directory / 'config.json').read_text())
    if expanded:
        configs = [config]
        if isinstance(config.get('text_config'), dict):
            configs.append(config['text_config'])
        if any(cfg.get('vocab_size') != 151911 for cfg in configs):
            raise ValueError(f'Expanded config vocabulary is not 151911: {directory}')
        if config.get('rope_scaling', {}).get('mrope_section') != [16, 24, 24]:
            raise ValueError(f'Missing expanded mRoPE configuration: {directory}')
        from safetensors import safe_open
        index_file = directory / 'model.safetensors.index.json'
        weight_map = json.loads(index_file.read_text())['weight_map'] if index_file.is_file() else None
        verified_keys = set()
        for key in ('model.language_model.embed_tokens.weight', 'model.embed_tokens.weight', 'lm_head.weight'):
            if weight_map is not None and key not in weight_map:
                continue
            weight_file = directory / (weight_map[key] if weight_map is not None else 'model.safetensors')
            with safe_open(str(weight_file), framework='pt', device='cpu') as handle:
                if key not in handle.keys():
                    continue
                if handle.get_slice(key).get_shape()[0] != 151911:
                    raise ValueError(f'Expanded weight rows mismatch: {key}')
                verified_keys.add(key)
        if 'lm_head.weight' not in verified_keys or not any('embed_tokens.weight' in key for key in verified_keys):
            raise ValueError('Expanded embedding/lm_head weight keys missing')
    return {'vocab_size': len(actual), 'new_token_ids': NEW_IDS if expanded else {}}


def prepare(source, destination, loss):
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if not CHECKPOINT.fullmatch(source.name):
        raise ValueError(f'Expected a numeric original stage2 checkpoint: {source}')
    if destination.exists():
        raise FileExistsError(f'Expansion destination already exists: {destination}')
    verify(source, False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'EXPAND_SOURCE={source}\nEXPAND_DESTINATION={destination}', flush=True)
    subprocess.run([sys.executable, str(EXPAND), str(source), str(destination)], check=True)
    info = verify(destination, True)
    info.update({'stage2_checkpoint': str(source), 'prepared_model': str(destination),
                 'loss_type': loss or None, 'expansion_script': str(EXPAND),
                 'optimizer_scheduler_policy': 'Only model weights passed to next stage; no resume_from_checkpoint.'})
    with (destination / 'duplex_training_source.json').open('x') as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    locating = subparsers.add_parser('locate')
    locating.add_argument('root', type=Path)
    preparing = subparsers.add_parser('prepare')
    preparing.add_argument('source', type=Path)
    preparing.add_argument('destination', type=Path)
    preparing.add_argument('--loss-type', default='')
    checking = subparsers.add_parser('verify')
    checking.add_argument('model', type=Path)
    checking.add_argument('--expanded', action='store_true')
    args = parser.parse_args()
    if args.command == 'locate':
        print(locate(args.root))
    elif args.command == 'prepare':
        prepare(args.source, args.destination, args.loss_type)
    else:
        print(json.dumps(verify(args.model, args.expanded), ensure_ascii=False))


if __name__ == '__main__':
    main()

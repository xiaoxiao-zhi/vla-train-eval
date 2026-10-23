import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import json
from pathlib import Path

from safetensors import safe_open
from tokenizers import Tokenizer

RUN = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/duplex_v3_500traj_ce/training/run-20261015-164152-TKsPJA')
TOKENS = {'<|silence|>': 151908, '<|obs_user|>': 151909, '<|obs_asst|>': 151910}


def locate_saved_checkpoint(run=RUN):
    candidates = list(run.glob('checkpoint-243')) + list(run.glob('v0-*/checkpoint-243'))
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
    if state['global_step'] != 243 or state['max_steps'] != 243:
        raise ValueError('Checkpoint is not the completed duplex stage')
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


def prepare_view(checkpoint, target, token_ids=None):
    required_tokens = TOKENS if token_ids is None else token_ids
    if target.exists():
        raise FileExistsError('Model view already exists')
    config = json.loads((checkpoint / 'config.json').read_text())
    tokenizer = json.loads((checkpoint / 'tokenizer_config.json').read_text())
    backend = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
    if any(backend.encode(token, add_special_tokens=False).ids != [idx] for token, idx in required_tokens.items()):
        raise ValueError('Final tokenizer has unexpected special token IDs')
    patched = []
    extra_tokens = tokenizer.get('extra_special_tokens')
    if isinstance(extra_tokens, list):
        tokenizer['additional_special_tokens'] = list(dict.fromkeys(
            tokenizer.get('additional_special_tokens', []) + extra_tokens))
        del tokenizer['extra_special_tokens']
        patched.append('tokenizer.extra_special_tokens_list_to_additional_special_tokens')
    text_config = config.get('text_config', config)
    rope = text_config.get('rope_parameters')
    if rope:
        scaling = {key: value for key, value in rope.items() if key != 'rope_theta'}
        if 'rope_scaling' not in config:
            config['rope_scaling'] = scaling
            patched.append('config.rope_scaling')
        if 'rope_scaling' not in text_config:
            text_config['rope_scaling'] = scaling
            patched.append('config.text_config.rope_scaling')
    combined_path = checkpoint / 'processor_config.json'
    combined = json.loads(combined_path.read_text()) if combined_path.exists() else None
    image_path = checkpoint / 'preprocessor_config.json'
    if image_path.exists():
        processor = json.loads(image_path.read_text())
    elif combined is not None and 'image_processor' in combined:
        processor = dict(combined['image_processor'])
        processor['processor_class'] = combined['processor_class']
        patched.append('preprocessor_from_combined_processor_config')
    else:
        raise ValueError('No recognized image processor configuration')
    size = processor.get('size', {})
    for key, size_key in (('min_pixels', 'shortest_edge'), ('max_pixels', 'longest_edge')):
        if key not in processor and size_key in size:
            processor[key] = size[size_key]
            patched.append('preprocessor.' + key)
    template_path = checkpoint / 'chat_template.jinja'
    if not tokenizer.get('chat_template') and template_path.exists():
        tokenizer['chat_template'] = template_path.read_text()
        patched.append('tokenizer.chat_template')
    if not tokenizer.get('chat_template'):
        raise ValueError('Final checkpoint has no usable chat template')
    target.mkdir(parents=True)
    copied = {'config.json', 'preprocessor_config.json', 'processor_config.json', 'tokenizer_config.json'}
    for path in checkpoint.iterdir():
        if not path.is_file() or path.name in copied:
            continue
        if path.suffix == '.safetensors' or path.name in (
            'model.safetensors.index.json', 'tokenizer.json', 'vocab.json', 'merges.txt',
            'generation_config.json', 'special_tokens_map.json', 'added_tokens.json',
            'chat_template.jinja', 'processor_config.json', 'video_preprocessor_config.json',
        ):
            (target / path.name).symlink_to(path)
    for name, content in (('config.json', config), ('preprocessor_config.json', processor),
                          ('tokenizer_config.json', tokenizer)):
        (target / name).write_text(json.dumps(content, ensure_ascii=False, indent=2))
    if combined is not None:
        combined['image_processor'] = {key: value for key, value in processor.items() if key != 'processor_class'}
        (target / 'processor_config.json').write_text(json.dumps(combined, ensure_ascii=False, indent=2))
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    manifest = {'source_checkpoint': str(checkpoint), 'global_step': state['global_step'],
                'config_patch_keys': patched, 'weights_unchanged': True,
                'required_token_ids': required_tokens}
    if required_tokens == TOKENS:
        manifest['duplex_token_ids'] = TOKENS
    (target / 'eval_source.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('MODEL_VIEW_READY', target, 'SOURCE', checkpoint, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    checkpoint = locate_saved_checkpoint(args.run)
    if checkpoint is None:
        raise SystemExit(2)
    if args.out is None:
        print(checkpoint, flush=True)
    else:
        prepare_view(checkpoint, args.out)


if __name__ == '__main__':
    main()

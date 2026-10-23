#!/usr/bin/env python
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from swift.model import get_model_processor
from swift.template import get_template

NEW_TOKENS = ['<|silence|>', '<|obs_user|>', '<|obs_asst|>']


def tensor_part(directory, key, rows=None):
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    with safe_open(directory / index['weight_map'][key], framework='pt', device='cpu') as file:
        return file.get_tensor(key) if rows is None else file.get_slice(key)[rows]


def verify_weights(source, model):
    original = AutoTokenizer.from_pretrained(source)
    expanded = AutoTokenizer.from_pretrained(model)
    assert len(original) == 151908 and len(expanded) == 151911
    original_vocab, expanded_vocab = original.get_vocab(), expanded.get_vocab()
    assert all(expanded_vocab[token] == idx for token, idx in original_vocab.items())
    new_ids = {token: expanded.convert_tokens_to_ids(token) for token in NEW_TOKENS}
    assert list(new_ids.values()) == [151908, 151909, 151910]
    assert all(expanded(token, add_special_tokens=False)['input_ids'] == [idx] for token, idx in new_ids.items())
    source_index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    model_index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    assert source_index.keys() == model_index.keys()
    embedding = next(key for key in source_index if key.endswith('embed_tokens.weight'))
    head = next(key for key in source_index if key.endswith('lm_head.weight'))
    selected = [key for key in source_index if 'layers.0.' in key and key.endswith('q_proj.weight')]
    selected += [key for key in source_index if 'layers.27.' in key and key.endswith('q_proj.weight')]
    selected += [key for key in source_index if key.endswith('language_model.norm.weight')]
    assert len(selected) >= 2
    for key in selected:
        assert torch.equal(tensor_part(source, key), tensor_part(model, key)), key
    for key in (embedding, head):
        for start in (0, 151640, 151890):
            assert torch.equal(tensor_part(source, key, slice(start, start + 16)),
                               tensor_part(model, key, slice(start, start + 16))), key
    config = json.loads((model / 'config.json').read_text())
    assert config['vocab_size'] == config['text_config']['vocab_size'] == 151911
    assert config['text_config']['rope_parameters']['mrope_section'] == [16, 24, 24]
    return new_ids, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    new_ids, checked_keys = verify_weights(args.source, args.model)
    stats = json.loads(Path(str(args.data) + '.stats.json').read_text())
    assert stats['selected_trajectories'] == stats['qa_pairs'] == 500
    assert all(stats['verification'].values())
    digest = hashlib.sha256()
    count, longest, speaking, silent = 0, None, None, None
    with args.data.open('rb') as file:
        for line in file:
            digest.update(line)
            row = json.loads(line)
            count += 1
            if longest is None or len(row['messages'][0]['content']) > len(longest['messages'][0]['content']):
                longest = row
            if row['messages'][1]['content'].rstrip().endswith('<|silence|>'):
                silent = silent or row
            else:
                speaking = speaking or row
    assert count == stats['filtered']['frames'] == 62118 and digest.hexdigest() == stats['sha256']
    assert speaking is not None and silent is not None
    model, processor = get_model_processor(str(args.model), model_type='qwen2_vl',
                                          torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_impl='sdpa')
    model.config.use_cache = False
    template = get_template(processor, template_type='qwen2_vl', max_length=8192,
                            truncation_strategy='raise', loss_scale='last_round')
    template.set_mode('train')
    template.model = model
    template.register_post_encode_hook([model])
    encoded, lengths = [], []
    for kind, row in [('silence', silent), ('speaking', speaking), ('longest_text', longest)]:
        inputs = {'messages': row['messages'], 'images': row['images']}
        entry = template.encode(inputs)
        labels = entry['labels']
        positions = [index for index, value in enumerate(labels) if value != -100]
        assert positions and labels[positions[0]] == processor.tokenizer.convert_tokens_to_ids('<|reserved_special_token_178|>')
        assert all(value == -100 for value in labels[:positions[0]])
        assert len(entry['input_ids']) <= 8192
        lengths.append({'kind': kind, 'base': row['base'], 't': row['t'], 'tokens': len(entry['input_ids']),
                        'supervised_tokens': len(positions), 'images': len(row['images'])})
        encoded.append(entry)
    model.train()
    model.gradient_checkpointing_enable()
    for name, parameter in model.named_parameters():
        if 'visual.' in name:
            parameter.requires_grad_(False)
    batch = template.data_collator(encoded[:2])
    batch.pop('loss_scale', None)
    batch = {key: value.to('cuda:0') if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    output = model(**batch)
    assert output.loss is not None and torch.isfinite(output.loss)
    loss = float(output.loss.detach())
    output.loss.backward()
    gradients = model.get_output_embeddings().weight.grad
    assert gradients is not None and torch.isfinite(gradients).all()
    grad_norm = float(gradients.norm())
    assert grad_norm > 0
    model.zero_grad(set_to_none=True)
    template.remove_post_encode_hook()
    result = {'source_checkpoint': str(args.source), 'prepared_model': str(args.model), 'dataset': str(args.data),
              'dataset_sha256': stats['sha256'], 'dataset_frames': count, 'global_batch': 256,
              'max_steps': (count + 255) // 256, 'loss_type': None, 'camera_weighting': False,
              'loss_scale': 'last_round', 'new_tokens': new_ids, 'original_vocab_mapping_unchanged': True,
              'checked_unchanged_weight_keys': checked_keys, 'existing_embedding_head_rows_checked': True,
              'samples': lengths, 'builtin_ce_forward_loss': loss, 'lm_head_gradient_norm': grad_norm,
              'forward_backward_passed': True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print('PREFLIGHT_OK', json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

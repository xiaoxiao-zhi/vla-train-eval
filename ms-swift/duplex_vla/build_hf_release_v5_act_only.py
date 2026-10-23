#!/usr/bin/env python3
"""构建 v5_act_only 的 HF 发布目录.

复用 prepare_v5_split_eval 的定位校验与 prepare_view 补丁逻辑, 物化真实文件,
仅保留推理所需文件 (剔除 trainer_state/optimizer/rng/eval_source 等训练态产物), 附 README.
"""
import sys as _sys
for _p in ('/apdcephfs/private_chengzhixu/evaluation', '/apdcephfs/private_chengzhixu/JarvisVLA',
           '/apdcephfs/private_chengzhixu/ms-swift/duplex_vla'):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import json
import shutil
from pathlib import Path

from prepare_v5_split_eval import TOKENS, locate_final_checkpoint
from duplex.prepare_duplex500_eval import prepare_view

SHARE = Path('/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu')
RUN = SHARE / 'checkpoint/v5_split_action_fa/run-20261020-194239-O4DmiI/action_only_ce_flash_attn'
STEP = 1162
STAGING = SHARE / 'checkpoint/v5_split_action_fa/hf_release/_staging_v5_act_only'
RELEASE = SHARE / 'checkpoint/v5_split_action_fa/hf_release/v5_act_only'
KEEP = {
    'config.json', 'preprocessor_config.json', 'processor_config.json', 'tokenizer_config.json',
    'tokenizer.json', 'chat_template.jinja', 'generation_config.json',
    'model.safetensors.index.json',
}

README = '''---
license: apache-2.0
base_model: JarvisVLA-Qwen2-VL-7B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- minecraft
- vision-language-action
- vla
---

# v5_act_only

Action-only ablation model of the JarvisVLA (Minecraft VLA) v5 pipeline, trained with **flash-attn**.

- **Base model**: JarvisVLA-Qwen2-VL-7B (Qwen2-VL 7B backbone, full-parameter tuning with ViT and aligner frozen)
- **Training data**: the action (VLA) split of the v5 stage2 dataset — 297,270 frames from 4,706 complete Minecraft
  trajectories (mine_block / kill_entity / craft_item), single-turn three-frame flat format, no text-QA rows
- **Loss**: ordinary cross-entropy on the current assistant turn (action tokens), loss_scale=last_round
- **Hyperparameters**: 1,162 steps (~1 epoch), global batch 256 (8 GPUs x 2 x 16), lr 1e-5 cosine, warmup 50,
  bf16, DeepSpeed ZeRO2, max_length 8192, flash-attn 2.8.3 attention
- **Inputs**: instruction + last 2 frames + current frame (3 images); **Outputs**: reserved action tokens
  (ids 151835-151907) that decode to Minecraft keyboard/camera actions

Intended for research on vision-language-action models and action-only vs mixed action+QA training ablations.
'''


def main():
    checkpoint = locate_final_checkpoint(RUN, STEP)
    if checkpoint is None:
        raise SystemExit('final checkpoint not found')
    if STAGING.exists() or RELEASE.exists():
        raise FileExistsError('staging or release dir already exists')
    prepare_view(checkpoint, STAGING, token_ids=TOKENS)
    RELEASE.mkdir(parents=True)
    for path in STAGING.iterdir():
        if path.name == 'eval_source.json':
            continue
        if path.name in KEEP or path.suffix == '.safetensors':
            shutil.copy2(path, RELEASE / path.name)
    shutil.rmtree(STAGING)
    (RELEASE / 'README.md').write_text(README)
    files = sorted(p.name for p in RELEASE.iterdir())
    total = sum(p.stat().st_size for p in RELEASE.iterdir())
    manifest = {
        'source_checkpoint': str(checkpoint), 'global_step': STEP, 'attn_impl': 'flash_attn',
        'files': files, 'total_bytes': total,
        'excluded': ['trainer_state.json', 'optimizer/scheduler/rng states', 'training_args.bin',
                     'zero_to_fp32.py', 'args.json', 'logging.jsonl', 'eval_source.json'],
    }
    (RELEASE / 'release_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('HF_RELEASE_READY', RELEASE)
    print('files:', len(files), 'total_GB: %.1f' % (total / 1e9))
    for name in files:
        print(' -', name)


if __name__ == '__main__':
    main()

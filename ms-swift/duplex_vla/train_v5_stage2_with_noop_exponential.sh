#!/bin/bash
set -uo pipefail
set +e

ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
BASE=$SHARE/HF_MODEL/JarvisVLA-Qwen2-VL-7B
DATA=$SHARE/dataset/duplex_vla_v5/stage2_with_noop_300k.jsonl
TRAIN_SH=$ROOT/ms-swift/duplex_vla/train_v2_stage.sh
PREPARE_PY=$SHARE/conda_envs/JarvisVLA/bin/python
TRAIN_PY=$SHARE/conda_envs/swift/bin/python
OUTPUT_ROOT=${V5WN_OUTPUT_ROOT:-$SHARE/checkpoint/v5_stage2_with_noop_camera_exponential}
STEPS=${V5WN_STEPS:-1935}
LR=${V5WN_LR:-1e-5}

export REPORT_TO=wandb
export WANDB_PROJECT=v4-pipeline
export WANDB_MODE=online
export MCVLA_REPORT_TOKEN_TYPE_ACC=1
export CAMERA_EXP_ALPHA=1.0
export CAMERA_EXP_TAU=4.0
export CAMERA_EXP_NORMALIZATION=global
export CAMERA_LOG_BATCH_WEIGHTS=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
unset WANDB_RUN_ID WANDB_RESUME

for file in "$BASE/config.json" "$BASE/tokenizer.json" "$DATA" "$DATA.stats.json" "$TRAIN_SH" "$PREPARE_PY" "$TRAIN_PY"; do
    [ -f "$file" ] || { echo "Missing required input: $file"; exit 1; }
done

"$PREPARE_PY" - "$DATA" "$STEPS" <<'PYEOF'
import json
import sys
from pathlib import Path
data, steps = sys.argv[1:3]
s = json.loads(Path(data + '.stats.json').read_text())
if s['total_after'] != 495345 or s['vla_after'] != 299975 or s['qa'] != 195370:
    raise ValueError('Unexpected stage2_with_noop_300k composition')
if int(steps) <= 0:
    raise ValueError('Training steps must be positive')
print(f'V5WN_DATA_READY rows={s["total_after"]} vla={s["vla_after"]} qa={s["qa"]} noop_pct={s["noop_pct_after"]} steps={steps}', flush=True)
PYEOF
[ "$?" -eq 0 ] || exit 1

mkdir -p "$OUTPUT_ROOT" || exit 1
exec 8>"$OUTPUT_ROOT/train.lock"
flock -n 8 || { echo 'Another camera_exponential training is using this output root'; exit 1; }
OUT=$(mktemp -d "$OUTPUT_ROOT/run-$(date +%Y%m%d-%H%M%S)-XXXXXX") || exit 1
echo "V5WN_TRAIN_OUTPUT=$OUT"
exec > >(tee -a "$OUT/launch.log") 2>&1
export CAMERA_WEIGHT_LOG_DIR="$OUT"
export WANDB_NAME="v5-stage2_with_noop_camera_exponential-$(basename "$OUT")"
export WANDB_RUN_GROUP="v5wn-$(basename "$OUT")"
cd "$ROOT/ms-swift" || exit 1

"$PREPARE_PY" - "$OUT" "$BASE" "$DATA" "$STEPS" "$LR" <<'PYEOF'
import hashlib
import json
import os
import sys
from pathlib import Path
out, model, data, steps, lr = sys.argv[1:]
paths = [model + '/config.json', data, data + '.stats.json',
         '/apdcephfs/private_chengzhixu/ms-swift/mcvla/camera_distance_loss.py',
         '/apdcephfs/private_chengzhixu/ms-swift/mcvla/camera_exponential_loss.py',
         '/apdcephfs/private_chengzhixu/ms-swift/mcvla/run_train.py',
         '/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/train_v2_stage.sh',
         '/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/train_v5_stage2_with_noop_exponential.sh']
metadata = {
    'model': model, 'dataset': data, 'output': out,
    'max_steps': int(steps), 'learning_rate': float(lr),
    'loss_type': 'camera_exponential', 'loss_scale': 'last_round', 'global_batch_size': 256,
    'camera_weighting': {
        'top_k': 5, 'candidate_space': 'full_vocabulary', 'normalize_top5_probabilities': True,
        'invalid_or_cross_axis_distance': 20, 'distance': 'expected_absolute_bin_difference',
        'raw_weight': '1 + alpha * (1 - exp(-d / tau))',
        'alpha': float(os.environ['CAMERA_EXP_ALPHA']), 'tau': float(os.environ['CAMERA_EXP_TAU']),
        'normalization': os.environ['CAMERA_EXP_NORMALIZATION'],
        'normalization_population': 'effective_camera_x_and_y_per_forward',
        'normalize_across_gradient_accumulation': False, 'detach_weights': True,
        'batch_weight_logging': False,
    },
    'dataset_note': 'V5 stage2 with no-ops restored then jointly subsampled by complete segment to ~300k VLA rows; no additional no-op filtering.',
    'queued_after': 'v5_ce_camera_exponential four-phase pipeline process exit; no exit-status or trainer-state check performed on the prior pipeline.',
    'report_to': os.environ['REPORT_TO'], 'wandb_project': os.environ['WANDB_PROJECT'],
    'wandb_name': os.environ['WANDB_NAME'], 'source_sha256': {},
}
for path in paths:
    metadata['source_sha256'][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
Path(out, 'experiment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
PYEOF
[ "$?" -eq 0 ] || exit 1

echo "TRAIN_START stage2_with_noop_camera_exponential $(date --iso-8601=seconds) model=$BASE steps=$STEPS lr=$LR"
bash "$TRAIN_SH" "$BASE" "$DATA" "$OUT" "$STEPS" "$LR" last_round camera_exponential
echo "TRAIN_PROCESS_EXITED stage2_with_noop_camera_exponential $(date --iso-8601=seconds); no exit-status or trainer-state check performed"
echo "V5WN_DONE outputs=$OUT"

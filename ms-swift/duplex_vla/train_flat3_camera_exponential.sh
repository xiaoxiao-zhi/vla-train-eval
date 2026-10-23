#!/bin/bash
# Usage: bash train_flat3_camera_exponential.sh [model] [dataset] [steps] [lr]
# TRAIN_LOSS_TYPE='' selects the original framework loss path.
set -euo pipefail

ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
MODEL=${1:-$SHARE/HF_MODEL/JarvisVLA-Qwen2-VL-7B}
DATA=${2:-$SHARE/dataset/duplex_vla_v4/stage2_train_flat3.jsonl}
DEFAULT_STEPS=2900
if [ "${DATA##*/}" = 'stage2_train_flat3_v3.jsonl' ]; then
    DEFAULT_STEPS=1936
fi
STEPS=${3:-$DEFAULT_STEPS}
LR=${4:-1e-5}
LOSS_TYPE=${TRAIN_LOSS_TYPE-camera_exponential}
LOSS_SCALE=${TRAIN_LOSS_SCALE:-last_round}
TRAIN_SH=$ROOT/ms-swift/duplex_vla/train_v2_stage.sh
PY=$SHARE/conda_envs/swift/bin/python

case "$LOSS_TYPE" in
    camera_exponential|camera_weighted|'') ;;
    *) echo "Unsupported TRAIN_LOSS_TYPE: $LOSS_TYPE"; exit 1 ;;
esac
export CAMERA_EXP_ALPHA=${CAMERA_EXP_ALPHA:-1.0}
export CAMERA_EXP_TAU=${CAMERA_EXP_TAU:-4.0}
export CAMERA_EXP_NORMALIZATION=${CAMERA_EXP_NORMALIZATION:-global}
export CAMERA_LOG_BATCH_WEIGHTS=${CAMERA_LOG_BATCH_WEIGHTS:-1}
export PYTHONUNBUFFERED=1
OUT_ROOT=${TRAIN_OUTPUT_ROOT:-$SHARE/checkpoint/s2v4_flat3_${LOSS_TYPE:-baseline}}

for file in "$MODEL/config.json" "$DATA" "$TRAIN_SH" "$PY"; do
    [ -f "$file" ] || { echo "Missing input: $file"; exit 1; }
done
if pgrep -f '[r]un_train.py|[v]llm serve|[g]pu_burn.py' >/dev/null; then
    echo 'Training/inference processes are still running; refusing to launch'
    exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$GPU_COUNT" -eq 8 ] || { echo "Expected 8 GPUs, found $GPU_COUNT"; exit 1; }
MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
[ "$MAX_USED" -lt 2000 ] || { echo "GPU memory still occupied: ${MAX_USED}MiB"; exit 1; }

mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/train.lock"
flock -n 9 || { echo 'Another launcher is using this output root'; exit 1; }
OUT=$(mktemp -d "$OUT_ROOT/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")
exec >>"$OUT/launch.log" 2>&1
export CAMERA_WEIGHT_LOG_DIR="$OUT"
export REPORT_TO=${REPORT_TO:-wandb}
export WANDB_PROJECT=${WANDB_PROJECT:-v4-pipeline}
export WANDB_NAME="flat3-${LOSS_TYPE:-baseline}-$(basename "$OUT")"
unset WANDB_RUN_ID
cd "$ROOT/ms-swift"

"$PY" - "$OUT" "$MODEL" "$DATA" "$STEPS" "$LR" "$LOSS_TYPE" "$LOSS_SCALE" <<'PYEOF'
import hashlib
import json
import os
import sys
from pathlib import Path
out, model, data, steps, lr, loss_type, loss_scale = sys.argv[1:]
metadata = {
    'model': model, 'dataset': data, 'max_steps': int(steps),
    'learning_rate': float(lr), 'loss_type': loss_type or None,
    'loss_scale': loss_scale, 'global_batch_size': 256,
    'camera_weighting': None,
    'source_sha256': {},
}
if loss_type == 'camera_exponential':
    metadata['camera_weighting'] = {
        'top_k': 5, 'candidate_space': 'full_vocabulary',
        'normalize_top5_probabilities': True, 'invalid_or_cross_axis_distance': 20,
        'distance': 'expected_absolute_bin_difference',
        'raw_weight': '1 + alpha * (1 - exp(-d / tau))',
        'alpha': float(os.environ['CAMERA_EXP_ALPHA']),
        'tau': float(os.environ['CAMERA_EXP_TAU']),
        'normalization': os.environ['CAMERA_EXP_NORMALIZATION'],
        'normalization_population': 'effective_camera_x_and_y_per_forward',
        'normalize_across_gradient_accumulation': False,
        'detach_weights': True,
        'batch_weight_logging': os.environ['CAMERA_LOG_BATCH_WEIGHTS'] != '0',
        'batch_weight_format': 'json_one_file_per_rank_per_microbatch',
        'batch_weight_directory': str(Path(out) / 'batch_weights'),
    }
for name in ('mcvla/camera_distance_loss.py', 'mcvla/camera_exponential_loss.py',
             'mcvla/run_train.py', 'duplex_vla/train_v2_stage.sh',
             'duplex_vla/train_flat3_camera_exponential.sh'):
    metadata['source_sha256'][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
Path(out, 'experiment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
PYEOF

bash "$TRAIN_SH" "$MODEL" "$DATA" "$OUT" "$STEPS" "$LR" "$LOSS_SCALE" "$LOSS_TYPE"

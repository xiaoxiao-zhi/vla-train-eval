#!/bin/bash
set -uo pipefail
set +e

ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
BASE=$SHARE/HF_MODEL/JarvisVLA-Qwen2-VL-7B
STAGE2_DATA=$SHARE/dataset/duplex_vla_v5/stage2.jsonl
DUPLEX_DATA=$SHARE/dataset/duplex_vla_v5/duplex.jsonl
TRAIN_SH=$ROOT/ms-swift/duplex_vla/train_v2_stage.sh
HANDOFF=$ROOT/ms-swift/duplex_vla/v5_checkpoint_handoff.py
EXPAND=$ROOT/ms-swift/duplex_vla/expand_ckpt_vocab_v4.py
PREPARE_PY=$SHARE/conda_envs/JarvisVLA/bin/python
TRAIN_PY=$SHARE/conda_envs/swift/bin/python
OUTPUT_ROOT=${V5_OUTPUT_ROOT:-$SHARE/checkpoint/v5_ce_camera_exponential}
STAGE2_STEPS=${V5_STAGE2_STEPS:-1925}
DUPLEX_STEPS=${V5_DUPLEX_STEPS:-500}
STAGE2_LR=${V5_STAGE2_LR:-1e-5}
DUPLEX_LR=${V5_DUPLEX_LR:-5e-6}
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    '') ;;
    *) echo 'Usage: bash train_v5_stage2_then_duplex.sh [--dry-run]'; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo 'Unexpected arguments'; exit 2; }

export REPORT_TO=wandb
export WANDB_PROJECT=v4-pipeline
export MCVLA_REPORT_TOKEN_TYPE_ACC=1
export CAMERA_EXP_ALPHA=1.0
export CAMERA_EXP_TAU=4.0
export CAMERA_EXP_NORMALIZATION=global
export CAMERA_LOG_BATCH_WEIGHTS=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
unset WANDB_RUN_ID WANDB_RESUME

for file in "$BASE/config.json" "$BASE/tokenizer.json" "$STAGE2_DATA" "$STAGE2_DATA.stats.json" \
            "$DUPLEX_DATA" "$DUPLEX_DATA.stats.json" "$TRAIN_SH" "$HANDOFF" "$EXPAND" "$PREPARE_PY" "$TRAIN_PY"; do
    [ -f "$file" ] || { echo "Missing required input: $file"; exit 1; }
done

"$PREPARE_PY" - "$STAGE2_DATA" "$DUPLEX_DATA" "$STAGE2_STEPS" "$DUPLEX_STEPS" <<'PYEOF'
import json
import sys
from pathlib import Path
stage2, duplex, stage2_steps, duplex_steps = sys.argv[1:]
s = json.loads(Path(stage2 + '.stats.json').read_text())
d = json.loads(Path(duplex + '.stats.json').read_text())
if s['total'] != 492640 or s['vla'] != 297270 or s['qa'] != 195370:
    raise ValueError('Unexpected published v5 stage2 composition')
if d['target_frames'] != 128000 or abs(d['actual_frames'] - 128000) > 200 or d['stage2_sha256'] != s['sha256']:
    raise ValueError('Expected approximately 128000 duplex frames built from the current v5 stage2')
if int(stage2_steps) <= 0 or int(duplex_steps) <= 0:
    raise ValueError('Training steps must be positive')
print(f'V5_DATA_READY stage2_rows={s["total"]} duplex_rows={d["actual_frames"]} stage2_steps={stage2_steps} duplex_steps={duplex_steps}', flush=True)
PYEOF
[ "$?" -eq 0 ] || exit 1

run_stage() {
    local name=$1 model=$2 data=$3 steps=$4 lr=$5 loss=$6
    local output=$SEQUENCE/$name
    local command=(env "CAMERA_WEIGHT_LOG_DIR=$output" "WANDB_NAME=v5-$name-$(basename "$SEQUENCE")"
        "WANDB_RUN_GROUP=v5-$(basename "$SEQUENCE")"
        bash "$TRAIN_SH" "$model" "$data" "$output" "$steps" "$lr" last_round "$loss")
    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'TRAIN %s: ' "$name"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    unset WANDB_RUN_ID WANDB_RESUME
    echo "TRAIN_START $name $(date --iso-8601=seconds) model=$model loss=${loss:-ordinary_ce}"
    "${command[@]}"
    echo "TRAIN_PROCESS_EXITED $name $(date --iso-8601=seconds); proceeding without exit-status or trainer-state checks"
}

run_duplex() {
    local stage2_name=$1 duplex_name=$2 loss=$3
    local checkpoint prepared
    if [ "$DRY_RUN" -eq 1 ]; then
        checkpoint="$SEQUENCE/$stage2_name/v0-<swift-time>/checkpoint-<largest-numeric-step>"
        prepared="$SEQUENCE/prepared/$stage2_name/checkpoint-<largest-numeric-step>-v4vocab"
        printf 'LOCATE only this stage: %q %q locate %q\n' "$PREPARE_PY" "$HANDOFF" "$SEQUENCE/$stage2_name"
        printf 'EXPAND: '
        printf '%q ' "$PREPARE_PY" "$HANDOFF" prepare "$checkpoint" "$prepared" --loss-type "$loss"
        printf '\n'
    else
        checkpoint=$("$PREPARE_PY" "$HANDOFF" locate "$SEQUENCE/$stage2_name") || exit 1
        prepared="$SEQUENCE/prepared/$stage2_name/$(basename "$checkpoint")-v4vocab"
        printf '%s\n' "$checkpoint" > "$SEQUENCE/$stage2_name.checkpoint.txt" || exit 1
        echo "STAGE2_HANDOFF stage=$stage2_name checkpoint=$checkpoint prepared=$prepared loss=${loss:-ordinary_ce}"
        "$PREPARE_PY" "$HANDOFF" prepare "$checkpoint" "$prepared" --loss-type "$loss" || exit 1
    fi
    run_stage "$duplex_name" "$prepared" "$DUPLEX_DATA" "$DUPLEX_STEPS" "$DUPLEX_LR" "$loss"
}

if [ "$DRY_RUN" -eq 1 ]; then
    SEQUENCE=$OUTPUT_ROOT/DRY_RUN
else
    "$PREPARE_PY" "$HANDOFF" verify "$BASE" || exit 1
    mkdir -p "$OUTPUT_ROOT" || exit 1
    exec 8>"$OUTPUT_ROOT/sequence.lock"
    flock -n 8 || { echo 'Another v5 sequence is using this output root'; exit 1; }
    SEQUENCE=$(mktemp -d "$OUTPUT_ROOT/run-$(date +%Y%m%d-%H%M%S)-XXXXXX") || exit 1
    echo "V5_TRAIN_SEQUENCE_OUTPUT=$SEQUENCE"
    exec > >(tee -a "$SEQUENCE/sequence.log") 2>&1
    cd "$ROOT/ms-swift" || exit 1
    "$PREPARE_PY" - "$SEQUENCE" "$BASE" "$STAGE2_DATA" "$DUPLEX_DATA" "$STAGE2_STEPS" "$DUPLEX_STEPS" "$STAGE2_LR" "$DUPLEX_LR" <<'PYEOF'
import json
import os
import sys
from pathlib import Path
sequence, model, stage2, duplex, stage2_steps, duplex_steps, stage2_lr, duplex_lr = sys.argv[1:]
phases = [
    {'name': 'stage2_ce', 'loss_type': None, 'dataset': stage2, 'initial_model': model, 'max_steps': int(stage2_steps), 'learning_rate': float(stage2_lr)},
    {'name': 'stage2_camera_exponential', 'loss_type': 'camera_exponential', 'dataset': stage2, 'initial_model': model, 'max_steps': int(stage2_steps), 'learning_rate': float(stage2_lr)},
    {'name': 'duplex_ce', 'loss_type': None, 'dataset': duplex, 'stage2_source': 'stage2_ce', 'max_steps': int(duplex_steps), 'learning_rate': float(duplex_lr)},
    {'name': 'duplex_camera_exponential', 'loss_type': 'camera_exponential', 'dataset': duplex, 'stage2_source': 'stage2_camera_exponential', 'max_steps': int(duplex_steps), 'learning_rate': float(duplex_lr)},
]
plan = {
    'phases': phases, 'global_batch_size': 256, 'loss_scale': 'last_round',
    'training_launcher': '/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/train_v2_stage.sh',
    'training_exit_policy': 'Wait synchronously for process termination; ignore training exit code and do not inspect trainer_state/global_step. Next training is unconditional except required checkpoint/model preparation.',
    'checkpoint_policy': 'Largest numeric checkpoint only inside the fresh matching stage output; Swift v0 subdirectory supported; no previous run or derived checkpoint fallback.',
    'duplex_model_policy': 'Expand matching stage2 weights from tokenizer 151908 to 151911 with silence/obs_user/obs_asst; initialize fresh optimizer and scheduler.',
    'new_token_ids': {'<|silence|>': 151908, '<|obs_user|>': 151909, '<|obs_asst|>': 151910},
    'camera_exponential': {'alpha': 1.0, 'tau': 4.0, 'normalization': 'global', 'batch_weight_logging': False},
    'report_to': os.environ['REPORT_TO'], 'wandb_project': os.environ['WANDB_PROJECT'],
    'data_sha256': {name: json.loads(Path(path + '.stats.json').read_text()).get('sha256') for name, path in (('stage2', stage2), ('duplex', duplex))},
}
Path(sequence, 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
PYEOF
    [ "$?" -eq 0 ] || exit 1
fi

run_stage stage2_ce "$BASE" "$STAGE2_DATA" "$STAGE2_STEPS" "$STAGE2_LR" ''
run_stage stage2_camera_exponential "$BASE" "$STAGE2_DATA" "$STAGE2_STEPS" "$STAGE2_LR" camera_exponential
run_duplex stage2_ce duplex_ce ''
run_duplex stage2_camera_exponential duplex_camera_exponential camera_exponential

if [ "$DRY_RUN" -eq 0 ]; then
    echo "ALL_FOUR_TRAINING_PROCESSES_EXITED $(date --iso-8601=seconds) outputs=$SEQUENCE"
fi

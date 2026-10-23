#!/bin/bash
# action_only 数据 + flash-attn 重训 (速度对比实验): 与 SDPA 版 train_v5_split_action_then_qa.sh 的
# action_only_ce 阶段完全同参数 (stage2_action.jsonl, 1162步, lr 1e-5, last_round, 普通CE, 基座直训),
# 唯一区别: TRAIN_ATTN_IMPL=flash_attn.
set -uo pipefail
set +e

ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
BASE=$SHARE/HF_MODEL/JarvisVLA-Qwen2-VL-7B
ACTION_DATA=$SHARE/dataset/duplex_vla_v5/stage2_action.jsonl
TRAIN_SH=$ROOT/ms-swift/duplex_vla/train_v2_stage.sh
PREPARE_PY=$SHARE/conda_envs/JarvisVLA/bin/python
TRAIN_PY=$SHARE/conda_envs/swift/bin/python
OUTPUT_ROOT=${V5_FA_OUTPUT_ROOT:-$SHARE/checkpoint/v5_split_action_fa}
ACTION_STEPS=${V5_ACTION_STEPS:-1162}
LR=${V5_SPLIT_LR:-1e-5}
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    '') ;;
    *) echo 'Usage: bash train_v5_action_only_fa.sh [--dry-run]'; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo 'Unexpected arguments'; exit 2; }

export REPORT_TO=wandb
export WANDB_PROJECT=v4-pipeline
export MCVLA_REPORT_TOKEN_TYPE_ACC=1
export CAMERA_LOG_BATCH_WEIGHTS=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export TRAIN_ATTN_IMPL=flash_attn
unset WANDB_RUN_ID WANDB_RESUME

for file in "$BASE/config.json" "$BASE/tokenizer.json" "$ACTION_DATA" "$ACTION_DATA.stats.json" "$TRAIN_SH" "$PREPARE_PY" "$TRAIN_PY"; do
    [ -f "$file" ] || { echo "Missing required input: $file"; exit 1; }
done

"$TRAIN_PY" -c "import flash_attn; print('FLASH_ATTN_AVAILABLE', flash_attn.__version__)" || exit 1

"$PREPARE_PY" - "$ACTION_DATA" "$ACTION_STEPS" <<'PYEOF'
import json
import sys
from pathlib import Path
action, action_steps = sys.argv[1:]
a = json.loads(Path(action + '.stats.json').read_text())
if a['rows'] != 297270 or a['output'] != action:
    raise ValueError('Unexpected action split stats')
if int(action_steps) <= 0:
    raise ValueError('Training steps must be positive')
print(f'V5_FA_DATA_READY action_rows={a["rows"]} action_steps={action_steps}', flush=True)
PYEOF
[ "$?" -eq 0 ] || exit 1

if [ "$DRY_RUN" -eq 1 ]; then
    SEQUENCE=$OUTPUT_ROOT/DRY_RUN
else
    mkdir -p "$OUTPUT_ROOT" || exit 1
    exec 8>"$OUTPUT_ROOT/sequence.lock"
    flock -n 8 || { echo 'Another action_fa sequence is using this output root'; exit 1; }
    SEQUENCE=$(mktemp -d "$OUTPUT_ROOT/run-$(date +%Y%m%d-%H%M%S)-XXXXXX") || exit 1
    echo "V5_FA_TRAIN_SEQUENCE_OUTPUT=$SEQUENCE"
    exec > >(tee -a "$SEQUENCE/sequence.log") 2>&1
    cd "$ROOT/ms-swift" || exit 1
    "$PREPARE_PY" - "$SEQUENCE" "$BASE" "$ACTION_DATA" "$ACTION_STEPS" "$LR" <<'PYEOF'
import json
import os
import sys
from pathlib import Path
sequence, model, action, action_steps, lr = sys.argv[1:]
plan = {
    'phases': [{'name': 'action_only_ce_flash_attn', 'loss_type': None, 'dataset': action,
                'initial_model': model, 'max_steps': int(action_steps),
                'learning_rate': float(lr), 'attn_impl': 'flash_attn'}],
    'global_batch_size': 256, 'loss_scale': 'last_round',
    'training_launcher': '/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/train_v2_stage.sh',
    'note': 'Speed ablation: rerun of the action_only_ce phase with flash-attn 2.8.3 (TRAIN_ATTN_IMPL=flash_attn); all other hyperparameters identical to the SDPA run run-20261019-174550-wL5JOY/action_only_ce.',
    'sdpa_reference': 'v5_split_action_qa/run-20261019-174550-wL5JOY/action_only_ce (TRAIN_START 2026-10-19T17:45:50, exited ~2026-10-20T04:29)',
    'report_to': os.environ['REPORT_TO'], 'wandb_project': os.environ['WANDB_PROJECT'],
    'data_sha256': json.loads(Path(action + '.stats.json').read_text()).get('source_sha256'),
}
Path(sequence, 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
PYEOF
    [ "$?" -eq 0 ] || exit 1
fi

run_stage() {
    local name=$1 model=$2 data=$3 steps=$4 lr=$5
    local output=$SEQUENCE/$name
    local command=(env "WANDB_NAME=v5fa-$name-$(basename "$SEQUENCE")"
        "WANDB_RUN_GROUP=v5fa-$(basename "$SEQUENCE")"
        bash "$TRAIN_SH" "$model" "$data" "$output" "$steps" "$lr" last_round '')
    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'TRAIN %s: ' "$name"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    unset WANDB_RUN_ID WANDB_RESUME
    echo "TRAIN_START $name $(date --iso-8601=seconds) model=$model loss=ordinary_ce attn=flash_attn"
    "${command[@]}"
    echo "TRAIN_PROCESS_EXITED $name $(date --iso-8601=seconds)"
}

run_stage action_only_ce_flash_attn "$BASE" "$ACTION_DATA" "$ACTION_STEPS" "$LR"

if [ "$DRY_RUN" -eq 0 ]; then
    echo "V5_FA_TRAIN_PROCESS_EXITED $(date --iso-8601=seconds) outputs=$SEQUENCE"
fi

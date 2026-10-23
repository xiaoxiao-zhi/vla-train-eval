#!/bin/bash
# v5 拆分数据消融: 先用 action 数据、再用 general-qa 数据, 均为普通 CE, 均直接从 JarvisVLA 基座训练.
# 参数与 v5 stage2 训练一致 (lr 1e-5, last_round, 全局batch 256, cosine, warmup50, bf16, ZeRO2);
# max_steps 按同一规则 ceil(行数/256) ≈ 1 epoch: action 297270->1162, qa 195370->764.
set -uo pipefail
set +e

ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
BASE=$SHARE/HF_MODEL/JarvisVLA-Qwen2-VL-7B
ACTION_DATA=$SHARE/dataset/duplex_vla_v5/stage2_action.jsonl
QA_DATA=$SHARE/dataset/duplex_vla_v5/stage2_general_qa.jsonl
TRAIN_SH=$ROOT/ms-swift/duplex_vla/train_v2_stage.sh
PREPARE_PY=$SHARE/conda_envs/JarvisVLA/bin/python
TRAIN_PY=$SHARE/conda_envs/swift/bin/python
OUTPUT_ROOT=${V5_SPLIT_OUTPUT_ROOT:-$SHARE/checkpoint/v5_split_action_qa}
ACTION_STEPS=${V5_ACTION_STEPS:-1162}
QA_STEPS=${V5_QA_STEPS:-764}
LR=${V5_SPLIT_LR:-1e-5}
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    '') ;;
    *) echo 'Usage: bash train_v5_split_action_then_qa.sh [--dry-run]'; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo 'Unexpected arguments'; exit 2; }

export REPORT_TO=wandb
export WANDB_PROJECT=v4-pipeline
export MCVLA_REPORT_TOKEN_TYPE_ACC=1
export CAMERA_LOG_BATCH_WEIGHTS=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
unset WANDB_RUN_ID WANDB_RESUME

for file in "$BASE/config.json" "$BASE/tokenizer.json" "$ACTION_DATA" "$ACTION_DATA.stats.json" \
            "$QA_DATA" "$QA_DATA.stats.json" "$TRAIN_SH" "$PREPARE_PY" "$TRAIN_PY"; do
    [ -f "$file" ] || { echo "Missing required input: $file"; exit 1; }
done

"$PREPARE_PY" - "$ACTION_DATA" "$QA_DATA" "$ACTION_STEPS" "$QA_STEPS" <<'PYEOF'
import json
import sys
from pathlib import Path
action, qa, action_steps, qa_steps = sys.argv[1:]
a = json.loads(Path(action + '.stats.json').read_text())
q = json.loads(Path(qa + '.stats.json').read_text())
if a['rows'] != 297270 or a['output'] != action:
    raise ValueError('Unexpected action split stats')
if q['rows'] != 195370 or q['output'] != qa:
    raise ValueError('Unexpected qa split stats')
if a['source_sha256'] != q['source_sha256']:
    raise ValueError('Split parts must come from the same stage2.jsonl')
if int(action_steps) <= 0 or int(qa_steps) <= 0:
    raise ValueError('Training steps must be positive')
print(f'V5_SPLIT_DATA_READY action_rows={a["rows"]} qa_rows={q["rows"]} '
      f'action_steps={action_steps} qa_steps={qa_steps}', flush=True)
PYEOF
[ "$?" -eq 0 ] || exit 1

run_stage() {
    local name=$1 model=$2 data=$3 steps=$4 lr=$5
    local output=$SEQUENCE/$name
    local command=(env "WANDB_NAME=v5split-$name-$(basename "$SEQUENCE")"
        "WANDB_RUN_GROUP=v5split-$(basename "$SEQUENCE")"
        bash "$TRAIN_SH" "$model" "$data" "$output" "$steps" "$lr" last_round '')
    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'TRAIN %s: ' "$name"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    unset WANDB_RUN_ID WANDB_RESUME
    echo "TRAIN_START $name $(date --iso-8601=seconds) model=$model loss=ordinary_ce"
    "${command[@]}"
    echo "TRAIN_PROCESS_EXITED $name $(date --iso-8601=seconds); proceeding without exit-status or trainer-state checks"
}

if [ "$DRY_RUN" -eq 1 ]; then
    SEQUENCE=$OUTPUT_ROOT/DRY_RUN
else
    "$PREPARE_PY" "$ROOT/ms-swift/duplex_vla/v5_checkpoint_handoff.py" verify "$BASE" || exit 1
    mkdir -p "$OUTPUT_ROOT" || exit 1
    exec 8>"$OUTPUT_ROOT/sequence.lock"
    flock -n 8 || { echo 'Another v5 split sequence is using this output root'; exit 1; }
    SEQUENCE=$(mktemp -d "$OUTPUT_ROOT/run-$(date +%Y%m%d-%H%M%S)-XXXXXX") || exit 1
    echo "V5_SPLIT_TRAIN_SEQUENCE_OUTPUT=$SEQUENCE"
    exec > >(tee -a "$SEQUENCE/sequence.log") 2>&1
    cd "$ROOT/ms-swift" || exit 1
    "$PREPARE_PY" - "$SEQUENCE" "$BASE" "$ACTION_DATA" "$QA_DATA" "$ACTION_STEPS" "$QA_STEPS" "$LR" <<'PYEOF'
import json
import os
import sys
from pathlib import Path
sequence, model, action, qa, action_steps, qa_steps, lr = sys.argv[1:]
phases = [
    {'name': 'action_only_ce', 'loss_type': None, 'dataset': action, 'initial_model': model,
     'max_steps': int(action_steps), 'learning_rate': float(lr)},
    {'name': 'qa_only_ce', 'loss_type': None, 'dataset': qa, 'initial_model': model,
     'max_steps': int(qa_steps), 'learning_rate': float(lr)},
]
plan = {
    'phases': phases, 'global_batch_size': 256, 'loss_scale': 'last_round',
    'training_launcher': '/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/train_v2_stage.sh',
    'training_exit_policy': 'Wait synchronously for process termination; ignore training exit code and do not inspect trainer_state/global_step. Next training is unconditional.',
    'checkpoint_policy': 'Largest numeric checkpoint only inside the fresh matching stage output; Swift v0 subdirectory supported; no previous run or derived checkpoint fallback.',
    'note': 'Ablation: v5 stage2.jsonl split into action (297270 rows, 4706 segments) and general-qa (195370 rows) parts; both phases train directly from the JarvisVLA base with ordinary CE; steps = ceil(rows/256) one epoch, all other hyperparameters identical to v5 stage2 trainings.',
    'report_to': os.environ['REPORT_TO'], 'wandb_project': os.environ['WANDB_PROJECT'],
    'data_sha256': {name: json.loads(Path(path + '.stats.json').read_text()).get('source_sha256')
                    for name, path in (('action_source_stage2', action), ('qa_source_stage2', qa))},
}
Path(sequence, 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
PYEOF
    [ "$?" -eq 0 ] || exit 1
fi

run_stage action_only_ce "$BASE" "$ACTION_DATA" "$ACTION_STEPS" "$LR"
run_stage qa_only_ce "$BASE" "$QA_DATA" "$QA_STEPS" "$LR"

if [ "$DRY_RUN" -eq 0 ]; then
    echo "V5_SPLIT_TWO_TRAINING_PROCESSES_EXITED $(date --iso-8601=seconds) outputs=$SEQUENCE"
fi

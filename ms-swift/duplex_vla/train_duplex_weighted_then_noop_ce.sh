#!/bin/bash
set -euo pipefail

ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
LAUNCHER=$ROOT/ms-swift/duplex_vla/train_v2_stage.sh
OUTPUT_ROOT=$SHARE/checkpoint/duplex_weighted_then_noop_ce
DUPLEX_MODEL=$SHARE/checkpoint/duplex_v3_500traj_ce/prepared/checkpoint-1936-v4vocab
DUPLEX_DATA=$SHARE/dataset/flat3_ablation/duplex_train_v3_500traj_filtered.jsonl
STAGE2_MODEL=$SHARE/HF_MODEL/JarvisVLA-Qwen2-VL-7B
STAGE2_DATA=$SHARE/dataset/flat3_ablation/stage2_train_flat3_vla300k_qav3.jsonl
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    '') ;;
    *) echo 'Usage: bash train_duplex_weighted_then_noop_ce.sh [--dry-run]'; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo 'Unexpected arguments'; exit 2; }

run_stage() {
    local name=$1 model=$2 data=$3 steps=$4 lr=$5 loss=$6
    local output=$SEQUENCE/$name
    local command=(env
        CAMERA_EXP_ALPHA=1.0 CAMERA_EXP_TAU=4.0 CAMERA_EXP_NORMALIZATION=global
        CAMERA_LOG_BATCH_WEIGHTS=0 "CAMERA_WEIGHT_LOG_DIR=$output"
        MCVLA_REPORT_TOKEN_TYPE_ACC=1 PYTHONUNBUFFERED=1
        REPORT_TO=wandb WANDB_PROJECT=v4-pipeline "WANDB_NAME=$name-$(basename "$SEQUENCE")"
        bash "$LAUNCHER" "$model" "$data" "$output" "$steps" "$lr" last_round "$loss")
    if [ "$DRY_RUN" -eq 1 ]; then
        printf 'STAGE %s: ' "$name"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    echo "STAGE_START $name $(date --iso-8601=seconds) model=$model data=$data steps=$steps lr=$lr loss=${loss:-ordinary_ce}"
    unset WANDB_RUN_ID WANDB_RESUME
    "${command[@]}"
    echo "STAGE_COMPLETE $name $(date --iso-8601=seconds)"
}

if [ "$DRY_RUN" -eq 1 ]; then
    SEQUENCE=$OUTPUT_ROOT/DRY_RUN
else
    mkdir -p "$OUTPUT_ROOT"
    exec 8>"$OUTPUT_ROOT/sequence.lock"
    flock -n 8 || { echo 'Another sequence is using this output root'; exit 1; }
    SEQUENCE=$(mktemp -d "$OUTPUT_ROOT/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")
    echo "TRAIN_SEQUENCE_OUTPUT=$SEQUENCE"
    exec > >(tee -a "$SEQUENCE/sequence.log") 2>&1
    trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "SEQUENCE_FAILED rc=$rc; subsequent training will not start"; fi' EXIT
    cd "$ROOT/ms-swift"
fi

run_stage duplex_camera_exponential "$DUPLEX_MODEL" "$DUPLEX_DATA" 243 5e-6 camera_exponential
run_stage stage2_noop_unfiltered_ce "$STAGE2_MODEL" "$STAGE2_DATA" 1936 1e-5 ''
if [ "$DRY_RUN" -eq 0 ]; then
    echo "SEQUENCE_COMPLETE $(date --iso-8601=seconds)"
fi

#!/bin/bash
# ===== v2 训练 workflow 单步脚本 (stage2 或 duplex 续训, 参数化) =====
# 用法:
#   bash train_v2_stage.sh <model_dir> <dataset> <output_dir> <max_steps> [lr] [loss_scale] [loss_type]
# 全局 batch 256 = 8卡 x per-device 2 x 梯度累计 16
set -e
set -o pipefail

MODEL=$1; DATA=$2; OUTPUT=$3; MAX_STEPS=$4
LR=${5:-1e-5}
LOSS_SCALE=${6:-default}   # duplex 数据必须传 last_round
LOSS_TYPE=${7:-}           # 可选: camera_weighted 等 (注册于 run_train.py 的 loss_map)
ATTN_IMPL=${TRAIN_ATTN_IMPL:-}  # 可选: flash_attn / sdpa / eager (默认不传, 走 transformers 自动解析)

PYTHON=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/swift/bin/python
TRAIN_ENTRY=/apdcephfs/private_chengzhixu/ms-swift/mcvla/run_train.py

REPORT_TO=${REPORT_TO:-wandb}
if [ "$REPORT_TO" = "wandb" ]; then
    export http_proxy="http://star-proxy.oa.com:3128"
    export https_proxy="http://star-proxy.oa.com:3128"
    export no_proxy="localhost,127.0.0.1"
    export WANDB_PROJECT="${WANDB_PROJECT:-v4-pipeline}"
    echo "[wandb] 上报已启用: project=$WANDB_PROJECT"
fi

EXTRA_ARGS=()
if [ -n "$LOSS_TYPE" ]; then
    EXTRA_ARGS+=(--loss_type "$LOSS_TYPE")
    echo "[loss] loss_type=$LOSS_TYPE"
fi
if [ -n "$ATTN_IMPL" ]; then
    EXTRA_ARGS+=(--attn_impl "$ATTN_IMPL")
    echo "[attn] attn_impl=$ATTN_IMPL"
fi

mkdir -p $OUTPUT

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
$PYTHON -m torch.distributed.run --nproc_per_node 8 $TRAIN_ENTRY \
    --model $MODEL \
    --model_type qwen2_vl \
    --template qwen2_vl \
    --dataset $DATA \
    --streaming true \
    --shuffle_buffer_size 4096 \
    --split_dataset_ratio 0 \
    --max_steps $MAX_STEPS \
    --loss_scale $LOSS_SCALE \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --warmup_steps 50 \
    --max_length 8192 \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --report_to $REPORT_TO \
    --output_dir $OUTPUT \
    --deepspeed zero2 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a $OUTPUT/train.log

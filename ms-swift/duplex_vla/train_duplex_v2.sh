#!/bin/bash
# ===== 全双工 v2 训练 (新格式: 标准 messages + <|silence|>, swift 原生管线) =====
# v1 (train_duplex.py 自定义管线 + duplex_vla_v2 旧格式) 保持不动, 独立共存.
#
# 数据: duplex_vla_v3/duplex_train.jsonl (多轮, 历史窗口2, QA 跨帧流式回答)
# 基座: JarvisVLA-Qwen2-VL-7B-silence (官方 + <|silence|> id=151908)
# 入口: run_train.py (不传 --loss_type -> 默认全词表 CE)
set -e

PYTHON=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/swift/bin/python
TRAIN_ENTRY=/apdcephfs/private_chengzhixu/ms-swift/mcvla/run_train.py
MODEL=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B-silence
DATA=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/duplex_vla_v3/duplex_train.jsonl
OUTPUT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/duplex_v2

# 超参 (对齐 stage2 惯例; 198,170 轮 / (4卡x8x4) = 1548 步/epoch)
MAX_STEPS=${MAX_STEPS:-1600}
PER_DEVICE_BS=${PER_DEVICE_BS:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_LENGTH=8192
SAVE_STEPS=400
SHUFFLE_BUFFER=4096
NPROC=${NPROC:-4}
GPUS=${GPUS:-4,5,6,7}

# wandb (可选)
REPORT_TO=${REPORT_TO:-none}
if [ "$REPORT_TO" = "wandb" ]; then
    export http_proxy="http://star-proxy.oa.com:3128"
    export https_proxy="http://star-proxy.oa.com:3128"
    export no_proxy="localhost,127.0.0.1"
    export WANDB_PROJECT="duplex-v2"
fi

mkdir -p $OUTPUT

CUDA_VISIBLE_DEVICES=$GPUS \
$PYTHON -m torch.distributed.run --nproc_per_node $NPROC $TRAIN_ENTRY \
    --model $MODEL \
    --model_type qwen2_vl \
    --template qwen2_vl \
    --dataset $DATA \
    --streaming true \
    --shuffle_buffer_size $SHUFFLE_BUFFER \
    --split_dataset_ratio 0 \
    --max_steps $MAX_STEPS \
    --loss_scale last_round \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 50 \
    --max_length $MAX_LENGTH \
    --save_steps $SAVE_STEPS \
    --save_total_limit 5 \
    --logging_steps 5 \
    --report_to $REPORT_TO \
    --output_dir $OUTPUT \
    --deepspeed zero2 \
    2>&1 | tee -a $OUTPUT/train.log

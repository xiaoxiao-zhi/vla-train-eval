#!/bin/bash
# JarvisVLA 风格 SFT 阶段二: 混合 VLA 数据 + 通用 QA 数据训练.
# BASE=jarvisvla (推荐路径): 直接从官方 JarvisVLA-Qwen2-VL-7B 起训, 跳过阶段一;
# BASE=qwen2_5_vl: 从自训阶段一 checkpoint 起训.
#
# 与阶段一 (train_mcvla.sh) 的区别:
#   1. 模型起点: 阶段一产出的 checkpoint (含已训练的动作 token embedding)
#   2. 数据: mcvla VLA 样本 (313k) + 通用 QA (100k) 合并后的流式 jsonl (~413k)
#   3. loss: 不传 --loss_type, 即 swift 默认全词表 CE
#      (通用 QA 的答案覆盖整个自然语言词表, 动作位置不能再做候选子集受限 CE)
#
# 基座选择 (BASE 环境变量切换, 默认 qwen2_5_vl):
#   BASE=qwen2_5_vl (默认): 阶段一用 train_mcvla.sh (Qwen2.5-VL) 的产出
#   BASE=jarvisvla:         直接从官方 JarvisVLA-Qwen2-VL-7B 起训, 跳过阶段一
#       (官方模型已具备阶段一能力: 动作头经 377 万样本 SFT, MC-VLP 预训练,
#        无需再用我们的 31 万样本重复阶段一, 直接混合 VLA+QA 做阶段二)
#
# 兼容性: 不修改 run_train.py / train_mcvla.sh / action_constrained_loss.py 任何逻辑;
#         run_train.py 注册的 action_constrained loss 仅在 --loss_type action_constrained
#         时生效, 阶段一脚本可照常运行.
#
# 前置:
#   1) BASE=jarvisvla: 无阶段一依赖, 官方模型目录可直接作 --model
#      BASE=qwen2_5_vl: 需阶段一已完成, checkpoint 含完整模型+tokenizer
#   2) 通用 QA 已生成: make_general_qa.py -> dataset/general_qa/train.jsonl (已完成, 100k)
#   3) 合并数据 (已完成, 413,303 行):
#      python merge_stage2_data.py   # -> dataset/mcvla_stage2/train.jsonl
#
# 注意:
#   - streaming 模式必须设 --max_steps: 413,303 / (8卡 x 8 x 4) ≈ 1615 steps ≈ 1 epoch
#   - 训练中可随时停掉, --resume_from_checkpoint auto 续训

set -e

# ===== wandb 上报 (可选, 默认关闭) =====
# 启用方式: REPORT_TO=wandb WANDB_API_KEY=<你的key> bash train_mcvla_stage2.sh
# (key 也可先 wandb login 一次持久化, 之后无需再传)
REPORT_TO=${REPORT_TO:-none}
if [ "$REPORT_TO" = "wandb" ]; then
    export http_proxy="http://star-proxy.oa.com:3128"
    export https_proxy="http://star-proxy.oa.com:3128"
    export no_proxy="localhost,127.0.0.1"
    export WANDB_PROJECT="${WANDB_PROJECT:-mcvla-train}"
    echo "[wandb] 已启用上报, project=$WANDB_PROJECT"
fi

# ===== 基座选择 (与阶段一一致) =====
BASE=${BASE:-qwen2_5_vl}

if [ "$BASE" = "jarvisvla" ]; then
    MODEL=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B
    TEMPLATE=qwen2_vl
    LR=1e-5                  # 混训新格式(history=1)+保持动作头, 官方步长5e-6与1e-5之间取偏高
    OUTPUT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/jarvisvla_s2
else
    MODEL=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/v2-20260920-214927/checkpoint-1225
    TEMPLATE=qwen2_5_vl
    LR=1e-5                  # 阶段二为延续训练, 建议低于阶段一的 3e-5, 防止动作头遗忘 (可按需调回)
    OUTPUT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/stage2
fi

PYTHON=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/swift/bin/python
TRAIN_ENTRY=/apdcephfs/private_chengzhixu/ms-swift/mcvla/run_train.py   # 注册受限loss的入口; 不传 --loss_type 时走默认全词表CE

# ===== 训练超参 =====
MAX_STEPS=2916           # VLA 帧样本 446,440 + 通用QA 300,000 = 746,440 行 / 256 ≈ 1 epoch
PER_DEVICE_BS=8
GRAD_ACCUM=4             # 全局 batch = 8卡 x 8 x 4 = 256
MAX_LENGTH=4096
SAVE_STEPS=500
SHUFFLE_BUFFER=4096      # 流式 shuffle 缓冲

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
$PYTHON -m torch.distributed.run --nproc_per_node 8 $TRAIN_ENTRY \
    --model $MODEL \
    --model_type qwen2_vl \
    --template $TEMPLATE \
    --dataset /apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_stage2_v2/train.jsonl \
    --streaming true \
    --shuffle_buffer_size $SHUFFLE_BUFFER \
    --split_dataset_ratio 0 \
    --max_steps $MAX_STEPS \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_length $MAX_LENGTH \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --logging_steps 10 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing true \
    --output_dir $OUTPUT \
    --report_to $REPORT_TO \
    --loss_scale last_round \
    --deepspeed zero2
# 续训时追加: --resume_from_checkpoint $OUTPUT/checkpoint-<step>

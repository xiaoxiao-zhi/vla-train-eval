#!/bin/bash
# JarvisVLA 风格 SFT 阶段一 (ms-swift, 流式训练)
# 样本: 指令 + 上一帧 + 上一动作 + 当前帧 -> 当前动作保留字序列
# 微调: 全参 (冻结视觉端, 训练语言主干+动作token embedding)
#
# 支持两种基座 (BASE 环境变量切换, 默认 qwen2_5_vl 保持原行为):
#   BASE=qwen2_5_vl (默认): 词表扩展后的 Qwen2.5-VL-7B-Instruct-mcvla
#       - 需先跑 extend_vocab.py 生成该模型目录
#       - template: qwen2_5_vl, LR 3e-5
#   BASE=jarvisvla: HF_MODEL/JarvisVLA-Qwen2-VL-7B
#       - Qwen2-VL-7B 架构, 词表已含全部动作保留字 (id 151675~151907, 无需 extend_vocab)
#       - 动作头已经过 JarvisVLA 官方 SFT, LR 对齐官方 5e-6
#       - template: qwen2_vl
#
# 两个基座共用的不变部分: 训练数据 (token 字符串存储, tokenizer 运行时转 id)、
# 冻结策略、8卡 ZeRO-2。
# loss: 不传 --loss_type, 走 swift 默认全词表 CE (与 JarvisVLA 官方训练一致;
#       受限 loss 已弃用, 如需恢复在启动参数加 --loss_type action_constrained,
#       run_train.py 仍保留 ActionConstrainedLoss 的注册)。
#
# 用法:
#   bash train_mcvla.sh                    # 默认 qwen2_5_vl, 与原行为一致
#   BASE=jarvisvla bash train_mcvla.sh      # JarvisVLA-Qwen2-VL-7B 基座
#
# 前置:
#   1) 数据生产者已在运行 (make_sft_stream.py 持续写 train.jsonl + frames/)
#      启动本脚本前确认生产者已领先 (progress.json 的 rows_written > 数千)
#   2) BASE=qwen2_5_vl 时需已运行 extend_vocab.py 生成 $MODEL
#
# 注意:
#   - streaming 模式必须设 --max_steps (数据总量未知, 按预估上限设)
#     当前清单 31 万样本: 8卡 x per_device 8 x accum 4 = 256/step -> 约 1225 steps/epoch
#   - 训练中可随时停掉, --resume_from_checkpoint auto 续训

set -e

# ===== wandb 上报 (可选, 默认关闭) =====
# 启用方式: REPORT_TO=wandb WANDB_API_KEY=<你的key> bash train_mcvla.sh
# (key 也可先 wandb login 一次持久化, 之后无需再传)
REPORT_TO=${REPORT_TO:-none}
if [ "$REPORT_TO" = "wandb" ]; then
    export http_proxy="http://star-proxy.oa.com:3128"
    export https_proxy="http://star-proxy.oa.com:3128"
    export no_proxy="localhost,127.0.0.1"
    export WANDB_PROJECT="${WANDB_PROJECT:-mcvla-train}"
    echo "[wandb] 已启用上报, project=$WANDB_PROJECT"
fi

# ===== 基座选择 =====
BASE=${BASE:-qwen2_5_vl}

if [ "$BASE" = "jarvisvla" ]; then
    MODEL=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B
    TEMPLATE=qwen2_vl
    LR=5e-6                  # JarvisVLA 官方 SFT LR (起点已具备VLA能力, 用小步长防破坏动作头)
    OUTPUT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/jarvisvla_s1
else
    MODEL=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/Qwen2.5-VL-7B-Instruct-mcvla
    TEMPLATE=qwen2_5_vl
    LR=3e-5
    OUTPUT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint
fi

PYTHON=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/swift/bin/python
TRAIN_ENTRY=/apdcephfs/private_chengzhixu/ms-swift/mcvla/run_train.py   # 保留受限loss注册的入口; 不传 --loss_type 时走默认全词表CE

# ===== 训练超参 =====
MAX_STEPS=3675           # 流式模式必须: 31.3万样本/256 ≈ 1225 steps/epoch, 此处 3 epoch
                         # (v2 诊断: 1 epoch 严重欠拟合, 训练样本匹配率仅 4%)
PER_DEVICE_BS=8          # history=1 样本序列 ~700 token
GRAD_ACCUM=4             # 全局 batch = 8卡 x 8 x 4 = 256
MAX_LENGTH=4096
SAVE_STEPS=1000
SHUFFLE_BUFFER=4096      # 流式 shuffle 缓冲

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
$PYTHON -m torch.distributed.run --nproc_per_node 8 $TRAIN_ENTRY \
    --model $MODEL \
    --template $TEMPLATE \
    --dataset /apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft/train.jsonl \
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
    --warmup_steps 200 \
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
    --deepspeed zero2
# 续训时追加: --resume_from_checkpoint $OUTPUT/checkpoint-<step>

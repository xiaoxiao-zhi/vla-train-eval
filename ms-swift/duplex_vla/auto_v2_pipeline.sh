#!/bin/bash
# ===== v2 完整训练 workflow =====
#   Stage 1: 官方基座 -> stage2_hist0 / stage2_hist1 (串行 8 卡)
#   Stage 2: 两个 stage2 模型各自扩词表 (+<|silence|>)
#   Stage 3: 扩词表后的模型 + duplex 数据续训 (last_round loss)
# 全局 batch 256 = 8卡 x 2 x 16
set -e
set -o pipefail
cd /apdcephfs/private_chengzhixu/ms-swift/duplex_vla

BASE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B
DATA_DIR=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/duplex_vla_v3
CKPT_DIR=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint
PY=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/swift/bin/python

# 步数: stage2 73.5万/256 ~ 2872 步/epoch; duplex 19.8万/256 ~ 774 步/epoch
S2_STEPS=${S2_STEPS:-2900}
DX_STEPS=${DX_STEPS:-800}
S2_LR=${S2_LR:-1e-5}
DX_LR=${DX_LR:-5e-6}

echo "===== [1/5] stage2 hist0 训练 ====="
bash train_v2_stage.sh $BASE $DATA_DIR/stage2_train_hist0.jsonl \
    $CKPT_DIR/s2v3_hist0 $S2_STEPS $S2_LR last_round

echo "===== [2/5] stage2 hist1 训练 ====="
bash train_v2_stage.sh $BASE $DATA_DIR/stage2_train_hist1.jsonl \
    $CKPT_DIR/s2v3_hist1 $S2_STEPS $S2_LR last_round

echo "===== [3/5] 两个模型扩词表 ====="
CK0=$(ls -d $CKPT_DIR/s2v3_hist0/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
CK1=$(ls -d $CKPT_DIR/s2v3_hist1/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "hist0 最终 ckpt: $CK0"; echo "hist1 最终 ckpt: $CK1"
test -n "$CK0" && test -n "$CK1" || { echo "错误: 未找到 checkpoint"; exit 1; }
$PY expand_ckpt_vocab.py $CK0 $CKPT_DIR/s2v3_hist0-silence
$PY expand_ckpt_vocab.py $CK1 $CKPT_DIR/s2v3_hist1-silence

echo "===== [4/5] duplex 续训 (hist0 系) ====="
bash train_v2_stage.sh $CKPT_DIR/s2v3_hist0-silence $DATA_DIR/duplex_train.jsonl \
    $CKPT_DIR/duplex_v2_hist0 $DX_STEPS $DX_LR last_round

echo "===== [5/5] duplex 续训 (hist1 系) ====="
bash train_v2_stage.sh $CKPT_DIR/s2v3_hist1-silence $DATA_DIR/duplex_train.jsonl \
    $CKPT_DIR/duplex_v2_hist1 $DX_STEPS $DX_LR last_round

echo "===== workflow 全部完成 ====="

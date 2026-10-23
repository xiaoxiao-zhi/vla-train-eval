#!/bin/bash
# ===== v2 训练 workflow 第二批: hist2/hist3 全线 =====
#   Stage 1: 两个 stage2 checkpoint 扩词表 (+<|silence|>, 含 rope_scaling 自检)
#   Stage 2: duplex 续训 × 4 (先 hist0/hist1 系, 再 hist2/hist3 系)
# 复用已完成的 s2v3_hist0/hist1 checkpoint; hist2/hist3 stage2 需另行训练 (不在本脚本).
# 全局 batch 256 = 8卡 x per-device 2 x 梯度累计 16, last_round loss.
set -e
set -o pipefail
cd /apdcephfs/private_chengzhixu/ms-swift/duplex_vla

CKPT_DIR=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint
DATA_DIR=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/duplex_vla_v3
PY=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/swift/bin/python

DX_STEPS=${DX_STEPS:-800}
DX_LR=${DX_LR:-5e-6}

# ---- checkpoint 取法 (兼容 v*/ 子目录, 空则报错退出, 不再静默跳过) ----
latest_ck() {
    local d=$(ls -d $1/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
    [ -z "$d" ] && d=$(ls -d $1/checkpoint-* 2>/dev/null | sort -V | tail -1)
    [ -z "$d" ] && { echo "错误: $1 下无 checkpoint"; exit 1; }
    echo "$d"
}

echo "===== [1/6] 扩词表: s2v3_hist0 / s2v3_hist1 ====="
CK0=$(latest_ck $CKPT_DIR/s2v3_hist0)
CK1=$(latest_ck $CKPT_DIR/s2v3_hist1)
echo "hist0 ckpt: $CK0"; echo "hist1 ckpt: $CK1"
$PY expand_ckpt_vocab.py $CK0 $CKPT_DIR/s2v3_hist0-silence
$PY expand_ckpt_vocab.py $CK1 $CKPT_DIR/s2v3_hist1-silence

echo "===== [2/6] duplex 续训 (hist0 系) ====="
bash train_v2_stage.sh $CKPT_DIR/s2v3_hist0-silence $DATA_DIR/duplex_train.jsonl \
    $CKPT_DIR/duplex_v2_hist0 $DX_STEPS $DX_LR last_round

echo "===== [3/6] duplex 续训 (hist1 系) ====="
bash train_v2_stage.sh $CKPT_DIR/s2v3_hist1-silence $DATA_DIR/duplex_train.jsonl \
    $CKPT_DIR/duplex_v2_hist1 $DX_STEPS $DX_LR last_round

echo "===== [4/6] stage2 hist2 训练 ====="
bash train_v2_stage.sh /apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B \
    $DATA_DIR/stage2_train_hist2.jsonl $CKPT_DIR/s2v3_hist2 2900 1e-5 last_round

echo "===== [5/6] stage2 hist3 训练 ====="
bash train_v2_stage.sh /apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B \
    $DATA_DIR/stage2_train_hist3.jsonl $CKPT_DIR/s2v3_hist3 2900 1e-5 last_round

echo "===== [6/6] duplex 续训 (hist2/hist3 系) ====="
CK2=$(latest_ck $CKPT_DIR/s2v3_hist2)
CK3=$(latest_ck $CKPT_DIR/s2v3_hist3)
$PY expand_ckpt_vocab.py $CK2 $CKPT_DIR/s2v3_hist2-silence
$PY expand_ckpt_vocab.py $CK3 $CKPT_DIR/s2v3_hist3-silence
bash train_v2_stage.sh $CKPT_DIR/s2v3_hist2-silence $DATA_DIR/duplex_train.jsonl \
    $CKPT_DIR/duplex_v2_hist2 $DX_STEPS $DX_LR last_round
bash train_v2_stage.sh $CKPT_DIR/s2v3_hist3-silence $DATA_DIR/duplex_train.jsonl \
    $CKPT_DIR/duplex_v2_hist3 $DX_STEPS $DX_LR last_round

echo "===== 第二批 workflow 全部完成 ====="

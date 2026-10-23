#!/bin/bash
# 自动流水线: 等待VLA数据生产完成 -> 校验 -> 合并stage2 -> GPU机启动训练
# 用法: nohup bash auto_pipeline.sh > /tmp/auto_pipeline.log 2>&1 &
set -u

VLA=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset/mcvla_sft_50_200
MCVLA_DIR=/apdcephfs/private_chengzhixu/ms-swift/mcvla
LOG=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint/stage2_train.log

echo "[$(date)] 等待 VLA 生产完成 ..."
while pgrep -f make_sft_stream > /dev/null; do sleep 60; done
sleep 15  # 等文件句柄释放/flush

# 1. 校验行数 (预期 ~424k, 允许 400k 以上)
N=$(wc -l < "$VLA/train.jsonl")
echo "[$(date)] VLA 生产结束: $N 行"
if [ "$N" -lt 400000 ]; then
    echo "[ABORT] 行数异常 ($N < 400000), 不启动训练, 请检查"
    exit 1
fi

# 2. 合并 stage2 (VLA + 通用QA -> ~620k 行)
echo "[$(date)] 合并 stage2 数据 ..."
cd "$MCVLA_DIR" || exit 1
/usr/bin/python merge_stage2_data.py || exit 1
echo "[$(date)] 合并完成"

# 3. GPU 机启动训练 (BASE=jarvisvla, 官方模型直接起)
echo "[$(date)] 启动 GPU 训练 ..."
timeout 120 bash /apdcephfs/private_chengzhixu/gpu_exec.sh \
    "cd $MCVLA_DIR && mkdir -p $(dirname $LOG) && BASE=jarvisvla setsid nohup bash train_mcvla_stage2.sh < /dev/null > $LOG 2>&1 & echo TRAIN_STARTED \$!" 150 \
    && echo "[$(date)] 训练已启动, 日志: $LOG" \
    || echo "[ABORT] GPU 启动失败, 手动执行: BASE=jarvisvla bash train_mcvla_stage2.sh"

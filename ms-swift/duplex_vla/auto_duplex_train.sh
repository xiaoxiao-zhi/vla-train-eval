#!/bin/bash
# 全双工训练自动串联: 轮询 stage2 完成 -> 扩词表 -> GPU冒烟 -> 正式训练.
#
# 流程:
#   1. 每 10 分钟检测 stage2 (jarvisvla_s2) 是否训完 (checkpoint-2423 存在 且 训练进程已退出)
#   2. make_duplex_base.py --base <stage2最新checkpoint> --out HF_MODEL/JarvisVLA-s2-duplex (CPU)
#   3. GPU 反向冒烟 (train_duplex.py --smoke, 2 step, 失败即停)
#   4. 正式训练 -> checkpoint/duplex_v1 (与其他 checkpoint 平级)
#
# 用法 (GPU 机):
#   nohup bash auto_duplex_train.sh > auto_duplex.log 2>&1 &
set -e

CKPT_ROOT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint
S2_DIR=$CKPT_ROOT/jarvisvla_s2
DUPLEX_BASE_OUT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-s2-duplex
DUPLEX_CKPT_OUT=$CKPT_ROOT/duplex_v1
PY=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/swift/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
MAX_STEPS=2423

log() { echo "[$(date '+%F %T')] $*"; }

# 等待 gpu_burn 的 burn worker 退出 (避免训练启动时显存被 burn 占用导致 OOM)
wait_burn_done() {
    for i in $(seq 1 40); do   # 最多等 10 分钟
        if ! pgrep -f "multiprocessing.spawn" > /dev/null; then
            return 0
        fi
        log "gpu_burn worker 运行中, 等待其结束 ($i/40)..."
        sleep 15
    done
    log "警告: burn worker 未在 10 分钟内退出, 继续启动 (有 OOM 风险)"
}

# ---------- 1. 等待 stage2 完成 ----------
log "开始轮询 stage2 ($S2_DIR) ..."
while true; do
    # 最新 run 目录
    RUN=$(ls -dt $S2_DIR/v* 2>/dev/null | head -1)
    # 注意: pgrep -f 用 ERE, 多模式分隔符是 | (不要用 \|)
    if [ -n "$RUN" ] && [ -d "$RUN/checkpoint-$MAX_STEPS" ] && ! pgrep -f "train_mcvla_stage2|run_train.py" > /dev/null; then
        # 进程退出且最终 checkpoint 存在; 再确认权重文件完整
        if ls $RUN/checkpoint-$MAX_STEPS/model-*.safetensors > /dev/null 2>&1 || \
           [ -f $RUN/checkpoint-$MAX_STEPS/model.safetensors ]; then
            S2_CKPT=$RUN/checkpoint-$MAX_STEPS
            log "stage2 完成: $S2_CKPT"
            break
        fi
    fi
    sleep 600
done

# 等显存彻底释放
sleep 60
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8

# ---------- 2. 生成全双工基座 (从 stage2 checkpoint 续训) ----------
log "生成全双工基座 (CPU, 约4分钟): $DUPLEX_BASE_OUT"
if [ -e "$DUPLEX_BASE_OUT" ] && [ -n "$(ls -A $DUPLEX_BASE_OUT 2>/dev/null)" ]; then
    log "基座目录已存在, 跳过生成: $DUPLEX_BASE_OUT"
else
    CUDA_VISIBLE_DEVICES='' $PY $HERE/make_duplex_base.py \
        --base "$S2_CKPT" --out "$DUPLEX_BASE_OUT"
fi

# ---------- 3. 正式训练 (checkpoint 与其他训练平级) ----------
wait_burn_done   # 若 gpu_burn 正在烧卡, 等它结束再启动 (防 OOM)
log "启动全双工训练 -> $DUPLEX_CKPT_OUT (362 step ≈ 1 epoch)"
mkdir -p "$DUPLEX_CKPT_OUT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
$PY -m torch.distributed.run --nproc_per_node 8 $HERE/train_duplex.py \
    --model "$DUPLEX_BASE_OUT" \
    --output "$DUPLEX_CKPT_OUT" \
    --max-steps 362 --lr 1e-5 --save-steps 100
log "全部完成: stage2 -> 全双工续训一气呵成, checkpoint 在 $DUPLEX_CKPT_OUT"

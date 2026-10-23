#!/bin/bash
# ===== v4 训练 pipeline: 3 个任务串行 =====
#
# 已训好 (不重训): s2v3_hist1/v0-20260925-055214/checkpoint-2900 (stage2 多轮 hist1)
# 旧半成品已由用户手动删除
#
# 任务 (串行):
#   1. stage2 flat3: 单轮拍平 3图连排 (无新token, 官方基座直接训, 2900步)
#   2. 全双工 v4:    从已扩词表 checkpoint-2900-v4vocab 续训 (600步)
#   3. 全双工 v4 actonly: 同一起点, 数据换 actonly (600步, 与 2 对照)
#
# 每个任务结束后校验最终 checkpoint: rope_scaling(mrope) 存在 + vocab 正确
# 用法 (GPU 机): setsid nohup bash pipeline_v4_train.sh > /dev/null 2>&1 &
set -e
set -o pipefail

CKPT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint
BASE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B
DATA=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset
TRAIN_SH=/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/train_v2_stage.sh
PY=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/JarvisVLA/bin/python

# 全双工共用起点: 已训好的 stage2 hist2 (3图, 与flat3/全双工对齐) + 扩词表
# 扩词表产物若未就绪则等待 (expand_hist2.log 的后台任务, 最多60分钟)
S2_HIST2_VOCAB=$CKPT/s2v3_hist2/v0-20260929-011248/checkpoint-2900-v4vocab
LOG=/apdcephfs/private_chengzhixu/pipeline_v4_train.log
for i in $(seq 1 120); do
  [ -f "$S2_HIST2_VOCAB/model.safetensors" ] && break
  sleep 30
done
for f in config.json model.safetensors tokenizer.json; do
  [ -f "$S2_HIST2_VOCAB/$f" ] || { echo "起点 checkpoint 缺 $f: $S2_HIST2_VOCAB" >> $LOG; exit 1; }
done
DATA_V4=$DATA/duplex_vla_v4/duplex_train_filtered.jsonl
DATA_ACT=$DATA/duplex_vla_v4_actonly/duplex_train_filtered.jsonl
DATA_FLAT3=$DATA/duplex_vla_v4/stage2_train_flat3.jsonl
for f in "$DATA_V4" "$DATA_ACT" "$DATA_FLAT3"; do
  [ -f "$f" ] || { echo "数据文件不存在: $f" >> $LOG; exit 1; }
done

# 起点扩词表自检: vocab_size=151911 + 恰好 3 个新 token
$PY -c "
import json
cfg = json.load(open('$S2_HIST2_VOCAB/config.json'))
assert cfg.get('vocab_size') == 151911, 'vocab_size: %s' % cfg.get('vocab_size')
assert cfg.get('rope_scaling'), 'rope_scaling 缺失'
tj = json.load(open('$S2_HIST2_VOCAB/tokenizer.json'))
ids = {t['content'] for t in tj['added_tokens'] if t['id'] >= 151908}
assert ids == {'<|silence|>', '<|obs_user|>', '<|obs_asst|>'}, '新token: %s' % ids
print('起点校验通过')
" >> $LOG 2>&1 || exit 1

# 训练产物处理函数: 从官方基座复制 rope_scaling 修复 config (swift 保存必丢), 并校验 vocab
#   $1=checkpoint目录  $2=vocab模式 (151911=扩词表版 / plain=未扩版)
fix_and_check_ckpt() {
  local C=$1 MODE=$2
  $PY - "$C" "$MODE" "$BASE" <<'PYEOF'
import json, sys, os
ck, mode, src_base = sys.argv[1], sys.argv[2], sys.argv[3]
cfg_p = os.path.join(ck, 'config.json')
assert os.path.exists(cfg_p), '缺 config.json: %s' % ck
cfg = json.load(open(cfg_p))

# mrope 修复: 从官方基座复制 rope_scaling (顶层 + text_config), swift 保存必丢
src_cfg = json.load(open(os.path.join(src_base, 'config.json')))
def get_rope(d):
    rs = (d or {}).get('rope_scaling')
    return rs if isinstance(rs, dict) and 'mrope_section' in rs else None
src_top = get_rope(src_cfg)
src_tc = get_rope(src_cfg.get('text_config')) or src_top
fixed = []
if src_top and not get_rope(cfg):
    cfg['rope_scaling'] = src_top
    fixed.append('rope_scaling顶层')
tc = cfg.get('text_config')
if isinstance(tc, dict) and src_tc and not get_rope(tc):
    tc['rope_scaling'] = src_tc
    fixed.append('rope_scaling.text_config')
if fixed:
    json.dump(cfg, open(cfg_p, 'w'), indent=2, ensure_ascii=False)

# vocab/tokenizer 校验
errs = []
if mode == '151911':
    if cfg.get('vocab_size') != 151911:
        errs.append('vocab_size=%s 应为 151911' % cfg.get('vocab_size'))
    tj = json.load(open(os.path.join(ck, 'tokenizer.json')))
    ids = {t['content'] for t in tj['added_tokens'] if t['id'] >= 151908}
    if ids != {'<|silence|>', '<|obs_user|>', '<|obs_asst|>'}:
        errs.append('新token集合错: %s' % ids)
else:
    tj = json.load(open(os.path.join(ck, 'tokenizer.json')))
    ids = {t['content'] for t in tj['added_tokens'] if t['id'] >= 151908}
    if ids:
        errs.append('不应有新token 却存在: %s' % ids)
assert not errs, '; '.join(errs)
print(('修复 %s; ' % ','.join(fixed) if fixed else '无需修复; ') + '校验通过: %s' % ck)
PYEOF
}

# 取目录下最大 step 的 checkpoint-* (排除带后缀的衍生目录)
last_ckpt() {
  ls -d "$1"/checkpoint-* 2>/dev/null | grep -vE 'vocab|qwen2-vl' | sort -t- -k2 -n | tail -1
}

LOG=/apdcephfs/private_chengzhixu/pipeline_v4_train.log
echo "== pipeline start $(date) ==" >> $LOG

# wandb 上报 (三个任务共用项目, 各自 run)
export REPORT_TO=wandb
export WANDB_PROJECT=v4-pipeline

# ---------- 1. stage2 flat3 (官方基座, 无新token) ----------
# SKIP_TASK1=1: 任务1已训练完成 (2026-10-01 断点续跑), 跳过训练只补修复校验
if [ "${SKIP_TASK1:-0}" = "1" ]; then
  OUT1=$CKPT/s2v4_flat3/v0-20260930-225227/v0-20260930-225519
  echo "[1/3] 跳过训练 (已完成), 目录: $OUT1" >> $LOG
else
  OUT1=$CKPT/s2v4_flat3/v0-$(date +%Y%m%d-%H%M%S)
  echo "[1/3] stage2 flat3: $BASE -> $OUT1 $(date)" >> $LOG
  # 注意: 不用 set -e 语义, 以产物为准 (训练收尾 wandb/NCCL 可能非零退出但产物完好)
  bash $TRAIN_SH $BASE $DATA_FLAT3 $OUT1 2900 1e-5 last_round >> $LOG 2>&1 || \
    echo "[1/3] 训练脚本退出码非零, 以 checkpoint 产物为准判断" >> $LOG
fi
CK1=$(last_ckpt "$OUT1")
[ -n "$CK1" ] || { echo "[1/3] 未找到 checkpoint, 中止" >> $LOG; exit 1; }
fix_and_check_ckpt "$CK1" plain >> $LOG 2>&1 || { echo "[1/3] checkpoint 校验失败: $CK1" >> $LOG; exit 1; }
echo "[1/3] 完成 $CK1 $(date)" >> $LOG

# ---------- 2. 全双工 v4 (hist2-v4vocab 起点) ----------
OUT2=$CKPT/duplex_v4_hist2/v0-$(date +%Y%m%d-%H%M%S)
echo "[2/3] 全双工 v4: $S2_HIST2_VOCAB -> $OUT2 $(date)" >> $LOG
bash $TRAIN_SH $S2_HIST2_VOCAB $DATA_V4 $OUT2 600 5e-6 last_round >> $LOG 2>&1 || \
  echo "[2/3] 训练脚本退出码非零, 以 checkpoint 产物为准判断" >> $LOG
CK2=$(last_ckpt "$OUT2")
[ -n "$CK2" ] || { echo "[2/3] 未找到 checkpoint, 中止" >> $LOG; exit 1; }
fix_and_check_ckpt "$CK2" 151911 >> $LOG 2>&1 || { echo "[2/3] checkpoint 校验失败: $CK2" >> $LOG; exit 1; }
echo "[2/3] 完成 $CK2 $(date)" >> $LOG

# ---------- 3. 全双工 v4 actonly (同起点对照) ----------
OUT3=$CKPT/duplex_v4_actonly_hist2/v0-$(date +%Y%m%d-%H%M%S)
echo "[3/3] 全双工 v4 actonly: $S2_HIST2_VOCAB -> $OUT3 $(date)" >> $LOG
bash $TRAIN_SH $S2_HIST2_VOCAB $DATA_ACT $OUT3 600 5e-6 last_round >> $LOG 2>&1 || \
  echo "[3/3] 训练脚本退出码非零, 以 checkpoint 产物为准判断" >> $LOG
CK3=$(last_ckpt "$OUT3")
[ -n "$CK3" ] || { echo "[3/3] 未找到 checkpoint, 中止" >> $LOG; exit 1; }
fix_and_check_ckpt "$CK3" 151911 >> $LOG 2>&1 || { echo "[3/3] checkpoint 校验失败: $CK3" >> $LOG; exit 1; }
echo "[3/3] 完成 $CK3 $(date)" >> $LOG

echo "== pipeline done $(date) ==" >> $LOG

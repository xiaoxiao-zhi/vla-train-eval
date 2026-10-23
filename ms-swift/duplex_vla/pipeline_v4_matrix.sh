#!/bin/bash
# ===== v4 矩阵续跑: 补齐 2x2 中 flat3 起点的两格 =====
#
# 矩阵结构 (stage2 两种 x 全双工两种):
#   hist1 x v4       -> 老 pipeline [2/3] 正在跑
#   hist1 x actonly  -> 老 pipeline [3/3] 会跑
#   flat3 x v4       -> 本脚本任务1
#   flat3 x actonly  -> 本脚本任务2
#
# 逻辑: 等老 pipeline 完全退出 -> (若QA已被拉起先杀, 断点续跑无损失)
#       -> 校验 flat3 扩词表产物 -> flat3 x v4 -> flat3 x actonly -> 启动 QA
#
# 用法 (GPU 机): setsid nohup bash pipeline_v4_matrix.sh > /dev/null 2>&1 &
set -u

CKPT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint
BASE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B
DATA=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset
TRAIN_SH=/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/train_v2_stage.sh
PY=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/JarvisVLA/bin/python

FLAT3_VOCAB=$CKPT/s2v4_flat3/v0-20260930-225227/v0-20260930-225519/checkpoint-2900-v4vocab
DATA_V4=$DATA/duplex_vla_v4/duplex_train_filtered.jsonl
DATA_ACT=$DATA/duplex_vla_v4_actonly/duplex_train_filtered.jsonl
for f in "$DATA_V4" "$DATA_ACT"; do
  [ -f "$f" ] || { echo "数据文件不存在: $f" >&2; exit 1; }
done

LOG=/apdcephfs/private_chengzhixu/pipeline_v4_train.log
echo "== matrix start $(date) ==" >> $LOG

# ---------- 0. 等老 pipeline 退出 (hist1 两格跑完) ----------
while pgrep -f pipeline_v4_train > /dev/null; do sleep 120; done
echo "[matrix] 老 pipeline 已退出 $(date)" >> $LOG

# 老 pipeline 收尾会拉起 QA 采样 -> 杀掉 (训练优先, QA 最后断点续跑)
pkill -f sample_qa_replies 2>/dev/null; sleep 2
pkill -f auto_qa_sample 2>/dev/null; sleep 2
pkill -f 'vllm serve' 2>/dev/null; sleep 5
echo "[matrix] 已清理 QA 相关进程 (如有)" >> $LOG

# ---------- 1. 校验 flat3 扩词表产物 ----------
for f in config.json model.safetensors tokenizer.json; do
  [ -f "$FLAT3_VOCAB/$f" ] || { echo "[matrix] 扩词表产物缺 $f, 中止 (查看 /apdcephfs/private_chengzhixu/expand_flat3.log)" >> $LOG; exit 1; }
done
$PY -c "
import json
cfg = json.load(open('$FLAT3_VOCAB/config.json'))
assert cfg.get('vocab_size') == 151911, 'vocab_size: %s' % cfg.get('vocab_size')
assert cfg.get('rope_scaling'), 'rope_scaling 缺失'
tj = json.load(open('$FLAT3_VOCAB/tokenizer.json'))
ids = {t['content'] for t in tj['added_tokens'] if t['id'] >= 151908}
assert ids == {'<|silence|>', '<|obs_user|>', '<|obs_asst|>'}, '新token: %s' % ids
print('flat3 扩词表校验通过')
" >> $LOG 2>&1 || { echo "[matrix] 扩词表校验失败, 中止" >> $LOG; exit 1; }
echo "[matrix] flat3-v4vocab 就绪: $FLAT3_VOCAB" >> $LOG

# wandb 上报
export REPORT_TO=wandb
export WANDB_PROJECT=v4-pipeline

last_ckpt() {
  ls -d "$1"/checkpoint-* 2>/dev/null | grep -vE 'vocab|qwen2-vl' | sort -t- -k2 -n | tail -1
}

# 训练产物处理: mrope 修复 (从官方基座复制) + vocab 校验
fix_and_check() {
  local C=$1
  $PY - "$C" "$BASE" <<'PYEOF'
import json, sys, os
ck, src_base = sys.argv[1], sys.argv[2]
cfg_p = os.path.join(ck, 'config.json')
cfg = json.load(open(cfg_p))
src_cfg = json.load(open(os.path.join(src_base, 'config.json')))
def get_rope(d):
    rs = (d or {}).get('rope_scaling')
    return rs if isinstance(rs, dict) and 'mrope_section' in rs else None
src_top = get_rope(src_cfg)
src_tc = get_rope(src_cfg.get('text_config')) or src_top
fixed = []
if src_top and not get_rope(cfg):
    cfg['rope_scaling'] = src_top; fixed.append('顶层')
tc = cfg.get('text_config')
if isinstance(tc, dict) and src_tc and not get_rope(tc):
    tc['rope_scaling'] = src_tc; fixed.append('text_config')
if fixed:
    json.dump(cfg, open(cfg_p, 'w'), indent=2, ensure_ascii=False)
assert cfg.get('vocab_size') == 151911, 'vocab_size: %s' % cfg.get('vocab_size')
print(('修复%s; ' % ','.join(fixed) if fixed else '无需修复; ') + '校验通过: %s' % ck)
PYEOF
}

# ---------- 2. flat3 x v4 ----------
OUT_A=$CKPT/duplex_v4_flat3/v0-$(date +%Y%m%d-%H%M%S)
echo "[matrix 1/2] flat3 x v4: $FLAT3_VOCAB -> $OUT_A $(date)" >> $LOG
bash $TRAIN_SH $FLAT3_VOCAB $DATA_V4 $OUT_A 600 5e-6 last_round >> $LOG 2>&1 || \
  echo "[matrix 1/2] 训练脚本退出码非零, 以 checkpoint 产物为准判断" >> $LOG
CKA=$(last_ckpt "$OUT_A")
[ -n "$CKA" ] || { echo "[matrix 1/2] 未找到 checkpoint, 中止" >> $LOG; exit 1; }
fix_and_check "$CKA" >> $LOG 2>&1 || { echo "[matrix 1/2] 校验失败: $CKA" >> $LOG; exit 1; }
echo "[matrix 1/2] 完成 $CKA $(date)" >> $LOG

# ---------- 3. flat3 x actonly ----------
OUT_B=$CKPT/duplex_v4_actonly_flat3/v0-$(date +%Y%m%d-%H%M%S)
echo "[matrix 2/2] flat3 x actonly: $FLAT3_VOCAB -> $OUT_B $(date)" >> $LOG
bash $TRAIN_SH $FLAT3_VOCAB $DATA_ACT $OUT_B 600 5e-6 last_round >> $LOG 2>&1 || \
  echo "[matrix 2/2] 训练脚本退出码非零, 以 checkpoint 产物为准判断" >> $LOG
CKB=$(last_ckpt "$OUT_B")
[ -n "$CKB" ] || { echo "[matrix 2/2] 未找到 checkpoint, 中止" >> $LOG; exit 1; }
fix_and_check "$CKB" >> $LOG 2>&1 || { echo "[matrix 2/2] 校验失败: $CKB" >> $LOG; exit 1; }
echo "[matrix 2/2] 完成 $CKB $(date)" >> $LOG

# ---------- 4. 全部训练结束 -> 启动 QA 采样 ----------
echo "[matrix] 启动 QA 采样 $(date)" >> $LOG
setsid nohup bash /apdcephfs/private_chengzhixu/auto_qa_sample.sh > /dev/null 2>&1 < /dev/null &
echo "[matrix] QA 采样已后台启动 (断点续跑)" >> $LOG
echo "== matrix done $(date) ==" >> $LOG

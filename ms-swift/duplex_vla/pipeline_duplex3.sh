#!/bin/bash
# ===== 全双工三连训: hist2 起点 x 三个数据集 (只换数据, 其余不变) =====
#
# 任务0: 补 hist2 x v4 (duplex_v4_hist2 checkpoint-600) 的 mrope 修复+校验
#        (上次 pipeline 中断漏掉的一步)
# 任务1: duplex_train_actonly_filtered.jsonl          -> duplex_v4_actonly_hist2
# 任务2: duplex_train_histact_filtered.jsonl          -> duplex_v4_histact_hist2
# 任务3: duplex_train_histact_actonly_filtered.jsonl  -> duplex_v4_histact_actonly_hist2
#
# 训练参数与已完成的 hist2 x v4 完全一致: 600步 / lr 5e-6 / loss_scale last_round
# 起点: s2v3_hist2 checkpoint-2900-v4vocab (151911 词表)
# 本脚本自包含, 运行期间不会被修改 (上次教训: 运行中改脚本导致 bash 错位中断)
#
# 用法 (GPU 机): setsid nohup bash pipeline_duplex3.sh > /dev/null 2>&1 &
set -u

CKPT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint
BASE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B
DATA=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset
TRAIN_SH=/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/train_v2_stage.sh
PY=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/JarvisVLA/bin/python

S2=$CKPT/s2v3_hist2/v0-20260929-011248/checkpoint-2900-v4vocab
DATA_ACTONLY=$DATA/duplex_vla_v4/duplex_train_actonly_filtered.jsonl
DATA_HISTACT=$DATA/duplex_vla_v4/duplex_train_histact_filtered.jsonl
DATA_HISTACT_ACT=$DATA/duplex_vla_v4/duplex_train_histact_actonly_filtered.jsonl

LOG=/apdcephfs/private_chengzhixu/pipeline_duplex3.log

# ---------- 启动前检查 (缺一样直接退出, 不进入半途状态) ----------
err=0
for f in "$S2/model.safetensors" "$S2/config.json" "$S2/tokenizer.json" \
         "$DATA_ACTONLY" "$DATA_HISTACT" "$DATA_HISTACT_ACT"; do
  if [ ! -f "$f" ]; then echo "缺文件: $f" >> $LOG; err=1; fi
done
[ $err -eq 0 ] || { echo "== 启动前检查失败, 退出 $(date) ==" >> $LOG; exit 1; }
$PY -c "
import json
cfg = json.load(open('$S2/config.json'))
assert cfg.get('vocab_size') == 151911, 'vocab: %s' % cfg.get('vocab_size')
assert cfg.get('rope_scaling'), 'rope_scaling 缺失'
print('起点校验通过')
" >> $LOG 2>&1 || { echo "== 起点校验失败, 退出 ==" >> $LOG; exit 1; }

echo "== duplex3 start $(date) ==" >> $LOG

# mrope 修复 (从官方基座复制) + vocab 校验; 失败 -> 中止 (词表错=白训)
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
# vocab 校验: 以 tokenizer 实际词表为准 (swift 复合 config 的 vocab_size 可能在
# text_config 里或缺失, 不可靠); 要求恰好 3 个新 token
tj = json.load(open(os.path.join(ck, 'tokenizer.json')))
ids = {t['content'] for t in tj['added_tokens'] if t['id'] >= 151908}
assert ids == {'<|silence|>', '<|obs_user|>', '<|obs_asst|>'}, '新token: %s' % ids
tc_v = (cfg.get('text_config') or {}).get('vocab_size')
top_v = cfg.get('vocab_size')
assert 151911 in {top_v, tc_v}, 'vocab_size 顶层/text_config 均非151911: %s/%s' % (top_v, tc_v)
print(('修复%s; ' % ','.join(fixed) if fixed else '无需修复; ') + '校验通过: %s' % ck)
PYEOF
}

last_ckpt() {
  # swift 可能把 checkpoint 存外层或内层 v0-*/ 下, 两处都找 (取最大 step)
  local C
  C=$(ls -d "$1"/checkpoint-* 2>/dev/null | grep -vE 'vocab|qwen2-vl' | sort -t- -k2 -n | tail -1)
  if [ -z "$C" ]; then
    C=$(ls -d "$1"/v0-*/checkpoint-* 2>/dev/null | grep -vE 'vocab|qwen2-vl' | sort -t- -k2 -n | tail -1)
  fi
  echo "$C"
}

export REPORT_TO=wandb
export WANDB_PROJECT=v4-pipeline

run_task() {
  local TAG=$1 DATAF=$2 OUTBASE=$3
  local OUT=$CKPT/$OUTBASE/v0-$(date +%Y%m%d-%H%M%S)
  echo "[$TAG] $S2 -> $OUT (数据: $DATAF) $(date)" >> $LOG
  bash $TRAIN_SH $S2 "$DATAF" $OUT 600 5e-6 last_round >> $LOG 2>&1 || \
    echo "[$TAG] 训练脚本退出码非零, 以 checkpoint 产物为准判断" >> $LOG
  local CK=$(last_ckpt "$OUT")
  if [ -z "$CK" ]; then
    echo "[$TAG] !! 未找到 checkpoint, 中止 (数据/日志见 $LOG)" >> $LOG
    exit 1
  fi
  fix_and_check "$CK" >> $LOG 2>&1 || { echo "[$TAG] !! checkpoint 校验失败: $CK" >> $LOG; exit 1; }
  echo "[$TAG] 完成 $CK $(date)" >> $LOG
}

# ---------- 任务0: 补已完成产物的修复校验 (hist2 x v4 + hist2 x actonly) ----------
for CK0 in $CKPT/duplex_v4_hist2/v0-20261008-141334/v0-20261008-141620/checkpoint-600 \
           $CKPT/duplex_v4_actonly_hist2/v0-20261008-220902/v0-20261008-221155/checkpoint-600; do
  if [ -f "$CK0/config.json" ]; then
    echo "[0] 修复校验已完成产物: $CK0 $(date)" >> $LOG
    fix_and_check "$CK0" >> $LOG 2>&1 || { echo "[0] 校验失败 ($CK0), 中止" >> $LOG; exit 1; }
  else
    echo "[0] 跳过 (产物不存在: $CK0)" >> $LOG
  fi
done
echo "[0] 完成 $(date)" >> $LOG

# ---------- 任务1 已于 10/9 05:50 训完 (checkpoint-600), 不重跑 ----------
# run_task "1/3 actonly" "$DATA_ACTONLY" "duplex_v4_actonly_hist2"

run_task "2/3 histact"         "$DATA_HISTACT"     "duplex_v4_histact_hist2"
run_task "3/3 histact_actonly" "$DATA_HISTACT_ACT" "duplex_v4_histact_actonly_hist2"

echo "== duplex3 done $(date) ==" >> $LOG

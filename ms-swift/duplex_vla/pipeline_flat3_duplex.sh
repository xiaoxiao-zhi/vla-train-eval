#!/bin/bash
# ===== flat3 重训 -> 扩词表 -> 全双工续训(无历史动作版) =====
#
# 链路: 官方基座 --(stage2_train_flat3, 2900步)--> flat3新ckpt
#       --(扩词表+3token)--> v4vocab
#       --(duplex_train_filtered, 600步)--> 全双工最终模型
#
# 防找错checkpoint措施 (针对历史事故):
#   1. 全新目录名 (s2v4_flat3_r2 / duplex_v4_flat3base), 与旧目录零混淆
#   2. checkpoint 用递归 find 取 max-step (抗 swift 版本目录多层嵌套), 排除 vocab/qwen2-vl 衍生
#   3. 每阶段交接断言 vocab_size/token集合/rope_scaling, 不符立即中止 (不静默)
#   4. 每步把实际使用的绝对路径写入日志, 事后可审计
#   5. 最终 ckpt 自动打服务化补丁 (chat_template + max_pixels, 今日 vLLM 踩坑项)
set -u

CKPT=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/checkpoint
BASE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/HF_MODEL/JarvisVLA-Qwen2-VL-7B
DATA=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/dataset
TRAIN_SH=/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/train_v2_stage.sh
EXPAND=/apdcephfs/private_chengzhixu/ms-swift/duplex_vla/expand_ckpt_vocab_v4.py
PY=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu/conda_envs/JarvisVLA/bin/python

DATA_FLAT3=$DATA/duplex_vla_v4/stage2_train_flat3.jsonl
DATA_DUPLEX=$DATA/duplex_vla_v4/duplex_train_filtered.jsonl
LOG=/apdcephfs/private_chengzhixu/pipeline_flat3_duplex.log

for f in "$DATA_FLAT3" "$DATA_DUPLEX" "$BASE/config.json" "$TRAIN_SH" "$EXPAND"; do
  [ -f "$f" ] || { echo "输入缺失: $f" >> $LOG; exit 1; }
done

# 递归找 max-step checkpoint (排除带后缀衍生目录)
last_ckpt() {
  find "$1" -maxdepth 4 -type d -name 'checkpoint-*' 2>/dev/null | \
    grep -vE 'vocab|qwen2-vl' | sort -t- -k2 -n | tail -1
}

# 产物校验+修复: rope_scaling 回补 + vocab 断言 ($1=ckpt $2=plain|151911)
check_ckpt() {
  $PY - "$1" "$2" "$BASE" <<'PYEOF'
import json, sys, os
ck, mode, src_base = sys.argv[1], sys.argv[2], sys.argv[3]
# rope_scaling 修复 (swift 保存必丢)
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
# vocab 断言
errs = []
if mode == '151911':
    if cfg.get('vocab_size') != 151911:
        errs.append('vocab_size=%s 应151911' % cfg.get('vocab_size'))
    tj = json.load(open(os.path.join(ck, 'tokenizer.json')))
    ids = {t['content'] for t in tj['added_tokens'] if t['id'] >= 151908}
    if ids != {'<|silence|>', '<|obs_user|>', '<|obs_asst|>'}:
        errs.append('新token集合错: %s' % ids)
else:
    tj = json.load(open(os.path.join(ck, 'tokenizer.json')))
    ids = {t['content'] for t in tj['added_tokens'] if t['id'] >= 151908}
    if ids:
        errs.append('plain版不应有新token: %s' % ids)
assert not errs, '; '.join(errs)
# 权重断言: 分片感知 (单文件 或 index+shards)
_w = (os.path.exists(os.path.join(ck, 'model.safetensors'))
      or os.path.exists(os.path.join(ck, 'model.safetensors.index.json')))
assert _w, '缺权重文件(单文件/分片index均无)'
print('校验通过%s: %s' % (('(修复rope_scaling:' + ','.join(fixed) + ')') if fixed else '', ck))
PYEOF
}

echo "== pipeline start $(date) ==" >> $LOG

# wandb 上报 (三个任务各自 run, 同一 project)
export REPORT_TO=wandb
export WANDB_PROJECT=v4-pipeline

# ---------- 1/3 stage2 flat3 重训 (官方基座, 2900步) ----------
# SKIP_S1=1: stage1 已训完 (2026-10-11), 跳过训练只做校验
if [ "${SKIP_S1:-0}" = "1" ]; then
  OUT1=$CKPT/s2v4_flat3_r2
  echo "[1/3] 跳过训练 (已完成), 目录: $OUT1" >> $LOG
else
  OUT1=$CKPT/s2v4_flat3_r2/v0-$(date +%Y%m%d-%H%M%S)
  echo "[1/3] stage2 flat3 重训: $BASE -> $OUT1 $(date)" >> $LOG
  bash $TRAIN_SH $BASE $DATA_FLAT3 $OUT1 2900 1e-5 last_round >> $LOG 2>&1 || \
    echo "[1/3] 训练退出码非零, 以 checkpoint 产物为准判断" >> $LOG
fi
CK1=$(last_ckpt "$OUT1")
[ -n "$CK1" ] || { echo "[1/3] 未找到checkpoint: $OUT1 中止 $(date)" >> $LOG; exit 1; }
echo "[1/3] 实际checkpoint: $CK1" >> $LOG
check_ckpt "$CK1" plain >> $LOG 2>&1 || { echo "[1/3] 校验失败: $CK1 中止" >> $LOG; exit 1; }
echo "[1/3] 完成 $(date)" >> $LOG

# ---------- 2/3 扩词表 (+3 token -> 151911) ----------
CK1V=$CK1-v4vocab
echo "[2/3] 扩词表: $CK1 -> $CK1V $(date)" >> $LOG
$PY $EXPAND "$CK1" "$CK1V" >> $LOG 2>&1 || { echo "[2/3] 扩词表脚本失败 中止" >> $LOG; exit 1; }
{ [ -f "$CK1V/model.safetensors" ] || [ -f "$CK1V/model.safetensors.index.json" ]; } || \
  { echo "[2/3] 扩词表产物缺失: $CK1V 中止" >> $LOG; exit 1; }
check_ckpt "$CK1V" 151911 >> $LOG 2>&1 || { echo "[2/3] 校验失败: $CK1V 中止" >> $LOG; exit 1; }
echo "[2/3] 完成: $CK1V $(date)" >> $LOG

# ---------- 3/3 全双工续训 (无历史动作数据, 600步) ----------
OUT2=$CKPT/duplex_v4_flat3base/v0-$(date +%Y%m%d-%H%M%S)
echo "[3/3] 全双工续训: $CK1V -> $OUT2 (duplex_train_filtered) $(date)" >> $LOG
bash $TRAIN_SH "$CK1V" $DATA_DUPLEX "$OUT2" 600 5e-6 last_round >> $LOG 2>&1 || \
  echo "[3/3] 训练退出码非零, 以 checkpoint 产物为准判断" >> $LOG
CK2=$(last_ckpt "$OUT2")
[ -n "$CK2" ] || { echo "[3/3] 未找到checkpoint: $OUT2 中止 $(date)" >> $LOG; exit 1; }
echo "[3/3] 实际checkpoint: $CK2" >> $LOG
check_ckpt "$CK2" 151911 >> $LOG 2>&1 || { echo "[3/3] 校验失败: $CK2 中止" >> $LOG; exit 1; }

# ---------- 服务化补丁 (swift 存盘缺 chat_template/max_pixels, 今日vLLM踩坑项) ----------
$PY - "$CK2" "$BASE" <<'PYEOF'
import json, shutil, sys, os
ck, base = sys.argv[1], sys.argv[2]
p = os.path.join(ck, 'tokenizer_config.json')
shutil.copy(p, p + '.pre_patch.bak')
d = json.load(open(p))
d['chat_template'] = json.load(open(os.path.join(base, 'tokenizer_config.json')))['chat_template']
json.dump(d, open(p, 'w'), ensure_ascii=False, indent=2)
p2 = os.path.join(ck, 'processor_config.json')
shutil.copy(p2, p2 + '.pre_patch.bak')
d2 = json.load(open(p2))
d2.setdefault('image_processor', {})
d2['image_processor']['max_pixels'] = 1605632
d2['image_processor']['min_pixels'] = 3136
json.dump(d2, open(p2, 'w'), indent=2)
print('服务化补丁完成 (chat_template + max_pixels):', ck)
PYEOF
>> $LOG 2>&1

echo "== pipeline done $(date) ==" >> $LOG
echo "== 最终全双工模型: $CK2 ==" >> $LOG

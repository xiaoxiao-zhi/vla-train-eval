#!/bin/bash
set -euo pipefail

ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
BASE=$SHARE/HF_MODEL/JarvisVLA-Qwen2-VL-7B
DATA=$SHARE/dataset/duplex_vla_v4/stage2_train_flat3.jsonl
TRAIN_SH=$ROOT/ms-swift/duplex_vla/train_v2_stage.sh
LOSS_FILE=$ROOT/ms-swift/mcvla/camera_distance_loss.py
PY=$SHARE/conda_envs/swift/bin/python
OUT_ROOT=$SHARE/checkpoint/s2v4_flat3_top5weighted

mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/train.lock"
flock -n 9 || { echo '已有同实验训练脚本在运行'; exit 1; }

for file in "$BASE/config.json" "$DATA" "$TRAIN_SH" "$LOSS_FILE" "$PY"; do
    [ -f "$file" ] || { echo "输入文件缺失: $file"; exit 1; }
done
if pgrep -f '[r]un_train.py|[v]llm serve|[g]pu_burn.py' >/dev/null; then
    echo '发现现有训练、推理或占卡进程，停止启动以避免资源冲突'
    exit 1
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$GPU_COUNT" -eq 8 ] || { echo "需要8卡，当前为$GPU_COUNT"; exit 1; }
MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
[ "$MAX_USED" -lt 2000 ] || { echo "GPU仍有显存占用: ${MAX_USED}MiB，停止启动"; exit 1; }

OUT=$(mktemp -d "$OUT_ROOT/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")
exec >>"$OUT/launch.log" 2>&1
export REPORT_TO=wandb
export WANDB_PROJECT=v4-pipeline
export WANDB_MODE=online
export WANDB_NAME="flat3-top5weighted-$(basename "$OUT")"
export PYTHONUNBUFFERED=1
export CAMERA_LOG_BATCH_WEIGHTS=1
export CAMERA_WEIGHT_LOG_DIR="$OUT"
unset WANDB_RUN_ID
cd "$ROOT/ms-swift"

"$PY" - "$OUT" "$BASE" "$DATA" "$LOSS_FILE" "$TRAIN_SH" <<'PYEOF'
import hashlib
import json
import os
import sys
out, base, data, loss_file, train_file = sys.argv[1:]
metadata = {
    'model': base,
    'dataset': data,
    'output_root': out,
    'loss_type': 'camera_weighted',
    'camera_weighting': {
        'top_k': 5,
        'candidate_space': 'full_vocabulary',
        'normalize_top5_probabilities': True,
        'invalid_or_cross_axis_distance': 20,
        'distance': 'absolute_bin_difference',
        'weight': '0.5 + max(mean_distance - 1, 0) * 1.5 / 19',
        'apply': 'multiply_camera_token_cross_entropy',
        'detach_weights': True,
        'noncamera_target_weight': 1.0,
        'batch_weight_logging': {
            'enabled': True,
            'directory': os.path.join(out, 'batch_weights'),
            'format': 'jsonl_one_microbatch_per_line_per_rank',
            'group_by': ['phase', 'optimizer_step_target', 'rank', 'micro_batch_in_step'],
        },
    },
    'max_steps': 2900,
    'learning_rate': 1e-5,
    'loss_scale': 'last_round',
    'global_batch_size': 256,
    'report_to': 'wandb',
    'wandb_project': os.environ['WANDB_PROJECT'],
    'wandb_name': os.environ['WANDB_NAME'],
    'source_sha256': {},
}
for path in (loss_file, train_file):
    with open(path, 'rb') as f:
        metadata['source_sha256'][path] = hashlib.sha256(f.read()).hexdigest()
with open(os.path.join(out, 'experiment.json'), 'w') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
PYEOF

echo "训练开始 $(date): $OUT"
set +e
bash "$TRAIN_SH" "$BASE" "$DATA" "$OUT" 2900 1e-5 last_round camera_weighted
TRAIN_RC=$?
set -e
echo "训练退出码=$TRAIN_RC $(date)"

"$PY" - "$OUT" "$BASE" "$DATA" "$TRAIN_RC" <<'PYEOF'
import json
import os
from pathlib import Path
import sys
out, base, data, returncode = sys.argv[1:]
candidates = list(Path(out).rglob('checkpoint-2900'))
candidates = [p for p in candidates if p.is_dir()]
assert len(candidates) == 1, '本次运行的checkpoint-2900不唯一或缺失: %s' % candidates
ck = candidates[0]
state = json.loads((ck / 'trainer_state.json').read_text())
assert state['global_step'] == 2900, '训练未达到2900步'
args = json.loads((ck / 'args.json').read_text())
assert args.get('model') == base, '模型起点不匹配'
assert args.get('dataset') in ([data], data), '训练数据不匹配'
assert args.get('loss_type') == 'camera_weighted', '未启用camera_weighted'
assert args.get('max_steps') == 2900, '训练步数不匹配'
assert 'wandb' in args.get('report_to', []), '未配置wandb'
index = ck / 'model.safetensors.index.json'
if index.exists():
    weight_map = json.loads(index.read_text())['weight_map']
    shards = set(weight_map.values())
    assert shards, '空权重索引'
else:
    shards = {'model.safetensors'}
for name in shards:
    path = ck / name
    assert path.is_file() and path.stat().st_size > 0, '缺失或空权重: %s' % path
with open(os.path.join(out, 'training_result.json'), 'w') as f:
    json.dump({'checkpoint': str(ck), 'global_step': 2900,
               'training_exit_code': int(returncode), 'weights_verified': True}, f, indent=2)
print('最终checkpoint: %s' % ck, flush=True)
PYEOF
exit "$TRAIN_RC"

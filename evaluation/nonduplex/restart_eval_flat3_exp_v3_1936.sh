#!/bin/bash
set -euo pipefail
ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
TAG=flat3_exp_v3_1936
MODEL=$ROOT/eval_model_views/$TAG
BASE=$ROOT/JarvisVLA/outputs/rollout/eval_4tasks_s2v3
CODE=/apdcephfs/private_chengzhixu/evaluation/nonduplex
PY=$SHARE/conda_envs/JarvisVLA/bin/python
BASE_URLS=${1:-${EVAL_BASE_URLS:-}}
[ -n "$BASE_URLS" ] || { echo 'Usage: restart_eval_flat3_exp_v3_1936.sh <comma-separated vLLM base URLs>'; exit 1; }
BASE_URL=${BASE_URLS%%,*}
LOG=$BASE/_logs/${TAG}_retry_pipeline.log
mkdir -p "$BASE/_logs"
exec 9>"$BASE/_logs/${TAG}_rollout.lock"
flock -n 9 || { echo 'Retry rollout already running'; exit 1; }
exec >>"$LOG" 2>&1
echo "==== $TAG rollout retry start $(date) ===="
exec 8>"$BASE/_logs/summary.lock"
flock 8
"$PY" - "$BASE" "$TAG" <<'PYEOF'
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
base, tag = Path(sys.argv[1]), sys.argv[2]
source = base / tag
if not source.is_dir():
    raise SystemExit(f'Missing failed rollout directory: {source}')
results = list(source.glob('*/ep*/result.json'))
if len(results) != 12:
    raise SystemExit(f'Expected 12 failed episodes, found {len(results)}')
for path in results:
    result = json.loads(path.read_text())
    if 'err' not in result or result.get('steps') != -1:
        raise SystemExit(f'Refusing to archive a valid episode: {path}')
archive = base / '_failed' / tag / datetime.now().strftime('%Y%m%d-%H%M%S')
archive.mkdir(parents=True, exist_ok=False)
source.rename(archive / 'episodes')
logs = archive / 'logs'
logs.mkdir()
for path in list((base / '_logs').glob(f'{tag}_*eps*.log')) + [base / '_logs' / f'{tag}_4tasks.log']:
    if path.exists():
        path.rename(logs / path.name)
summary_path = base / 'summary.json'
if summary_path.exists():
    shutil.copy2(summary_path, archive / 'summary_before_retry.json')
    summary = json.loads(summary_path.read_text())
    for row in summary.values():
        row.pop(tag, None)
    temporary = base / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=1))
    temporary.replace(summary_path)
print(f'Archived failed rollout: {archive}', flush=True)
PYEOF
flock -u 8
export DISPLAY=:99
if ! pgrep -f '^Xvfb :99 ' >/dev/null; then
    setsid nohup Xvfb :99 -screen 0 1280x720x24 </dev/null >/dev/null 2>&1 &
    sleep 2
fi
"$PY" - "$CODE" "$SHARE/dataset/duplex_vla_v4/stage2_train_flat3_v3.jsonl" "$TAG" "$BASE_URL" <<'PYEOF'
import json
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, '/apdcephfs/private_chengzhixu/evaluation')
from nonduplex import run_eval_task as evaluation
with open(sys.argv[2]) as f:
    record = next(r for r in map(json.loads, f) if r.get('images'))
model = dict(evaluation.MODELS[sys.argv[3]])
model['base_url'] = sys.argv[4]
agent = evaluation.FLAT3_AGENT(
    checkpoint_path=model['ckpt'], base_url=model['base_url'],
    temperature=0.9, history_num=2, instruction_type='normal', action_chunk_len=1,
)
agent._flat_instruction = next(m['content'] for m in record['messages'] if m['role'] == 'user').split('\nobservation:')[0]
frames = [np.asarray(Image.open(path).convert('RGB')) for path in record['images']]
assert len(frames) == 3
agent._flat_hist = [agent.processor_wrapper.create_image_input(frame) for frame in frames[:-1]]
action = agent.forward([frames[-1]], ['mine_stone'])
assert sum(item['type'] == 'image_url' for item in agent.log_inputs[-1]['messages'][0]['content']) == 3
print('SMOKE_OK', agent.base_url, agent.log_outputs[-1]['output'], action, flush=True)
PYEOF
cd "$CODE"
"$PY" eval_4tasks_group.py --out "$BASE" --models "$TAG" --workers 4 --stagger 10 \
    --base-urls "$BASE_URLS" \
    >"$BASE/_logs/${TAG}_4tasks.log" 2>&1 &
ROLLOUT=$!
RC=0
wait "$ROLLOUT" || RC=$?
echo "Rollout exit=$RC $(date)"
exit "$RC"

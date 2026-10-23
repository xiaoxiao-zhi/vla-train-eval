#!/bin/bash
set -euo pipefail
ROOT=/apdcephfs/private_chengzhixu/JarvisVLA
CODE=/apdcephfs/private_chengzhixu/evaluation/nonduplex
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
OUT=/apdcephfs/private_chengzhixu/JarvisVLA/outputs/rollout/flat3_v5_duplex_camera_exp_500_20261019-162000
TAG=flat3_v5_duplex_camera_exp_500
URLS=http://127.0.0.1:8100/v1,http://127.0.0.1:8102/v1,http://127.0.0.1:8103/v1,http://127.0.0.1:8104/v1
PY=$SHARE/conda_envs/JarvisVLA/bin/python
mkdir -p "$OUT/_logs"
exec 9>"$OUT/rerun_craft.lock"
flock -n 9 || { echo 'Craft rerun already running'; exit 1; }
exec >>"$OUT/_logs/rerun_craft.log" 2>&1
echo "==== $TAG craft rerun start $(date --iso-8601=seconds) ===="
"$PY" - "$OUT" "$TAG" <<'PYEOF'
import json
import shutil
import sys
from pathlib import Path
root = Path(sys.argv[1]) / sys.argv[2]
craft = root / 'craft_craft_crafting_table'
backup = root / 'craft_craft_crafting_table_prev_20261019'
assert craft.is_dir(), 'craft task dir not found'
results = sorted(craft.glob('ep*/result.json'))
assert len(results) == 3 and all(
    'err' not in json.loads(p.read_text()) and json.loads(p.read_text())['steps'] >= 0
    for p in results), 'existing craft results are not 3 valid episodes'
assert not backup.exists(), 'backup dir already exists'
shutil.move(str(craft), str(backup))
others = sorted(p for p in root.glob('*/ep*/result.json')
                if p.parts[-3] != 'craft_craft_crafting_table_prev_20261019')
assert len(others) == 9 and all(
    'err' not in json.loads(p.read_text()) and json.loads(p.read_text())['steps'] >= 0
    for p in others), 'other tasks must keep 9 valid results'
print('CRAFT_BACKED_UP_TO', backup, flush=True)
PYEOF
export DISPLAY=:99 EVAL_BASE_SEED=42
if ! pgrep -f '^Xvfb :99 ' >/dev/null; then
    setsid nohup Xvfb :99 -screen 0 1280x720x24 </dev/null >/dev/null 2>&1 &
    sleep 2
fi
"$PY" "$CODE/eval_4tasks_group.py" --out "$OUT" --models "$TAG" \
    --workers 4 --stagger 10 --base-urls "$URLS"
"$PY" - "$OUT" "$TAG" <<'PYEOF'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]) / sys.argv[2]
results = sorted(root.glob('craft_craft_crafting_table/ep*/result.json'))
assert len(results) == 3
for path in results:
    result = json.loads(path.read_text())
    assert 'err' not in result and result['steps'] >= 0
others = sorted(p for p in root.glob('*/ep*/result.json')
                if p.parts[-3] != 'craft_craft_crafting_table_prev_20261019')
assert len(others) == 12
print('VALID_CRAFT_RERUN_3_EPISODES_AND_12_TOTAL', flush=True)
PYEOF
echo "==== $TAG craft rerun completed $(date --iso-8601=seconds) ===="

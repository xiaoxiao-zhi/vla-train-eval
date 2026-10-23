#!/bin/bash
set -euo pipefail
ROOT=/apdcephfs/private_chengzhixu/evaluation
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
MODEL=$SHARE/checkpoint/duplex_weighted_then_noop_ce/eval_views/stage2_noop_unfiltered_ce_1936
TAG=flat3_noop_unfiltered_ce_1936
OUT=/apdcephfs/private_chengzhixu/JarvisVLA/outputs/rollout/flat3_noop_unfiltered_ce_1936_20261016-185457
PY=$SHARE/conda_envs/JarvisVLA/bin/python
CHECK_PY=/opt/conda/envs/torch-base/bin/python
URLS=""
mkdir -p "$OUT/_logs"
exec 9>"$OUT/rerun.lock"
flock -n 9 || { echo 'Stage2 rerun already running'; exit 1; }
exec >>"$OUT/_logs/rerun_$(date +%Y%m%d-%H%M%S).log" 2>&1
echo "==== stage2 rerun start $(date --iso-8601=seconds) ===="
for port in 8100 8102 8103 8104; do
    if curl --noproxy '*' -fsS -m 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -Fq "$MODEL"; then
        URLS="${URLS:+$URLS,}http://127.0.0.1:$port/v1"
    fi
done
if [ -z "$URLS" ]; then
    echo 'No reusable servers; launching four independent TP1/DP1 services on GPU0-3'
    "$CHECK_PY" "$ROOT/duplex/vllm_eval_pool.py" \
        --model "$MODEL" --model-name "$MODEL" \
        --gpus 0,1,2,3 --ports 8100,8102,8103,8104 \
        --logs "$OUT/_logs/rerun_servers" >"$OUT/_logs/rerun_servers_launch.log" 2>&1
    URLS=$("$PY" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))))' \
        "$OUT/_logs/rerun_servers/base_urls.json")
fi
echo "REUSING_URLS $URLS"
"$PY" - "$OUT/$TAG" <<'PYEOF'
import json
import shutil
import sys
from pathlib import Path
root = Path(sys.argv[1])
kept, removed = [], []
for path in sorted(root.glob('*/ep*')):
    result = path / 'result.json'
    if result.is_file():
        payload = json.loads(result.read_text())
        if 'err' not in payload and payload.get('steps', -1) >= 0:
            kept.append(str(path.relative_to(root)))
            continue
    shutil.rmtree(path)
    removed.append(str(path.relative_to(root)))
print('KEPT_VALID_EPISODES', kept)
print('REMOVED_INVALID_EPISODES', removed)
if len(kept) != 3 or set(kept) != {'craft_craft_crafting_table/ep0', 'kill_kill_zombie/ep0', 'smelt_iron_ingot/ep0'}:
    raise ValueError('Valid episodes differ from the known three: %s' % kept)
for name in removed:
    if 'smelt_iron_ingot/ep0' in name or 'kill_kill_zombie/ep0' in name or 'craft_craft_crafting_table/ep0' in name:
        raise ValueError('A previously valid episode was removed: %s' % name)
PYEOF
export DISPLAY=:99
if ! pgrep -f '^Xvfb :99 ' >/dev/null; then
    setsid nohup Xvfb :99 -screen 0 1280x720x24 </dev/null >/dev/null 2>&1 &
    sleep 2
fi
"$PY" "$ROOT/nonduplex/eval_4tasks_group.py" --out "$OUT" --models "$TAG" \
    --workers 4 --stagger 10 --base-urls "$URLS" >"$OUT/_logs/rerun_rollout.log" 2>&1
"$PY" - "$OUT/$TAG" <<'PYEOF'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = {}
for path in sorted(root.glob('*/ep*/result.json')):
    payload = json.loads(path.read_text())
    if 'err' in payload or payload.get('steps', -1) < 0:
        raise ValueError('Invalid episode after rerun: %s %s' % (path, payload))
    rows[str(path.parent.relative_to(root))] = payload
if len(rows) != 12:
    raise ValueError('Expected 12 valid episodes, found %d: %s' % (len(rows), sorted(rows)))
print('VALID_RESULT_12_EPISODES', json.dumps(rows, ensure_ascii=False))
PYEOF
echo "==== stage2 rerun completed $(date --iso-8601=seconds) ===="

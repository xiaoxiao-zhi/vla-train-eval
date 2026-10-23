#!/bin/bash
set -euo pipefail
ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
CODE=$ROOT/ms-swift/duplex_vla
OUT=$ROOT/JarvisVLA/outputs/qa/trivia_validation200
MODEL=$SHARE/checkpoint/s2v4_flat3/v0-20260930-225227/v0-20260930-225519/checkpoint-2900
PY=/opt/conda/envs/torch-base/bin/python
VLLM=/opt/conda/envs/torch-base/bin/vllm
mkdir -p "$OUT"
exec 9>"$OUT/pipeline.lock"
flock -n 9 || { echo 'TriviaQA evaluation already running'; exit 1; }
exec >>"$OUT/pipeline.log" 2>&1
echo "==== TriviaQA validation 200 start $(date) ===="
[ -f "$OUT/sample_audit.json" ] && [ -f "$OUT/samples.jsonl" ]
[ ! -e "$OUT/results.jsonl" ] && [ ! -e "$OUT/summary.json" ]
SERVER=''
cleanup() {
    if [ -n "$SERVER" ]; then
        kill -TERM -- "-$SERVER" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if ss -ltnH 'sport = :8101' | grep -q .; then
    curl --noproxy '*' -fsS -m 5 http://127.0.0.1:8101/v1/models | grep -Fq "$MODEL" || {
        echo 'Port 8101 has another model; refusing to replace it'
        exit 1
    }
    echo 'Reusing existing ordinary flat3 service'
else
    for gpu in 4; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
        [ "$used" -lt 2000 ] || { echo "GPU $gpu occupied: ${used}MiB"; exit 1; }
    done
    setsid env CUDA_VISIBLE_DEVICES=4 "$VLLM" serve "$MODEL" \
        --runner generate --tensor-parallel-size 1 --data-parallel-size 1 \
        --host 127.0.0.1 --port 8101 --max-model-len 8192 \
        --gpu-memory-utilization 0.85 --trust-remote-code \
        --limit-mm-per-prompt '{"image": 8}' >"$OUT/vllm.log" 2>&1 &
    SERVER=$!
    echo "Restored ordinary flat3 vLLM pid=$SERVER"
    READY=0
    for i in $(seq 1 240); do
        if curl --noproxy '*' -fsS -m 3 http://127.0.0.1:8101/v1/models 2>/dev/null | grep -Fq "$MODEL"; then
            READY=1
            break
        fi
        kill -0 "$SERVER" 2>/dev/null || { echo 'vLLM exited before readiness'; exit 1; }
        sleep 5
    done
    [ "$READY" -eq 1 ] || { echo 'vLLM readiness timeout'; exit 1; }
fi
"$PY" "$CODE/eval_general_qa_holdout.py" --mode evaluate --sources trivia_qa --n 200 \
    --out "$OUT" --base-url http://127.0.0.1:8101/v1 --model "$MODEL" >"$OUT/eval.log" 2>&1
"$PY" - "$OUT/summary.json" <<'PYEOF'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
assert summary['n'] == summary['trivia_qa']['n'] == 200
print('QA_COMPLETE', json.dumps(summary, ensure_ascii=False), flush=True)
PYEOF
echo "==== TriviaQA validation 200 completed $(date) ===="

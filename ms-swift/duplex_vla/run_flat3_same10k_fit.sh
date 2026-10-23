#!/bin/bash
set -euo pipefail
ROOT=/apdcephfs/private_chengzhixu
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
CODE=$ROOT/ms-swift/duplex_vla
MODEL=$SHARE/checkpoint/s2v4_flat3/v0-20260930-225227/v0-20260930-225519/checkpoint-2900
DATA=$SHARE/dataset/duplex_vla_v4/stage2_train_flat3_v3.jsonl
REFERENCE=$ROOT/mem_check/flat3_exp_v3_1936_vla10k.jsonl
OUT=$ROOT/JarvisVLA/outputs/fit/flat3_same10k
RESULT=$OUT/flat3_unweighted_vla10k.jsonl
PY=/opt/conda/envs/torch-base/bin/python
VLLM=/opt/conda/envs/torch-base/bin/vllm
mkdir -p "$OUT"
exec 9>"$OUT/pipeline.lock"
flock -n 9 || { echo 'Same-sample fit already running'; exit 1; }
exec >>"$OUT/pipeline.log" 2>&1
echo "==== ordinary flat3 same-10k fit start $(date) ===="
[ ! -e "$RESULT" ] || { echo "Refusing to overwrite $RESULT"; exit 1; }
[ ! -e "$OUT/comparison.json" ] || { echo 'Existing comparison will not be overwritten'; exit 1; }
[ -f "$OUT/sample_identity.json" ] || { echo 'Missing verified sample identity'; exit 1; }
for gpu in 4; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
    [ "$used" -lt 2000 ] || { echo "GPU $gpu occupied: ${used}MiB"; exit 1; }
done
if ss -ltnH 'sport = :8101' | grep -q .; then
    echo 'Port 8101 occupied; refusing to replace a service'
    exit 1
fi
SERVER=''
cleanup() {
    if [ -n "$SERVER" ]; then
        kill -TERM -- "-$SERVER" 2>/dev/null || true
    fi
}
trap cleanup EXIT
setsid env CUDA_VISIBLE_DEVICES=4 "$VLLM" serve "$MODEL" \
    --runner generate --tensor-parallel-size 1 --data-parallel-size 1 \
    --host 127.0.0.1 --port 8101 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 8}' >"$OUT/vllm.log" 2>&1 &
SERVER=$!
echo "Baseline vLLM pid=$SERVER"
READY=0
for i in $(seq 1 240); do
    if curl --noproxy '*' -fsS -m 3 http://127.0.0.1:8101/v1/models 2>/dev/null | grep -Fq "$MODEL"; then
        READY=1
        break
    fi
    kill -0 "$SERVER" 2>/dev/null || { echo 'Baseline vLLM exited before readiness'; exit 1; }
    sleep 5
done
[ "$READY" -eq 1 ] || { echo 'Baseline vLLM readiness timeout'; exit 1; }
"$PY" - "$CODE" "$DATA" "$REFERENCE" "$MODEL" "$OUT" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from memorization_check import do_req, load_samples
from summarize_vla_fit import parse
samples = load_samples(sys.argv[2], 10000, 42, True, sys.argv[3])
out = Path(sys.argv[5])
audit = json.loads((out / 'sample_identity.json').read_text())
fingerprint = hashlib.sha256(json.dumps(samples, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
if fingerprint != audit['sample_input_sha256']:
    raise ValueError('Sample input changed since identity verification')
record, response = do_req(samples[0], 'http://127.0.0.1:8101/v1', sys.argv[4],
                          temperature=0.9, top_p=0.99, top_k=-1)
parsed = parse(response)
if parsed is None or any(len(parsed[axis]) != 1 for axis in ('x', 'y')):
    raise ValueError('Baseline smoke request returned invalid camera action: %s' % response)
smoke = {'idx': record['idx'], 'gold': record['gold'], 'output': response,
         'images_n': len(record['images']), 'sample_input_sha256': fingerprint}
with open(out / 'smoke.json', 'x') as f:
    json.dump(smoke, f, ensure_ascii=False, indent=2)
print('SMOKE_OK', record['idx'], response, flush=True)
PYEOF
cd "$CODE"
"$PY" memorization_check.py --data "$DATA" --vla-only \
    --reference-results "$REFERENCE" \
    --base-url http://127.0.0.1:8101/v1 --model-name "$MODEL" \
    --out "$RESULT" --n 10000 --seed 42 --concurrency 256 \
    --temperature 0.9 --top-p 0.99 --top-k -1 >"$OUT/fit.log" 2>&1
"$PY" summarize_vla_fit.py --input "$RESULT" --output "$RESULT.metrics.json"
"$PY" compare_vla_fit.py --baseline "$RESULT" --candidate "$REFERENCE" \
    --expected-n 10000 --out "$OUT/comparison.json"
echo "==== ordinary flat3 same-10k fit completed $(date) ===="

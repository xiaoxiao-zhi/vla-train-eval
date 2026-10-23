#!/bin/bash
set -euo pipefail
ROOT=/apdcephfs/private_chengzhixu
CODE=$ROOT/ms-swift/duplex_vla
BASE=$ROOT/JarvisVLA/outputs/qa/trivia_validation200
OUT=$ROOT/JarvisVLA/outputs/qa/trivia_validation200_latest
MODEL=$ROOT/eval_model_views/flat3_exp_v3_1936
PY=/opt/conda/envs/torch-base/bin/python
VLLM=/opt/conda/envs/torch-base/bin/vllm
mkdir -p "$OUT"
exec 9>"$OUT/pipeline.lock"
flock -n 9 || { echo 'Latest TriviaQA evaluation already running'; exit 1; }
exec >>"$OUT/pipeline.log" 2>&1
echo "==== Latest checkpoint TriviaQA 200 start $(date) ===="
[ ! -e "$OUT/results.jsonl" ] && [ ! -e "$OUT/summary.json" ]
"$PY" - "$BASE" "$OUT" "$MODEL" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path
base, out, model = map(Path, sys.argv[1:])
audit = json.loads((base / 'sample_audit.json').read_text())
assert audit['n'] == 200 and set(audit['sources']) == {'trivia_qa'}
assert hashlib.sha256((base / 'samples.jsonl').read_bytes()).hexdigest() == audit['samples_sha256']
source = json.loads((model / 'eval_source.json').read_text())['source_checkpoint']
assert Path(source).name == 'checkpoint-1936'
assert all(path.resolve().parent == Path(source).resolve() for path in model.glob('model-*.safetensors'))
for name in ('samples.jsonl', 'sample_audit.json'):
    target = out / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Refusing to replace sample link: {target}')
    target.symlink_to(base / name)
print('SAME_200_OK; model source:', source, flush=True)
PYEOF
SERVER=''
cleanup() {
    if [ -n "$SERVER" ]; then
        kill -TERM -- "-$SERVER" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if ss -ltnH 'sport = :8100' | grep -q .; then
    curl --noproxy '*' -fsS -m 5 http://127.0.0.1:8100/v1/models | grep -Fq "$MODEL" || {
        echo 'Port 8100 has another model; refusing to replace it'
        exit 1
    }
    echo 'Reusing existing latest-model service'
else
    for gpu in 0; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
        [ "$used" -lt 2000 ] || { echo "GPU $gpu occupied: ${used}MiB"; exit 1; }
    done
    setsid env CUDA_VISIBLE_DEVICES=0 "$VLLM" serve "$MODEL" \
        --runner generate --tensor-parallel-size 1 --data-parallel-size 1 \
        --host 127.0.0.1 --port 8100 --max-model-len 8192 \
        --gpu-memory-utilization 0.85 --trust-remote-code \
        --limit-mm-per-prompt '{"image": 8}' >"$OUT/vllm.log" 2>&1 &
    SERVER=$!
    echo "Latest-model vLLM pid=$SERVER"
    READY=0
    for i in $(seq 1 240); do
        if curl --noproxy '*' -fsS -m 3 http://127.0.0.1:8100/v1/models 2>/dev/null | grep -Fq "$MODEL"; then
            READY=1
            break
        fi
        kill -0 "$SERVER" 2>/dev/null || { echo 'Latest-model vLLM exited before readiness'; exit 1; }
        sleep 5
    done
    [ "$READY" -eq 1 ] || { echo 'Latest-model vLLM readiness timeout'; exit 1; }
fi
"$PY" "$CODE/eval_general_qa_holdout.py" --mode evaluate --sources trivia_qa --n 200 \
    --out "$OUT" --base-url http://127.0.0.1:8100/v1 --model "$MODEL" >"$OUT/eval.log" 2>&1
echo "Latest-model QA completed $(date)"
cleanup
SERVER=''
for i in $(seq 1 240); do
    [ ! -f "$BASE/summary.json" ] || break
    sleep 5
done
[ -f "$BASE/summary.json" ] || { echo 'Baseline QA summary not ready; latest-model results are saved'; exit 1; }
"$PY" - "$BASE" "$OUT" <<'PYEOF'
import json
import sys
from pathlib import Path
base, out = map(Path, sys.argv[1:])
a = json.loads((base / 'summary.json').read_text())
b = json.loads((out / 'summary.json').read_text())
assert a['n'] == b['n'] == 200
assert a['samples_sha256'] == b['samples_sha256']
assert a['generation'] == b['generation']
assert a['model'] != b['model']
records = []
for folder in (base, out):
    rows = [json.loads(line) for line in (folder / 'results.jsonl').read_text().splitlines()]
    indexed = {row['source_id']: row for row in rows}
    assert len(rows) == len(indexed) == 200
    records.append(indexed)
original, latest = records
assert original.keys() == latest.keys()
fields = ('question', 'gold', 'aliases', 'messages', 'source_split')
assert all(all(original[key][field] == latest[key][field] for field in fields) for key in original)
metrics = ('exact_match', 'answer_contains', 'token_f1')
comparison = {'n': 200, 'samples_sha256': a['samples_sha256'], 'same_inputs_gold_generation': True,
              'baseline_model': a['model'], 'latest_model': b['model'],
              'baseline': a['overall'], 'latest': b['overall'],
              'latest_minus_baseline': {metric: b['overall'][metric] - a['overall'][metric] for metric in metrics},
              'paired': {}}
for metric in ('exact_match', 'answer_contains'):
    comparison['paired'][metric] = {
        'both_correct_n': sum(original[key][metric] and latest[key][metric] for key in original),
        'baseline_only_correct_n': sum(original[key][metric] and not latest[key][metric] for key in original),
        'latest_only_correct_n': sum(latest[key][metric] and not original[key][metric] for key in original),
        'both_wrong_n': sum(not original[key][metric] and not latest[key][metric] for key in original),
    }
with open(out / 'comparison.json', 'x') as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)
print('QA_COMPARISON_COMPLETE', json.dumps(comparison, ensure_ascii=False), flush=True)
PYEOF
echo "==== Latest checkpoint TriviaQA 200 pipeline completed $(date) ===="

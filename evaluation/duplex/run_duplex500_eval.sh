#!/bin/bash
set -euo pipefail
ROOT=/apdcephfs/private_chengzhixu/JarvisVLA
CODE=/apdcephfs/private_chengzhixu/evaluation/duplex
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
OUT=${DUPLEX_EVAL_OUT:-$ROOT/outputs/rollout/duplex500_ce_ckpt243}
MODEL=${DUPLEX_EVAL_MODEL:-$SHARE/checkpoint/duplex_v3_500traj_ce/eval_views/checkpoint-243}
TAG=${DUPLEX_EVAL_TAG:-duplex500_ce_ckpt243}
TRAIN_RUN=${DUPLEX_TRAIN_RUN:-$SHARE/checkpoint/duplex_v3_500traj_ce/training/run-20261015-164152-TKsPJA}
QA=$ROOT/outputs/qa/trivia_validation200/samples.jsonl
DATA=$SHARE/dataset/flat3_ablation/duplex_train_v3_500traj_filtered.jsonl
PY=$SHARE/conda_envs/JarvisVLA/bin/python
CHECK_PY=/opt/conda/envs/torch-base/bin/python
BASE_URLS=${1:-${EVAL_BASE_URLS:-}}
[ -n "$BASE_URLS" ] || { echo 'Usage: run_duplex500_eval.sh <comma-separated vLLM base URLs>'; exit 1; }
BASE_URL=${BASE_URLS%%,*}
mkdir -p "$OUT/_logs"
exec 9>"$OUT/pipeline.lock"
flock -n 9 || { echo 'Duplex500 evaluation already running'; exit 1; }
exec >>"$OUT/_logs/pipeline.log" 2>&1
echo "==== $TAG pipeline start $(date) ===="
[ ! -e "$OUT/summary.json" ] || { echo 'Existing evaluation will not be overwritten'; exit 1; }
CHECKPOINT=$($CHECK_PY "$CODE/prepare_duplex500_eval.py" --run "$TRAIN_RUN")
if [ ! -d "$MODEL" ]; then
    "$CHECK_PY" "$CODE/prepare_duplex500_eval.py" --run "$TRAIN_RUN" --out "$MODEL"
fi
"$CHECK_PY" - "$MODEL" "$CHECKPOINT" "$QA" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path
model, checkpoint, qa = map(Path, sys.argv[1:])
source = json.loads((model / 'eval_source.json').read_text())
assert Path(source['source_checkpoint']).resolve() == checkpoint.resolve()
assert len(list(model.glob('model-*.safetensors'))) == 4
assert all(path.resolve().parent == checkpoint.resolve() for path in model.glob('model-*.safetensors'))
audit = json.loads((qa.parent / 'sample_audit.json').read_text())
assert audit['n'] == 200 and audit['selected_question_or_prompt_overlap_n'] == 0
assert hashlib.sha256(qa.read_bytes()).hexdigest() == audit['samples_sha256']
print('SAVED_CHECKPOINT_AND_QA_VERIFIED', checkpoint, flush=True)
PYEOF
"$CHECK_PY" - "$MODEL" <<'PYEOF'
import sys
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoProcessor
from vllm.transformers_utils.tokenizer import get_tokenizer
model = Path(sys.argv[1])
tokenizer = get_tokenizer(str(model), trust_remote_code=True)
assert tokenizer.get_vocab() == Tokenizer.from_file(str(model / 'tokenizer.json')).get_vocab()
for token, idx in (('<|silence|>', 151908), ('<|obs_user|>', 151909), ('<|obs_asst|>', 151910)):
    assert tokenizer(token, add_special_tokens=False)['input_ids'] == [idx]
processor = AutoProcessor.from_pretrained(model)
print('VLLM_ENV_TOKENIZER_AND_PROCESSOR_OK', type(processor).__name__, flush=True)
PYEOF
"$PY" - "$ROOT" "$DATA" "$MODEL" "$TAG" "$OUT" "$BASE_URL" <<'PYEOF'
import json
import sys
from pathlib import Path
import requests
from PIL import Image
from transformers import AutoTokenizer
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, '/apdcephfs/private_chengzhixu/evaluation')
from duplex.duplex_interact_v4 import ACT_RE, vllm_chat
from jarvisvla.inference.action_mapping import OneActionTokenizer
with open(sys.argv[2]) as file:
    row = json.loads(next(file))
images = [Image.open(path).convert('RGB') for path in row['images']]
with requests.Session() as session:
    session.trust_env = False
    prediction, request = vllm_chat(sys.argv[6], sys.argv[4],
                                    row['messages'][0]['content'], images, session)
match = ACT_RE.search(prediction)
if match is None:
    raise ValueError('Duplex smoke did not return a valid action: %r' % prediction)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[3])
decoded = OneActionTokenizer(tokenizer_type='qwen2_vl').decode(
    tokenizer(match.group(1), add_special_tokens=False)['input_ids'])
if not decoded:
    raise ValueError('Duplex smoke action decoding failed')
reconstructed = ''.join(item['text'] if item['type'] == 'text' else '<image>'
                        for item in request['messages'][0]['content'])
assert reconstructed == row['messages'][0]['content']
with (Path(sys.argv[5]) / 'smoke.json').open('x') as file:
    json.dump({'base': row['base'], 't': row['t'], 'images_n': len(images),
               'output': prediction, 'action_decoded': True, 'prompt_text_preserved': True}, file,
              ensure_ascii=False, indent=2)
print('REAL_DUPLEX_REQUEST_SMOKE_OK', prediction, flush=True)
PYEOF
export DISPLAY=:99
if ! pgrep -f '^Xvfb :99 ' >/dev/null; then
    setsid nohup Xvfb :99 -screen 0 1280x720x24 </dev/null >/dev/null 2>&1 &
    sleep 2
fi
"$PY" "$CODE/duplex_interact_v4.py" \
    --base-urls "$BASE_URLS" --model-name "$TAG" --tokenizer "$MODEL" \
    --hist-mode text --task all --workers 3 --stagger 10 --repeats 3 --seed 42 --qa-pool "$QA" \
    --out-dir "$OUT" >"$OUT/_logs/rollout.log" 2>&1
"$CHECK_PY" - "$OUT" <<'PYEOF'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
summary = json.loads((out / 'summary.json').read_text())
assert summary['total'] == 9
for task, data in summary['tasks'].items():
    assert data['total'] == 3
    for episode in range(3):
        directory = out / task.replace('/', '_') / ('ep%d' % episode)
        result = json.loads((directory / 'result.json').read_text())
        environment = json.loads((directory / 'environment.json').read_text())
        assert 'err' not in result and result['steps'] >= 0
        assert result['seed'] == environment['python_seed'] == 42 + episode
print('VALID_DUPLEX_RESULT_9_EPISODES', json.dumps(summary, ensure_ascii=False), flush=True)
PYEOF
echo "==== $TAG pipeline completed $(date) ===="

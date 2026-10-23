#!/bin/bash
set -euo pipefail
ROOT=/apdcephfs/private_chengzhixu/JarvisVLA
CODE=/apdcephfs/private_chengzhixu/evaluation/nonduplex
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
MODEL=$SHARE/checkpoint/v5_ce_camera_exponential/eval_views/duplex_camera_exp_500
TAG=flat3_v5_duplex_camera_exp_500
TRAIN_RUN=$SHARE/checkpoint/v5_ce_camera_exponential/run-20261016-224410-XIBAET/duplex_camera_exponential
DATA=$SHARE/dataset/flat3_ablation/stage2_train_flat3_vla300k_qav3.jsonl
OUT=${V5_EVAL_OUT:-$ROOT/outputs/rollout/${TAG}_$(date +%Y%m%d-%H%M%S)}
PY=$SHARE/conda_envs/JarvisVLA/bin/python
CHECK_PY=/opt/conda/envs/torch-base/bin/python
mkdir -p "$OUT/_logs"
exec 9>"$OUT/eval.lock"
flock -n 9 || { echo 'Evaluation already running'; exit 1; }
exec >>"$OUT/_logs/pipeline.log" 2>&1
echo "==== $TAG start $(date --iso-8601=seconds) ===="
if [ ! -f "$MODEL/eval_source.json" ]; then
    "$CHECK_PY" /apdcephfs/private_chengzhixu/evaluation/duplex/prepare_v5_camera_exp_eval.py \
        --run "$TRAIN_RUN" --out "$MODEL"
fi
"$CHECK_PY" - "$MODEL" <<'PYEOF'
import json
import sys
from pathlib import Path
model = Path(sys.argv[1])
source = json.loads((model / 'eval_source.json').read_text())
checkpoint = Path(source['source_checkpoint'])
assert checkpoint.name == 'checkpoint-500'
assert source['global_step'] == 500
assert len(list(model.glob('model-*.safetensors'))) == 4
assert all(path.resolve().parent == checkpoint.resolve() for path in model.glob('model-*.safetensors'))
for token, idx in (('<|silence|>', 151908), ('<|obs_user|>', 151909), ('<|obs_asst|>', 151910),
                   ('<|reserved_special_token_178|>', 151835), ('<|reserved_special_token_250|>', 151907)):
    assert source['required_token_ids'].get(token) == idx
print('SAVED_CHECKPOINT_VIEW_VERIFIED', checkpoint, flush=True)
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
/opt/conda/envs/torch-base/bin/python "/apdcephfs/private_chengzhixu/evaluation/duplex/vllm_eval_pool.py" \
    --model "$MODEL" --model-name "$MODEL" --gpus 0,1,2,3 --ports 8100,8102,8103,8104 \
    --logs "$OUT/_logs" >"$OUT/_logs/servers_launch.log" 2>&1
URLS=$("$PY" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))))' "$OUT/_logs/base_urls.json")
"$PY" - "$CODE" "$DATA" "$OUT/_logs/base_urls.json" "$MODEL" "$OUT" <<'PYEOF'
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, '/apdcephfs/private_chengzhixu/evaluation')
from nonduplex.run_eval_task import FLAT3_AGENT
with open(sys.argv[2]) as file:
    record = next(row for row in map(json.loads, file) if len(row.get('images', [])) == 3)
urls = json.loads(Path(sys.argv[3]).read_text())
frames = [np.asarray(Image.open(path).convert('RGB')) for path in record['images']]
results = []
for url in urls:
    agent = FLAT3_AGENT(checkpoint_path=sys.argv[4], base_url=url, temperature=0.9,
                       history_num=2, instruction_type='normal', action_chunk_len=1)
    agent._flat_instruction = record['messages'][0]['content'].split('\nobservation:')[0]
    agent._flat_hist = [agent.processor_wrapper.create_image_input(frame) for frame in frames[:-1]]
    action = agent.forward([frames[-1]], ['mine_block:stone'])
    content = agent.log_inputs[-1]['messages'][0]['content']
    assert sum(item['type'] == 'image_url' for item in content) == 3
    assert action
    results.append({'url': url, 'output': agent.log_outputs[-1]['output'], 'action_decoded': True})
with (Path(sys.argv[5]) / 'smoke.json').open('x') as file:
    json.dump(results, file, ensure_ascii=False, indent=2)
print('ALL_SERVERS_REAL_FLAT3_REQUEST_OK', urls, flush=True)
PYEOF
export DISPLAY=:99 EVAL_BASE_SEED=42
if ! pgrep -f '^Xvfb :99 ' >/dev/null; then
    setsid nohup Xvfb :99 -screen 0 1280x720x24 </dev/null >/dev/null 2>&1 &
    sleep 2
fi
"$PY" "$CODE/eval_4tasks_group.py" --out "$OUT" --models "$TAG" \
    --workers 4 --stagger 10 --base-urls "$URLS" >"$OUT/_logs/rollout.log" 2>&1
"$PY" - "$OUT" "$TAG" <<'PYEOF'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]) / sys.argv[2]
results = list(root.glob('*/ep*/result.json'))
assert len(results) == 12
for path in results:
    result = json.loads(path.read_text())
    assert 'err' not in result and result['steps'] >= 0
print('VALID_RESULT_12_EPISODES', flush=True)
PYEOF
echo "==== $TAG completed $(date --iso-8601=seconds) ===="

#!/bin/bash
set -euo pipefail
ROOT=/apdcephfs/private_chengzhixu/JarvisVLA
CODE=/apdcephfs/private_chengzhixu/evaluation/duplex
SHARE=/apdcephfs/share_chengzhixu/cephfs/apdcephfs_tj5/share_303556863/chengzhixu
export DUPLEX_EVAL_MODEL=$SHARE/checkpoint/duplex_weighted_then_noop_ce/eval_views/duplex_camera_exponential_243
export DUPLEX_EVAL_TAG=duplex500_camera_exponential_243
export DUPLEX_TRAIN_RUN=$SHARE/checkpoint/duplex_weighted_then_noop_ce/run-20261015-223946-56SfT5/duplex_camera_exponential
export DUPLEX_EVAL_OUT=${DUPLEX_EVAL_OUT:-$ROOT/outputs/rollout/${DUPLEX_EVAL_TAG}_$(date +%Y%m%d-%H%M%S)}
mkdir -p "$DUPLEX_EVAL_OUT/_logs"
exec 8>"$DUPLEX_EVAL_OUT/start.lock"
flock -n 8 || { echo 'Weighted duplex evaluation already starting'; exit 1; }
exec >>"$DUPLEX_EVAL_OUT/_logs/start.log" 2>&1
echo "==== weighted duplex start $(date --iso-8601=seconds) ===="
/opt/conda/envs/torch-base/bin/python "$CODE/vllm_eval_pool.py" \
    --model "$DUPLEX_EVAL_MODEL" --model-name "$DUPLEX_EVAL_TAG" \
    --gpus 4,5,6 --ports 8110,8112,8113 --logs "$DUPLEX_EVAL_OUT/_logs" \
    >"$DUPLEX_EVAL_OUT/_logs/servers_launch.log" 2>&1
URLS=$(/opt/conda/envs/torch-base/bin/python -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))))' \
    "$DUPLEX_EVAL_OUT/_logs/base_urls.json")
bash "$CODE/run_duplex500_eval.sh" "$URLS"
echo "==== weighted duplex completed $(date --iso-8601=seconds) ===="

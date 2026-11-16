#!/bin/bash
# Patch vLLM-Omni, write the deploy config and serve Qwen3-TTS 12Hz 1.7B Base on 127.0.0.1:8091.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8091}
CONFIG=${DEPLOY_CONFIG:-/tmp/qwen3-tts-deploy.yaml}

# Frames per streamed chunk: 1 frame first, then growing, so buffered audio stays ahead of playback (0001).
export VLLM_OMNI_STREAM_CHUNK_RAMP=${VLLM_OMNI_STREAM_CHUNK_RAMP:-1,2,4,8,16,25}
# FP8 code predictor on sm_89 (0004). Set 0 on other GPUs or to compare against BF16.
export VLLM_OMNI_CP_FP8=${VLLM_OMNI_CP_FP8:-1}
# Primed codec states cached per voice reference (0005).
export VLLM_OMNI_STREAM_PRIME_CACHE=${VLLM_OMNI_STREAM_PRIME_CACHE:-8}
# FP8 weight-only talker GEMMs up to this batch size (0006). 0 = off; it was neutral on an RTX 4000 Ada.
export VLLM_FP8_W8A16_MAX_M=${VLLM_FP8_W8A16_MAX_M:-0}
# Keep Triton autotune results between restarts: graph capture and autotuning take ~11 min cold.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.cache/triton}

bash "$HERE/apply-patches.sh"
python3 "$HERE/deploy_config.py" "$CONFIG"
exec vllm serve "$MODEL" --omni --deploy-config "$CONFIG" --trust-remote-code \
  --stage-init-timeout "${STAGE_INIT_TIMEOUT:-1800}" --host "$HOST" --port "$PORT"

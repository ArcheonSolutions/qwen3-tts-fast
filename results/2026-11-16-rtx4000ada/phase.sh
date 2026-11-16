#!/bin/bash
# phase.sh NAME MODE  (MODE: stock | config | patched)
set -uo pipefail
NAME=$1; MODE=$2
R=/root/results/$NAME; mkdir -p $R
MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3
B=/opt/bench/bin/python
export TRITON_CACHE_DIR=/root/.cache/triton
export LD_LIBRARY_PATH=$(ls -d /opt/bench/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
pkill -f "vllm serve"
for i in $(seq 1 60); do pgrep -f "vllm serve" > /dev/null || pgrep -f "VLLM" > /dev/null || break; sleep 2; done
pkill -9 -f "vllm serve"; sleep 3
while curl -s -o /dev/null http://127.0.0.1:8091/health; do sleep 2; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 2; done
START=$(date +%s)
case $MODE in
  stock)   vllm serve $MODEL --omni --trust-remote-code --stage-init-timeout 1800 --host 127.0.0.1 --port 8091 > $R/server.log 2>&1 & ;;
  firstdeploy) python3 - /tmp/first.yaml <<'PY'
import pathlib, sys, yaml, vllm_omni
config = yaml.safe_load((pathlib.Path(vllm_omni.__file__).parent/'deploy/qwen3_tts.yaml').read_text())
config['connectors']['connector_of_shared_memory']['extra']['codec_chunk_ramp'] = [2, 4, 8, 16, 25]
pathlib.Path(sys.argv[1]).write_text(yaml.safe_dump(config, sort_keys=False))
PY
           cp /tmp/first.yaml $R/
           vllm serve $MODEL --omni --deploy-config /tmp/first.yaml --trust-remote-code --stage-init-timeout 1800 --host 127.0.0.1 --port 8091 > $R/server.log 2>&1 & ;;
  ramponly) export VLLM_OMNI_STREAM_CHUNK_RAMP=1,2,4,8,16,25
           python3 /opt/qwen3-tts-fast/deploy_config.py /tmp/config.yaml && cp /tmp/config.yaml $R/
           vllm serve $MODEL --omni --deploy-config /tmp/config.yaml --trust-remote-code --stage-init-timeout 1800 --host 127.0.0.1 --port 8091 > $R/server.log 2>&1 & ;;
  config)  python3 /opt/qwen3-tts-fast/deploy_config.py /tmp/config.yaml && cp /tmp/config.yaml $R/
           vllm serve $MODEL --omni --deploy-config /tmp/config.yaml --trust-remote-code --stage-init-timeout 1800 --host 127.0.0.1 --port 8091 > $R/server.log 2>&1 & ;;
  patched) MODEL=$MODEL bash /opt/qwen3-tts-fast/serve.sh > $R/server.log 2>&1 & ;;
esac
SERVER=$!
until curl -sf http://127.0.0.1:8091/health > /dev/null; do
  sleep 10
  if ! kill -0 $SERVER 2>/dev/null; then echo "server died"; tail -40 $R/server.log; exit 1; fi
  if [ $(( $(date +%s) - START )) -gt 3000 ]; then echo timeout; exit 1; fi
done
echo "{\"ready_seconds\": $(( $(date +%s) - START ))}" | tee $R/startup.json
$B /opt/qwen3-tts-fast/register_voice.py /root/voice/ref/reference.wav "$(python3 -c 'import json;print(" ".join(r["text"] for r in json.load(open("/root/voice/ref/reference-transcript.json"))))')" --name default | tee $R/register.txt
nvidia-smi --query-gpu=memory.used --format=csv > $R/gpu_mem.txt
shift 2
for cmd in "$@"; do
  echo "== $cmd" | tee -a $R/bench.jsonl.log
  eval "$B $cmd" 2>&1 | tee -a $R/bench.jsonl.log
done
nvidia-smi --query-gpu=memory.used --format=csv >> $R/gpu_mem.txt
echo PHASE_DONE

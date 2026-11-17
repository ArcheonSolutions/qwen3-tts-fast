#!/bin/bash
# phase.sh NAME BRANCH OVERRIDES_JSON CMD...   (one fresh server per phase)
set -uo pipefail
NAME=$1; BRANCH=$2; OVR=$3; shift 3
R=/root/results/$NAME; mkdir -p $R
export PATH=/opt/omni/bin:$PATH TRITON_CACHE_DIR=/root/.cache/triton
MODEL=$(cat /root/model_path)
pkill -f "vllm serve"; sleep 5; pkill -9 -f "vllm serve"; sleep 3
while curl -s -o /dev/null http://127.0.0.1:8091/health; do sleep 2; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 2; done
cd /root/vllm-omni && git checkout -q $BRANCH && git log -1 --format='%H %s' > $R/commit.txt
python - "$OVR" $R/deploy.yaml <<'PY'
import json, pathlib, sys, yaml
cfg = yaml.safe_load(pathlib.Path('vllm_omni/deploy/qwen3_tts_fused_single_gpu.yaml').read_text())
cfg['stages'][0]['additional_config'].update(json.loads(sys.argv[1]))
pathlib.Path(sys.argv[2]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
START=$(date +%s)
vllm serve $MODEL --omni --deploy-config $R/deploy.yaml --trust-remote-code --stage-init-timeout 1800 --host 127.0.0.1 --port 8091 > $R/server.log 2>&1 &
SERVER=$!
until curl -sf http://127.0.0.1:8091/health > /dev/null; do
  sleep 10
  if ! kill -0 $SERVER 2>/dev/null; then echo "server died"; tail -60 $R/server.log; exit 1; fi
  if [ $(( $(date +%s) - START )) -gt 3000 ]; then echo timeout; exit 1; fi
done
echo "{\"ready_seconds\": $(( $(date +%s) - START ))}" | tee $R/startup.json
/opt/bench/bin/python /root/kit/register_voice.py /root/voice/clone_2.wav "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you." --name default | tee $R/register.txt
nvidia-smi --query-gpu=name,memory.used --format=csv > $R/gpu_mem.txt
for cmd in "$@"; do
  echo "== $cmd" | tee -a $R/bench.jsonl.log
  (cd /root/kit && eval "/opt/bench/bin/python $cmd") 2>&1 | tee -a $R/bench.jsonl.log
done
nvidia-smi --query-gpu=name,memory.used --format=csv >> $R/gpu_mem.txt
echo PHASE_DONE

# Runs on 2026-11-16, RTX 4000 Ada, vllm/vllm-omni:v0.32.0

Each folder is one fresh server started by `phase.sh <name> <mode>`, then the benchmark commands listed in its
`bench.jsonl.log`. `startup.json` is the time until `/health` answered, `gpu_mem.txt` the memory used before and
after the benchmarks.

| Folder | Server |
| --- | --- |
| `stock` | Stock vllm-omni, default `qwen3_tts.yaml` (two stages), no patches |
| `firstdeploy`, `firstdeploy2`, `firstdeploy_quality` | `qwen3_tts.yaml` + `codec_chunk_ramp: [2, 4, 8, 16, 25]` (`first.yaml`), no patches |
| `config` | `deploy_config.py`, no patches |
| `ramponly` | `deploy_config.py` + patch 0001 only |
| `patched`, `patched_more`, `patched_curl` | `serve.sh`: `deploy_config.py` + all patches |
| `patched_cpbf16` | Same with `VLLM_OMNI_CP_FP8=0` (code predictor in BF16) |

Voice: a 6.9 s French reference clip (not included). Synthesized WAVs and server logs are not included.

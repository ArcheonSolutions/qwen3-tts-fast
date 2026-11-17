# vllm-omni main + upstream pull requests, NVIDIA L4, 2026-11-17

Validation of the four upstream pull requests (ports of patches 0001–0005 and 0007 to vllm-omni `main`).

- GPU: NVIDIA L4 24 GB (sm_89), RunPod Secure Cloud, driver 595.91. The RTX 4000 Ada used for the
  0.32.0 results was out of stock; the L4 is the same architecture and upstream CI's GPU for these tests.
- Software: vLLM 0.33.0 (upstream's pinned wheel, commit 4ba787ec), torch 2.14.0+cu130,
  vllm-omni `main` at 174d2216c. Setup: [`kit/setup.sh`](kit/setup.sh).
- Model: Qwen3-TTS-12Hz-1.7B-Base. Reference voice: Qwen's public
  `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav` with its example transcript.
  Synthesized text is French (the bench sentences).
- Deploy config: upstream's `vllm_omni/deploy/qwen3_tts_fused_single_gpu.yaml` from each branch, unchanged,
  except `code_predictor_fp8: true` for the last row. Runner: [`kit/phase.sh`](kit/phase.sh),
  [`kit/run2.sh`](kit/run2.sh). One fresh server per row.

Each row adds one pull request to the previous row (`stackN` = main + PR 1..N merged):

| Folder | Branch | Calls with 0% gapped turns (60 s per level) |
| --- | --- | --- |
| `results/main` | main | none (100% gapped at 8, 16, 32) |
| `results/pr1` | + chunk ramp | 32 (50: 2.3%) |
| `results/pr2` | + voice-clone caches | 75 (100: 1.0%) |
| `results/pr3` | + codec GEMMs | 100 (150: 39%) |
| `results/pr4` | + FP8 code predictor | 100 (150: 32%), first audio p50 85 ms vs 101 ms |

Microbenchmarks (`results/micro`, scripts `kit/micro_codec.py`, `kit/micro_cp.py`): streaming codec step
time main vs the codec PR, and fused code predictor time per frame BF16 vs FP8 weights.

`results_run1/` is a first run in which `bench_tts.py` and `bench_calls.py` crashed on streams the server
ends early (`main` now aborts a stream that exceeds the codec token budget, 1–8 clauses per level here);
the scripts now count those as `failed`. Its quality runs (48 clips, Whisper CER and ECAPA similarity)
are the ones reported: main CER 2.0% mean, 1 bad clip, similarity 0.653; all four PRs with FP8 CER 3.2%,
2 bad clips, similarity 0.635. 48 clips per side cannot separate these.

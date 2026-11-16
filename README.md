# qwen3-tts-fast

Patches and a serving config for [vLLM-Omni](https://github.com/vllm-project/vllm-omni) 0.32.0 that let one
**NVIDIA RTX 4000 Ada (20 GB, $0.28/h)** stream
[Qwen3-TTS 12Hz 1.7B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) with voice cloning to
**300 simultaneous voice-agent calls without a playback gap**, up from 16 with the stock serving stack.
Same model, same voice, same output quality.

| Setup on one RTX 4000 Ada | Calls with 0% gapped turns | Back-to-back streams, audio produced per second |
| --- | --- | --- |
| Stock vLLM-Omni 0.32.0 (`qwen3_tts.yaml`) | none (every turn gaps, even at 4 calls) | 28 s/s |
| Stock + codec chunk ramp in the deploy config | **16** | 26 s/s |
| Serving config below + streaming fix (patch 0001) | **75** | 95 s/s |
| + kernel patches 0002–0007 | **300** | **161 s/s** |

All numbers: measured on 2026-11-16, TTS alone on the GPU, same cloned voice and sentences for every row.
Raw logs are in [`results/2026-11-16-rtx4000ada/`](results/2026-11-16-rtx4000ada/).

## What changed

### Serving config (`deploy_config.py`)

- **Single-stage fused pipeline** (`qwen3_tts_fused_single_gpu.yaml`): the talker decodes codec frames itself,
  no second stage.
- **Triton attention + full CUDA graphs.** In a voice agent every new clause is a new request, so most engine
  steps mix one prefill with the running decodes. FlashAttention 2 cannot CUDA-graph those mixed steps, so they
  ran eagerly at about 3x the cost of a decode step. `TRITON_ATTN` + `cudagraph_mode: FULL` graphs them.
- **FP8 talker linears** (`quantization: fp8`).
- `max_num_seqs` 96, `gpu_memory_utilization` 0.4 (~13 GB in use, leaving room for a speech recognition model
  on the same GPU).

### Patches (`patches/`, against vllm-omni 0.32.0 and vllm 0.32.0)

| Patch | What it does |
| --- | --- |
| `0001-stream-chunk-ramp` | The fused pipeline sends 1 frame, then waits for 25 frames (2 s of audio) before the next chunk, so playback stalls. Chunks now follow `VLLM_OMNI_STREAM_CHUNK_RAMP` (default `1,2,4,8,16,25` frames). |
| `0002-codec-transconv-as-2tap-conv` | Each transposed conv in the streaming codec decoder becomes a 2-tap conv: one im2col and one GEMM, no col2im kernel. |
| `0003-codec-igemm-all-units-autotune` | Every residual unit uses the implicit-GEMM dilated conv, with launch configs picked by `triton.autotune` (cached on disk). |
| `0004-code-predictor-fp8-chain-sm89` | The code predictor's 15 sequential steps per audio frame use FP8 (e4m3) weights with per-row scales on sm_89 (`VLLM_OMNI_CP_FP8=1`). |
| `0005-stream-prime-state-cache` | Every request for the same voice re-decoded the 72 reference frames through the codec to prime its decoder state, about two thirds of the codec's work per step. The primed state is now cached per reference and restored with one copy (`VLLM_OMNI_STREAM_PRIME_CACHE`, default 8 entries). |
| `0006-talker-fp8-w8a16-small-m` | Optional FP8 weight-only (W8A16) Triton GEMM for the talker at small batch (`VLLM_FP8_W8A16_MAX_M`, default `0` = off; neutral on an RTX 4000 Ada). |
| `0007-icl-ref-codec-embed-cache` | Caches the reference codec embeddings per voice: 16 fewer kernel launches per request. |

The mechanism figures above (3x eager cost, two thirds of the codec's work, 16 launches) come from profiling
during development. The results below were re-measured end to end.

All seven apply cleanly with `patch -p1` (no fuzz) to pristine vllm-omni 0.32.0 and vllm 0.32.0, inside the
`vllm/vllm-omni:v0.32.0` image and in `docker build`.

## Results

RTX 4000 Ada 20 GB (sm_89), RunPod Secure Cloud at $0.28/h, driver 595.91 (CUDA 13.2), 2026-11-16.
A *gap* is playback waiting more than 100 ms for audio. The runner for every row is
[`results/.../phase.sh`](results/2026-11-16-rtx4000ada/phase.sh); each row is one fresh server.

### Simulated calls (`bench/bench_calls.py`)

Each call speaks turns of 2–3 clauses in real time, requesting the next clause while the previous one plays,
then listens for 3–7 s before its next turn. 90 s per level. Cells: % of turns with a gap, then first audio of
a turn p50 / p95.

| Calls | Stock + chunk ramp | Config + 0001 | All patches |
| --- | --- | --- | --- |
| 4 | 0% · 0.12 / 0.19 s | | |
| 16 | 0% · 0.17 / 0.23 s | | |
| 20 | 6.6% | | |
| 32 | 45% | | |
| 50 | 90% | 0% · 0.08 / 0.11 s | |
| 75 | | 0% · 0.09 / 0.14 s | 0% · 0.045 / 0.058 s |
| 100 | | 0.4% | 0% · 0.047 / 0.061 s |
| 125 | | 25% | 0% · 0.049 / 0.066 s |
| 150 | | 62% | 0% · 0.053 / 0.074 s |
| 200 | | | 0% · 0.061 / 0.086 s |
| 250 | | | 0% · 0.076 / 0.112 s |
| 300 | | | 0% · 0.107 / 0.165 s |
| 400 | | | 0% gapped, but first audio 1.8 / 2.2 s (requests queue behind `max_num_seqs` 96) |

Fully stock (no chunk ramp): 100% of turns gapped at 4, 8, 16 and 32 calls. The serving config without 0001
also gaps 100% of turns at every level from 50 to 150, despite nearly 3x the throughput: the 2 s chunk stall hides it.

Turns twice as long (6–10 s of speech, `--turn-scale 2`): config + 0001 gapped 40% of turns at 100 calls and
87% at 125. All patches: 0% at 125 and 150.

### Back-to-back streams (`bench/bench_tts.py`, 0.2 s client buffer)

N workers each stream clauses with no pause, the worst case for TTS. Cells: % of streams with a gap.

| Streams | Stock | Stock + chunk ramp | Config + 0001 | All patches |
| --- | --- | --- | --- | --- |
| 1 | 100% | 0% | 0% | 0% |
| 8 | 100% | 0% | | |
| 12 | 100% | 5.6% | | |
| 16 | 100% | 33% | 0% | 0% |
| 24 | 100% | 71% | 0% | 0% |
| 32 | | | 4.2% | 0% |
| 64 | | | 45% | 0% |
| Audio per wall second, at the highest level | 28 s/s (24) | 26 s/s (24) | 95 s/s (64) | 161 s/s (64) |
| First audio, 1 stream | 0.099 s | 0.117 s | 0.061 s | 0.032 s |

### Patch 0004 on its own

All patches with the code predictor in BF16 (`VLLM_OMNI_CP_FP8=0`): 142 s/s at 64 streams (vs 160 with FP8),
300 calls still 0% gapped, first audio p50 0.129 s (vs 0.107 s). The other patches weren't measured one by one.

### Cost

At 300 calls per $0.28/h GPU, TTS costs about **$0.001 per call-hour**. At the stock + chunk ramp limit of 16,
it's $0.0175, and 300 calls would need 19 GPUs.

### With speech recognition on the same GPU

These numbers are TTS alone. A voice agent also transcribes the caller. In our production setup, Parakeet TDT
0.6B v3 (ONNX Runtime CUDA, batched) runs on the same GPU next to vLLM, through CUDA MPS. That deployment was
measured at 125 calls with no gapped turns and 150 with a 34 ms worst gap (November 2026, `bench_calls.py` plus
real transcriptions). That ASR load isn't in this repo.

If you transcribe on the same GPU, pick the ASR model carefully: Whisper `large-v3-turbo` always encodes a
padded 30 s window (~0.14 s of GPU per utterance on this card) and capped us at 50 calls. Parakeet only encodes
the real audio.

## Quality

`bench/bench_quality.py`: 12 French sentences × 12 = 144 clips per setup, read back by Whisper `large-v3-turbo`.
`bench/bench_similarity.py`: cosine similarity of each clip to the reference voice (SpeechBrain ECAPA-TDNN).

| | Stock + chunk ramp | All patches |
| --- | --- | --- |
| Character error rate, mean (max) | 0.02% (2.8%) | 0.00% (0.0%) |
| Speaker similarity to the reference, mean | 0.794 | 0.786 |

The similarity difference is within run-to-run noise (sampling at temperature 0.9): on two smaller runs of
48 clips, stock scored 0.783 and all patches 0.790. In one 48-clip run, one patched clip added two words at
the end ("C'est parti."); the 144-clip run had none.

## Usage

### Docker

```sh
docker build -t qwen3-tts-fast .
docker run --gpus all -p 8091:8091 -v ~/.cache:/root/.cache qwen3-tts-fast
```

### Existing vllm-omni 0.32.0 install

```sh
bash apply-patches.sh   # idempotent; patches the site-packages that holds vllm_omni and vllm
bash serve.sh           # writes the deploy config and runs `vllm serve ... --omni` on 127.0.0.1:8091
```

The first start captures CUDA graphs and autotunes the Triton kernels: about 6 minutes on a cold RTX 4000 Ada,
under 3 minutes once `TRITON_CACHE_DIR` holds the autotune results.

### Register a voice and stream speech

```sh
python register_voice.py reference.wav "Exact transcript of the reference clip." --name default

curl -s http://127.0.0.1:8091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Hello there!", "voice": "default", "task_type": "Base", "language": "English",
       "response_format": "pcm", "stream_format": "audio"}' > out.pcm   # 16-bit 24 kHz mono
```

Only clone voices you have the right to use.

### Benchmarks

```sh
pip install aiohttp requests
python bench/bench_tts.py --levels 1,16,24,32,64 --rounds 3 --warmup
python bench/bench_calls.py --calls 75,100,125,150,200,250,300 --seconds 90
pip install faster-whisper soundfile scipy numpy speechbrain
python bench/bench_quality.py --repeats 12 --out results/quality
python bench/bench_similarity.py reference.wav results/quality
```

Our runs used a 6.9 s French reference voice. Run `bench_tts.py` with `--warmup` before `bench_calls.py` so no
call pays for a Triton compile.

## Settings

| Variable | Default | |
| --- | --- | --- |
| `VLLM_OMNI_STREAM_CHUNK_RAMP` | `1,2,4,8,16,25` | Frames per streamed chunk (0001). One frame is 80 ms. |
| `VLLM_OMNI_CP_FP8` | `1` | FP8 code predictor (0004). Tuned for sm_89. |
| `VLLM_OMNI_STREAM_PRIME_CACHE` | `8` | Primed codec states kept per voice reference (0005). |
| `VLLM_FP8_W8A16_MAX_M` | `0` | W8A16 talker GEMMs up to this batch size (0006). |
| `QWEN3_TTS_MAX_STREAMS` | `96` | Talker `max_num_seqs`. Past ~300 calls, requests queue behind it. |
| `QWEN3_TTS_GPU_MEMORY` | `0.4` | `gpu_memory_utilization`. Raise it if nothing else shares the GPU. |
| `QWEN3_TTS_QUANTIZATION` | `fp8` | Talker linear layers, or `none`. |

## Limits

- Tested on one GPU model (RTX 4000 Ada, sm_89). The FP8 code predictor kernel targets sm_89. Other GPUs need
  their own measurements, and possibly `VLLM_OMNI_CP_FP8=0`.
- Pinned to vllm-omni 0.32.0 / vllm 0.32.0. The patches will need rebasing for later versions.
- The call simulation is synthetic, from one client on the same machine. Real callers have different turn
  lengths and pauses, and network jitter adds to the client buffer you need.

## License

Apache 2.0, same as vLLM and vLLM-Omni. Built by [Archeon Solutions](https://www.archeon.solutions).

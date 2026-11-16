"""Write the vLLM-Omni deploy config: the single-stage fused Qwen3-TTS pipeline tuned for one GPU.

Usage: python deploy_config.py out.yaml
Env: QWEN3_TTS_MAX_STREAMS (96), QWEN3_TTS_GPU_MEMORY (0.4), QWEN3_TTS_QUANTIZATION (fp8, or none).
"""
import os
import pathlib
import sys

import vllm_omni
import yaml


def build(env):
    deploy = pathlib.Path(vllm_omni.__file__).parent/'deploy'
    config = yaml.safe_load((deploy/'qwen3_tts_fused_single_gpu.yaml').read_text())
    talker = config['stages'][0]
    talker['max_num_seqs'] = int(env.get('QWEN3_TTS_MAX_STREAMS', '96'))
    talker['gpu_memory_utilization'] = float(env.get('QWEN3_TTS_GPU_MEMORY', '0.4'))
    # Every new clause is a new request, so most steps mix one prefill with running decodes.
    # FlashAttention 2 cannot CUDA-graph those steps; Triton attention can.
    talker['attention_config'] = {'backend': 'TRITON_ATTN'}
    talker['compilation_config']['cudagraph_mode'] = 'FULL'
    quantization = env.get('QWEN3_TTS_QUANTIZATION', 'fp8')
    if quantization != 'none':
        talker['quantization'] = quantization
    return config


if __name__ == '__main__':
    pathlib.Path(sys.argv[1]).write_text(yaml.safe_dump(build(os.environ), sort_keys=False))

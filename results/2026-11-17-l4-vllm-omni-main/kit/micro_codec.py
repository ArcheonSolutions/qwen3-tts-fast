"""Streaming codec decode time per step, Qwen3-TTS 12Hz codec dims (random weights), BF16, CUDA graphs not used."""
import json, sys, torch
from triton.testing import do_bench
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderConfig
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import StreamingCodecDecoder
c = json.load(open(sys.argv[1]))["decoder_config"]
cfg = Qwen3TTSTokenizerV2DecoderConfig(**{k: v for k, v in c.items()})
torch.manual_seed(0)
dec = Qwen3TTSTokenizerV2Decoder(cfg).eval()
with torch.no_grad():
    for n, p in dec.named_parameters():
        if n.endswith(("alpha", "beta")): p.uniform_(-0.5, 0.5)
dec.precompute_snake_caches()
dec = dec.to("cuda", torch.bfloat16)
dec.config.head_dim = c["head_dim"]
S = 129
stream = StreamingCodecDecoder(dec, num_slots=S, dtype=torch.bfloat16)
with torch.inference_mode():
    for B, T in [(1, 1), (16, 1), (32, 1), (64, 1), (128, 1), (1, 25), (16, 25)]:
        codes = torch.randint(0, 2048, (B, T, 16), device="cuda", dtype=torch.int32)
        slots = torch.arange(B, device="cuda", dtype=torch.int32)
        pos = torch.full((B,), 100, device="cuda", dtype=torch.int32)
        ms = do_bench(lambda: stream(codes, slots, pos), warmup=50, rep=300)
        print(json.dumps({"batch": B, "frames": T, "ms": round(ms, 3)}), flush=True)

"""Fused code predictor time per call (15 residual steps), Qwen3-TTS 1.7B predictor shapes (random weights): BF16 vs FP8 weights."""
import json, sys, torch
sys.path.insert(0, "/root/vllm-omni")
from triton.testing import do_bench
from tests.model_executor.models.qwen3_tts.test_fused_code_predictor_pdl import GROUPS, VOCAB, _predictor
from vllm_omni.model_executor.models.qwen3_tts import fused_code_predictor as fcp
with torch.inference_mode():
    preds = {"bf16": fcp.FusedCodePredictor(_predictor("cuda"), 128), "fp8": fcp.FusedCodePredictor(_predictor("cuda"), 128, fp8_weights=True)}
    assert preds["fp8"]._w8 and not preds["bf16"]._w8
    for B in (1, 8, 16, 32, 64, 128):
        code0 = torch.randint(0, VOCAB, (B, 1), device="cuda")
        emb = torch.randn(B, 1, 2048, device="cuda").to(torch.bfloat16)
        hid = torch.randn(B, 1, 2048, device="cuda").to(torch.bfloat16)
        u = torch.rand(B, GROUPS - 1, VOCAB, device="cuda").clamp_(1e-6, 1 - 1e-6)
        row = {"batch": B}
        for name, p in preds.items():
            g = torch.cuda.CUDAGraph()
            p(code0, emb, hid, 1 / 0.9, 50, u)
            with torch.cuda.graph(g):
                p(code0, emb, hid, 1 / 0.9, 50, u)
            row[name + "_ms"] = round(do_bench(g.replay, warmup=50, rep=300), 3)
        print(json.dumps(row), flush=True)

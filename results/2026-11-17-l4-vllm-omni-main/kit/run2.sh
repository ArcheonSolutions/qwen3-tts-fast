#!/bin/bash
P=/root/kit/phase.sh
TTS="bench/bench_tts.py --levels 1,16,32,64 --rounds 3 --warmup"
bash $P main main "{}" "bench/bench_tts.py --levels 1,8,16,32 --rounds 3 --warmup" "bench/bench_calls.py --calls 8,16,32 --seconds 60"
bash $P pr1 stack1 "{}" "$TTS" "bench/bench_calls.py --calls 16,32,50,75,100 --seconds 60"
bash $P pr2 stack2 "{}" "$TTS" "bench/bench_calls.py --calls 75,100,150,200 --seconds 60"
bash $P pr3 stack3 "{}" "$TTS" "bench/bench_calls.py --calls 100,150,200,250,300 --seconds 60"
bash $P pr4 stack4 "{\"code_predictor_fp8\": true}" "$TTS" "bench/bench_calls.py --calls 100,150,200,250,300 --seconds 60"
echo ALL_DONE

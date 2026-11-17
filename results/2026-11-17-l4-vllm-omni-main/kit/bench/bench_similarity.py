"""Speaker similarity of synthesized clips to the reference voice (SpeechBrain ECAPA-TDNN, cosine).

Usage: python bench_similarity.py reference.wav results/quality [more dirs...]
Run bench_quality.py first: it saves the clips as WAV files in its --out directory.
"""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier


def load(path):
    audio, rate = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        audio = resample_poly(audio, 16000, rate).astype(np.float32)
    return torch.from_numpy(audio).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('dirs', type=Path, nargs='+')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    model = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={'device': args.device})

    def embed(path):
        with torch.no_grad():
            return torch.nn.functional.normalize(model.encode_batch(load(path).to(args.device)).squeeze(), dim=0)

    reference = embed(args.reference)
    for folder in args.dirs:
        scores = [float(embed(clip) @ reference) for clip in sorted(folder.glob('*.wav'))]
        print(json.dumps({'dir': str(folder), 'clips': len(scores),
                          'similarity_mean': round(statistics.mean(scores), 4),
                          'similarity_min': round(min(scores), 4)}))


if __name__ == '__main__':
    main()

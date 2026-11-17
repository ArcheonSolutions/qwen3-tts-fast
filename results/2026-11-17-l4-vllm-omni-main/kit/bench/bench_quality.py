"""Synthesize a fixed French set through vLLM-Omni, transcribe it with Whisper, and report character error rate."""
import argparse
import asyncio
import json
from pathlib import Path
import re
import statistics
import unicodedata

import aiohttp
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SENTENCES = [
    "Bonjour, merci de votre appel, je suis ravi de pouvoir vous aider ce soir !",
    "Pouvez-vous me rappeler le numéro de votre commande, s'il vous plaît ?",
    "Votre colis a quitté l'entrepôt ce matin, et il arrive demain avant midi.",
    "Je note votre demande dans votre dossier, c'est promis.",
    "Nos équipes travaillent jour et nuit pour préparer toutes les livraisons.",
    "Est-ce que vous avez bien reçu le courriel de confirmation hier soir ?",
    "Il pleut beaucoup dans la région, et le livreur aura peut-être un peu de retard.",
    "Quel est le modèle qui vous intéresse le plus pour cet hiver ?",
    "Un vélo rouge, c'est un excellent choix, je vérifie le stock tout de suite.",
    "N'oubliez pas de garder votre facture jusqu'à la fin de la garantie.",
    "Votre voisin nous a aussi laissé un très gentil message.",
    "Je dois vous laisser maintenant, un autre client m'attend. Bonne soirée !",
]


def normalize(text):
    text = unicodedata.normalize('NFKD', text.lower())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


def edit_distance(a, b):
    row = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j]+1, row[j-1]+1, prev+(ca != cb))
    return row[-1]


async def synthesize(session, url, voice, text):
    payload = {'input': text, 'voice': voice, 'task_type': 'Base', 'language': 'French',
               'response_format': 'pcm', 'stream_format': 'audio'}
    async with session.post(f'{url}/v1/audio/speech', json=payload) as response:
        response.raise_for_status()
        return np.frombuffer(await response.read(), dtype='<i2').astype(np.float32)/32768


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8091')
    parser.add_argument('--voice', default='default')
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--concurrency', type=int, default=12)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, r, text) for r in range(args.repeats) for i, text in enumerate(SENTENCES)]
    limit = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as session:
        async def run(job):
            async with limit:
                return job, await synthesize(session, args.url, args.voice, job[2])
        results = await asyncio.gather(*(run(job) for job in jobs))
    from faster_whisper import WhisperModel
    model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
    rows = []
    for (i, r, text), audio in results:
        sf.write(out/f'{i:02d}-{r}.wav', audio, 24000)
        segments, _ = model.transcribe(resample_poly(audio, 2, 3).astype(np.float32), language='fr', beam_size=5,
                                       condition_on_previous_text=False, vad_filter=False)
        heard = ' '.join(s.text.strip() for s in segments)
        ref, hyp = normalize(text), normalize(heard)
        rows.append({'i': i, 'r': r, 'cer': edit_distance(ref, hyp)/max(1, len(ref)),
                     'seconds': round(len(audio)/24000, 2), 'heard': heard})
    (out/'transcripts.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    cer = [row['cer'] for row in rows]
    print(json.dumps({'out': str(out), 'clips': len(rows), 'cer_mean': round(statistics.mean(cer), 4),
                      'cer_median': round(statistics.median(cer), 4), 'cer_max': round(max(cer), 3),
                      'bad_clips': sum(c > .15 for c in cer),
                      'seconds_mean': round(statistics.mean(row['seconds'] for row in rows), 2)}))


if __name__ == '__main__':
    asyncio.run(main())

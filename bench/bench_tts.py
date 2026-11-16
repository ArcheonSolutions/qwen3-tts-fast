"""Load test vLLM-Omni Qwen3-TTS: first audio, real-time factor and playback underruns per concurrent stream.

Each of N workers streams clauses back to back. A stream "has a gap" when a client that starts playing
after --buffer seconds would wait more than 100 ms for audio."""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp

RATE = 24000
CLAUSES = {
    'fr': ["Bonjour, merci de votre appel, je suis ravi de pouvoir vous aider ce soir !",
           "Pouvez-vous me rappeler le numéro de votre commande, s'il vous plaît ?",
           "Votre colis a quitté l'entrepôt ce matin, et il arrive demain avant midi.",
           "Je note votre demande dans votre dossier, c'est promis."],
    'en': ["Hello, thanks for calling, I'm really happy to help you out this evening!",
           "Could you remind me of your order number, please?"],
}
LANGUAGES = {'fr': 'French', 'en': 'English', 'de': 'German', 'es': 'Spanish', 'it': 'Italian',
             'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'}

async def stream(session, url, voice, language, text, buffer_seconds):
    payload = {'input': text, 'voice': voice, 'task_type': 'Base',
               'language': LANGUAGES[language],
               'response_format': 'pcm', 'stream_format': 'audio'}
    started = time.perf_counter()
    first = None
    received = 0
    underrun = 0.
    async with session.post(f'{url}/v1/audio/speech', json=payload,
                            timeout=aiohttp.ClientTimeout(total=120)) as response:
        response.raise_for_status()
        async for data in response.content.iter_any():
            now = time.perf_counter()
            if first is None:
                first = now
            played = max(0., now-first-buffer_seconds)
            have = received/2/RATE
            if have < played:
                underrun = max(underrun, played-have)
            received += len(data)
    done = time.perf_counter()
    audio = received/2/RATE
    return {'ttfa': first-started, 'total': done-started, 'audio': audio,
            'rtf': (done-started)/audio if audio else float('inf'), 'underrun': underrun}


async def level(url, voice, language, concurrency, rounds, buffer_seconds):
    texts = CLAUSES[language]
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=2)) as session:
        await stream(session, url, voice, language, texts[0], buffer_seconds)
        results = []
        wall = time.perf_counter()

        async def worker(index):
            for turn in range(rounds):
                results.append(await stream(session, url, voice, language,
                                            texts[(index+turn) % len(texts)], buffer_seconds))

        await asyncio.gather(*(worker(i) for i in range(concurrency)))
        wall = time.perf_counter()-wall
    ttfa = sorted(r['ttfa'] for r in results)
    rtf = [r['rtf'] for r in results]
    return {'concurrency': concurrency, 'requests': len(results),
            'ttfa_p50': round(statistics.median(ttfa), 3),
            'ttfa_p95': round(ttfa[min(len(ttfa)-1, int(len(ttfa)*.95))], 3),
            'rtf_p50': round(statistics.median(rtf), 3), 'rtf_max': round(max(rtf), 3),
            'underrun_max': round(max(r['underrun'] for r in results), 3),
            'underrun_p95': round(sorted(r['underrun'] for r in results)[min(len(results)-1, int(len(results)*.95))], 3),
            'gapped_pct': round(100*sum(r['underrun'] > .1 for r in results)/len(results), 1),
            'audio_per_wall': round(sum(r['audio'] for r in results)/wall, 2)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8091')
    parser.add_argument('--voice', default='default')
    parser.add_argument('--language', default='fr')
    parser.add_argument('--levels', default='1,2,4,8,12,16')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--buffer', type=float, default=.2)
    parser.add_argument('--warmup', action='store_true')
    args = parser.parse_args()
    if args.warmup:
        for concurrency in (int(n) for n in args.levels.split(',')):
            await level(args.url, args.voice, args.language, concurrency, 1, args.buffer)
    for concurrency in (int(n) for n in args.levels.split(',')):
        print(json.dumps(await level(args.url, args.voice, args.language, concurrency,
                                     args.rounds, args.buffer)), flush=True)


if __name__ == '__main__':
    asyncio.run(main())

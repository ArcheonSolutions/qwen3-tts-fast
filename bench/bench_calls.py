"""Simulate concurrent voice-agent calls against vLLM-Omni: turns of streamed clauses with real-time playback and pauses.

Each call speaks a turn of 2-3 clauses, requesting the next clause while the previous one plays, then waits
3-7 s (the caller talking) before the next turn. A turn "has a gap" when playback waits more than --gap for audio.
This script loads TTS only; the published numbers also ran speech recognition on the same GPU."""
import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp

RATE = 24000
TURNS = [
    ["Bonjour, merci de votre appel !", "Comment puis-je vous aider ?"],
    ["Très bien, je regarde ça.", "Pouvez-vous me donner votre numéro de commande ?", "Il est sur votre facture."],
    ["Un vélo rouge, c'est un excellent choix.", "Je vérifie le stock tout de suite."],
    ["Votre colis est parti de l'entrepôt ce matin.", "Il arrive demain avant midi !", "Serez-vous chez vous ?"],
    ["N'oubliez pas de garder votre facture jusqu'à la fin de la garantie.", "Elle vous sera utile en cas de retour."],
]


class Clause:
    def __init__(self, session, url, voice, text):
        self.chunks = asyncio.Queue()
        self.started = time.perf_counter()
        self.task = asyncio.create_task(self._run(session, url, voice, text))

    async def _run(self, session, url, voice, text):
        payload = {'input': text, 'voice': voice, 'task_type': 'Base', 'language': 'French',
                   'response_format': 'pcm', 'stream_format': 'audio'}
        try:
            async with session.post(f'{url}/v1/audio/speech', json=payload,
                                    timeout=aiohttp.ClientTimeout(total=60)) as response:
                response.raise_for_status()
                async for data in response.content.iter_any():
                    self.chunks.put_nowait((time.perf_counter(), len(data)/2/RATE))
        finally:
            self.chunks.put_nowait(None)


async def call(session, url, voice, seconds, stats, gap_threshold, start_delay):
    await asyncio.sleep(start_delay)
    deadline = time.perf_counter()+seconds
    turn_index = random.randrange(len(TURNS))
    while time.perf_counter() < deadline:
        texts = TURNS[turn_index % len(TURNS)]
        turn_index += 1
        turn_started = time.perf_counter()
        clauses = [Clause(session, url, voice, texts[0])]
        playhead = None
        worst_gap = 0.
        for index in range(len(texts)):
            if index+1 < len(texts):
                await asyncio.sleep(random.uniform(.05, .3))
                clauses.append(Clause(session, url, voice, texts[index+1]))
            current = clauses[index]
            while (item := await current.chunks.get()) is not None:
                arrived, duration = item
                if playhead is None:
                    stats['turn_ttfa'].append(arrived-turn_started)
                    playhead = arrived
                elif arrived > playhead:
                    worst_gap = max(worst_gap, arrived-playhead)
                    playhead = arrived
                playhead += duration
            await current.task
        stats['turns'] += 1
        stats['gapped'] += worst_gap > gap_threshold
        stats['worst_gap'] = max(stats['worst_gap'], worst_gap)
        listen = max(0., playhead-time.perf_counter()) if playhead else 0.
        await asyncio.sleep(listen+random.uniform(3, 7))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8091')
    parser.add_argument('--calls', default='25,50,100')
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--gap', type=float, default=.1)
    parser.add_argument('--turn-scale', type=int, default=1)
    parser.add_argument('--voice', default='default')
    args = parser.parse_args()
    if args.turn_scale > 1:
        TURNS[:] = [turn*args.turn_scale for turn in TURNS]
    random.seed(7)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=2)) as session:
        for calls in (int(n) for n in args.calls.split(',')):
            stats = {'turn_ttfa': [], 'turns': 0, 'gapped': 0, 'worst_gap': 0.}
            ramp = min(20., args.seconds/3)
            await asyncio.gather(*(call(session, args.url, args.voice, args.seconds, stats, args.gap, random.uniform(0, ramp))
                                   for _ in range(calls)))
            ttfa = sorted(stats['turn_ttfa'])
            print(json.dumps({'calls': calls, 'turns': stats['turns'],
                              'turn_ttfa_p50': round(statistics.median(ttfa), 3),
                              'turn_ttfa_p95': round(ttfa[int(len(ttfa)*.95)], 3),
                              'turn_ttfa_max': round(ttfa[-1], 3),
                              'gapped_turns_pct': round(100*stats['gapped']/max(1, stats['turns']), 1),
                              'worst_gap': round(stats['worst_gap'], 3)}), flush=True)


if __name__ == '__main__':
    asyncio.run(main())

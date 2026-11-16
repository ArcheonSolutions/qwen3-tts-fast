"""Upload a reference voice to a running vLLM-Omni server.

Usage: python register_voice.py reference.wav "Exact transcript of the clip." [--name default]
Use 3-12 s of clean speech from one speaker, mono, no music.
"""
import argparse
import sys
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', type=Path)
    parser.add_argument('transcript')
    parser.add_argument('--name', default='default')
    parser.add_argument('--url', default='http://127.0.0.1:8091')
    parser.add_argument('--consent', default='reference-owner-consented',
                        help='Consent identifier stored with the voice. Only clone voices you have the right to use.')
    args = parser.parse_args()
    response = requests.post(f'{args.url}/v1/audio/voices', timeout=120,
                             files={'audio_sample': (args.wav.name, args.wav.read_bytes(), 'audio/wav')},
                             data={'name': args.name, 'consent': args.consent, 'ref_text': args.transcript})
    if response.status_code != 200:
        sys.exit(f'voice registration failed ({response.status_code}): {response.text}')
    print(f'registered voice {args.name!r}')


if __name__ == '__main__':
    main()

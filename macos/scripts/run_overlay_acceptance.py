#!/usr/bin/env python3
"""Run blur then LaMa through the same Swift JobRunner used by the app."""
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[2]
video, = list((repo/'downloads').glob('*EP.22*.mp4'))
probe = repo/'macos/.build/Build/Products/Debug/SubtitleRefineProbe'
root = repo/'macos/.build/m3'
root.mkdir(parents=True, exist_ok=True)
for cover, duration in [('blur', '30'), ('lama', '6')]:
    log = root/f'{cover}-run.log'
    print(f'Running {cover}, start 530, duration {duration}; log={log}', flush=True)
    with log.open('w') as output:
        subprocess.run([str(probe), 'overlay', str(video), str(root/cover), cover, duration],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    print(f'{cover}: PASS', flush=True)

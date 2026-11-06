#!/usr/bin/env python3
"""Two separate app-code probe processes: interrupt real OCR, then resume its cache."""
from pathlib import Path
import hashlib
import json
import subprocess

repo = Path(__file__).resolve().parents[2]
video, = list((repo/'downloads').glob('*EP.22*.mp4'))
probe = repo/'macos/.build/Build/Products/Debug/SubtitleRefineProbe'
root = repo/'macos/.build/m4-restart'
root.mkdir(parents=True, exist_ok=True)
with (root/'interrupt.log').open('w') as log:
    subprocess.run([str(probe), 'interrupt-overlay', str(video), str(root), 'blur', '6'], stdout=log, stderr=subprocess.STDOUT, check=True)
id = (root/'interrupted-job-id.txt').read_text()
job = next(j for j in json.loads((root/'jobs.json').read_text()) if j['id'] == id)
assert job['status'] == 'interrupted'
output = root/'jobs'/id/'output'
work, = list((output/'overlay').glob('*.onscreen-work'))
def stamps():
    return {name: {'sha256': hashlib.sha256((work/name).read_bytes()).hexdigest(), 'mtime_ns': (work/name).stat().st_mtime_ns}
            for name in ['clip.mp4', 'ocr.jsonl', 'events.json']}
before = stamps()
with (root/'resume.log').open('w') as log:
    subprocess.run([str(probe), 'resume', str(root)], stdout=log, stderr=subprocess.STDOUT, check=True)
after = stamps()
assert before == after, 'Resume must reuse the existing clip, OCR and event cache without rewriting'
job = next(j for j in json.loads((root/'jobs.json').read_text()) if j['id'] == id)
assert job['status'] == 'completed'
manifest = Path(job['manifestPath'])
assert manifest.parent == output
assert len(json.loads(manifest.read_text())['artifacts']) == 4
(root/'verification.json').write_text(json.dumps({'job_id': id, 'work': str(work), 'before': before, 'after': after,
    'status_before': 'interrupted', 'status_after': job['status'], 'manifest': str(manifest)}, ensure_ascii=False, indent=2))
print(f'PASS: two separate probe launches; interrupted -> completed; identical cache hashes and mtimes; {manifest}')

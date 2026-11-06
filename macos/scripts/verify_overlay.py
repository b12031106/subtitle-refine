#!/usr/bin/env python3
"""Decode M3 outputs and extract one PNG per cover mode via static_ffmpeg."""
from pathlib import Path
import hashlib
import json
import subprocess
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

repo = Path(__file__).resolve().parents[2]
root = repo/'macos/.build/m3'
ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise()
summary = {}
for cover, expected in [('blur',30),('lama',6)]:
    job_root = root/cover
    manifest = Path((job_root/'last-manifest-path.txt').read_text())
    artifacts = {a['kind']:manifest.parent/a['path'] for a in json.loads(manifest.read_text())['artifacts']}
    assert {'overlay_video','overlay_review','overlay_ass','overlay_events'} <= artifacts.keys()
    for path in artifacts.values():
        assert path.is_file() and path.stat().st_size
        assert path.resolve().is_relative_to(manifest.parent.resolve())
    review = artifacts['overlay_review'].read_text()
    assert '<!doctype html>' in review.lower() and 'data:image/jpeg;base64,' in review
    assert '原畫面' in review and '成品' in review
    events = json.loads(artifacts['overlay_events'].read_text())
    translated = [e for e in events if e.get('translate')]
    assert translated and any(any('\u4e00'<=c<='\u9fff' for c in e.get('translation','')) for e in translated)
    probe = json.loads(subprocess.check_output([ffprobe,'-v','error','-show_format','-show_streams','-of','json',str(artifacts['overlay_video'])]))
    duration = float(probe['format']['duration'])
    assert abs(duration-expected)<0.2, duration
    assert any(s['codec_type']=='video' and s['codec_name']=='h264' for s in probe['streams'])
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    subprocess.run([ffmpeg,'-v','error','-i',str(artifacts['overlay_video']),'-f','null','-'],check=True)
    frame = root/f'{cover}-frame.png'
    subprocess.run([ffmpeg,'-v','error','-y','-ss','1','-i',str(artifacts['overlay_video']),'-frames:v','1',str(frame)],check=True)
    summary[cover] = {'manifest':str(manifest),'duration':duration,'events':len(events),'translated':len(translated),
        'flagged':sum(bool(e.get('flags')) for e in events),'frame':str(frame),'ffprobe':probe,
        'artifacts':{kind:{'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for kind,path in artifacts.items()}}
    print(f'{cover}: PASS; duration={duration}; events={len(events)}; translated={len(translated)}; frame={frame}',flush=True)
(root/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

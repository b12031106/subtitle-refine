#!/usr/bin/env python3
"""Validate actual M2 outputs and extract burned-subtitle frames for visual review."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

root = Path(sys.argv[1]).resolve()
youtube = Path(sys.argv[2]).resolve()
manifest = Path((root/'last-manifest-path.txt').read_text())
ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise()
content = json.loads(manifest.read_text())
assert content['version'] == 1
artifacts = {}
for item in content['artifacts']:
    path = manifest.parent/item['path']
    data = path.read_bytes()
    assert data
    artifacts[item['kind']] = {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
assert {'srt','vtt','corrected_srt','burned_video'} <= artifacts.keys()
assert (manifest.parent/'subtitle.vtt').read_text().startswith('WEBVTT')
assert any('\u4e00' <= ch <= '\u9fff' for ch in (manifest.parent/'subtitle.srt').read_text())
assert 'Hello' in (manifest.parent/'corrected.srt').read_text()

def probe(path):
    return json.loads(subprocess.check_output([ffprobe,'-v','error','-show_format','-show_streams','-of','json',str(path)]))

burned = Path(artifacts['burned_video']['path'])
video_probe = probe(burned)
assert abs(float(video_probe['format']['duration'])-8) < 0.15
assert any(s['codec_name']=='h264' and s['codec_type']=='video' for s in video_probe['streams'])
assert any(s['codec_name']=='aac' and s['codec_type']=='audio' for s in video_probe['streams'])
for path in [burned,youtube]:
    subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'],check=True)
frames = root/'frames'
frames.mkdir(exist_ok=True)
for second in [1,6]:
    subprocess.run([ffmpeg,'-v','error','-y','-ss',str(second),'-i',str(burned),'-frames:v','1',str(frames/f'burned-{second}s.png')],check=True)
result={'manifest':str(manifest),'artifacts':artifacts,'burned_probe':video_probe,'youtube_probe':probe(youtube)}
(root/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps({'verified':True,'duration':video_probe['format']['duration'],'youtube_duration':result['youtube_probe']['format']['duration'],'report':str(root/'verification.json'),'frames':str(frames)},ensure_ascii=False))

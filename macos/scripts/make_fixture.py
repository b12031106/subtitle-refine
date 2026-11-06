#!/usr/bin/env python3
"""Create an eight-second spoken fixture; uses only say and static_ffmpeg."""
import json
from pathlib import Path
import subprocess
import sys
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'macos/.build/e2e/fixture').resolve()
root.mkdir(parents=True, exist_ok=True)
ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise()
speech = 'Hello everyone. Today is a sunny day. We are testing automatic subtitles and translation in this short video.'
subprocess.run(['/usr/bin/say', '-v', 'Samantha', '-r', '180', '-o', str(root/'speech.aiff'), speech], check=True)
subprocess.run([ffmpeg, '-hide_banner', '-y', '-f', 'lavfi', '-i', 'color=c=0x17394f:s=960x540:r=24',
                '-i', str(root/'speech.aiff'), '-af', 'apad', '-t', '8', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-movflags', '+faststart', str(root/'english-eight-seconds.mp4')], check=True)
probe = subprocess.check_output([ffprobe, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(root/'english-eight-seconds.mp4')])
(root/'ffprobe.json').write_bytes(probe)
(root/'speech.txt').write_text(speech+'\n')
print(json.dumps({'video':str(root/'english-eight-seconds.mp4'), 'ffmpeg':ffmpeg, 'duration':json.loads(probe)['format']['duration']}))

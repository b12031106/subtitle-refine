#!/usr/bin/env python3
"""Reburn real cached subtitles with AI/ASR calls forbidden; original job untouched."""
from pathlib import Path
import json, shutil, subprocess
from unittest.mock import patch
from subtitle_engine import worker
from subtitle_engine.processor import VideoSubtitleProcessor

repo=Path(__file__).resolve().parents[2]
root=repo/'macos/.build/subtitle-bg'
manifest=Path((root/'none/last-manifest-path.txt').read_text())
output=root/'cache-reburn'
if output.exists():
    shutil.rmtree(output)
shutil.copytree(manifest.parent,output)
options=json.loads((manifest.parent.parent/'options.json').read_text())
def stamps():
    return {name:{'sha256':worker.digest(output/name),'mtime_ns':(output/name).stat().st_mtime_ns}
        for name in ['transcribed.srt','corrected.srt','subtitle.srt']}
before=stamps(); checkpoint=json.loads((output/'checkpoint.json').read_text())
commands=[]; run=worker.run_command

def guarded(argv,cwd=None):
    assert 'whisper' not in argv, 'Changing only background must not run ASR'
    commands.append(argv)
    return run(argv,cwd=cwd)

with patch.object(VideoSubtitleProcessor,'correct_subtitle_with_llm',side_effect=AssertionError('Changing only background must not call AI')), patch.object(worker,'run_command',side_effect=guarded):
    worker.execute(root/'lecture-eight-seconds.mp4',output,options|{'subtitle_background':'box','subtitle_background_opacity':80},lambda _:None)
    assert len(commands)==1 and '-vf' in commands[0]
    assert 'BorderStyle=3' in commands[0][commands[0].index('-vf')+1]
    assert stamps()==before
    changed=json.loads((output/'checkpoint.json').read_text())
    assert changed['signature']==checkpoint['signature']
    assert changed['burn_signature']!=checkpoint['burn_signature']
    worker.execute(root/'lecture-eight-seconds.mp4',output,options|{'subtitle_background':'box','subtitle_background_opacity':80},lambda _:None)
    assert len(commands)==1, 'Unchanged settings reuse the burned video'
summary={'before':before,'after':stamps(),'text_signature_unchanged':True,'burn_signature_changed':True,
    'media_calls':len(commands),'ai_calls':0,'asr_calls':0,'same_settings_media_calls':0,'manifest':str(output/'manifest.json')}
(root/'cache-verification.json').write_text(json.dumps(summary,indent=2))
# Also exercise the old CLI's actual --only-embed path, with the same installed engine.
legacy=root/'legacy';legacy.mkdir(exist_ok=True)
import os,sys
env=os.environ.copy();env['PYTHON_DOTENV_DISABLED']='1'
with (root/'legacy-cli.log').open('w') as log:
    subprocess.run([sys.executable,'-m','subtitle_engine.cli','--video',str(root/'lecture-eight-seconds.mp4'),
        '--subtitle',str(manifest.parent/'subtitle.srt'),'--only-embed','--subtitle-background','box',
        '--subtitle-background-opacity','80'],cwd=legacy,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
assert (legacy/'output/lecture-eight-seconds_with_subtitles.mp4').stat().st_size>0
print(json.dumps(summary,indent=2))
print('Legacy CLI box burn: PASS')

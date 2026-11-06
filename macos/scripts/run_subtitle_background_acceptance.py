#!/usr/bin/env python3
"""Real speech/AI/burn through the app's JobRunner, with readable lecture clutter."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

repo=Path(__file__).resolve().parents[2]
root=repo/'macos/.build/subtitle-bg'
root.mkdir(parents=True,exist_ok=True)
ffmpeg,ffprobe=get_or_fetch_platform_executables_else_raise()
probe=repo/'macos/.build/Build/Products/Debug/SubtitleRefineProbe'
worker=repo/'macos/Resources/subtitle_worker.py'
lecture=Image.new('RGB',(1280,720),(68,89,104)); draw=ImageDraw.Draw(lecture)
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',30)
for y in range(25,710,43):
    draw.text((24,y),'EVERYONE HAS AI. WHO HAS THE ADVANTAGE?   ACCESS IS WIDESPREAD.',font=font,fill='white')
lecture.save(root/'lecture.png')
speech='Hello everyone. Today we are testing subtitles. Clear captions help us read the words on a busy screen.'
subprocess.run(['/usr/bin/say','-v','Samantha','-r','175','-o',str(root/'speech.aiff'),speech],check=True)
video=root/'lecture-eight-seconds.mp4'
with (root/'fixture.log').open('w') as log:
    subprocess.run([ffmpeg,'-hide_banner','-y','-loop','1','-framerate','24','-i',str(root/'lecture.png'),
        '-i',str(root/'speech.aiff'),'-af','apad','-t','8','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',
        '-movflags','+faststart',str(video)],stdout=log,stderr=subprocess.STDOUT,check=True)
results={}; images={}
for mode in ['none','box']:
    jobs=root/mode
    with (root/f'{mode}-run.log').open('w') as log:
        subprocess.run([str(probe),'background',str(video),str(jobs),str(worker),mode],stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest=Path((jobs/'last-manifest-path.txt').read_text())
    artifacts={a['kind']:manifest.parent/a['path'] for a in json.loads(manifest.read_text())['artifacts']}
    assert {'srt','vtt','corrected_srt','burned_video'} <= artifacts.keys()
    options=json.loads((manifest.parent.parent/'options.json').read_text())
    assert options['subtitle_background']==mode and options['subtitle_background_opacity']==80
    media=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_format','-show_streams','-of','json',str(artifacts['burned_video'])]))
    assert abs(float(media['format']['duration'])-8)<0.2
    assert any(s['codec_type']=='audio' for s in media['streams'])
    subprocess.run([ffmpeg,'-v','error','-i',str(artifacts['burned_video']),'-f','null','-'],check=True)
    frame=root/f'{mode}-frame.png'
    subprocess.run([ffmpeg,'-v','error','-y','-ss','3','-i',str(artifacts['burned_video']),'-frames:v','1',str(frame)],check=True)
    images[mode]=np.asarray(Image.open(frame).convert('L'),dtype=float)
    results[mode]={'manifest':str(manifest),'frame':str(frame),'options':options,
        'artifacts':{kind:{'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for kind,path in artifacts.items()},
        'duration':float(media['format']['duration'])}
    print(f'{mode}: full JobRunner -> Whisper -> correction -> translation -> burn PASS; {frame}',flush=True)
# Fixed subtitle-area ROI, independent of the renderer implementation and AI's wording.
a=images['none'][610:710,128:1152]; b=images['box'][610:710,128:1152]
metrics={'roi':[128,610,1152,710],'none_mean':float(a.mean()),'box_mean':float(b.mean()),
    'darkened_pixels_over_30':int(((a-b)>30).sum()),'upper_frame_mean_absolute_delta':float(abs(images['none'][:500]-images['box'][:500]).mean())}
assert metrics['none_mean']-metrics['box_mean']>2,metrics
assert metrics['darkened_pixels_over_30']>1000,metrics
assert metrics['upper_frame_mean_absolute_delta']<1,metrics
results['brightness']=metrics
(root/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(metrics,indent=2))

#!/usr/bin/env python3
"""Reburn exactly the accepted lecture clip/SRT with the current 80% default."""
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image
from subtitle_engine.burn import subtitle_filter
from subtitle_engine.processor import VideoSubtitleProcessor, get_ffmpeg_path
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

repo=Path(__file__).resolve().parents[2]
root=repo/'macos/.build/subtitle-bg'
manifest=Path((root/'box/last-manifest-path.txt').read_text())
video=root/'lecture-eight-seconds.mp4'
srt=manifest.parent/'subtitle.srt'
out=root/'box80'
out.mkdir(exist_ok=True)
ffmpeg,ffprobe=get_or_fetch_platform_executables_else_raise()
assert get_ffmpeg_path() == ffmpeg
# Use the shared production burn implementation; no new transcription/translation.
processor=VideoSubtitleProcessor()
result=Path(processor.embed_subtitle_to_video(str(video),str(srt),str(out),subtitle_background='box'))
style=subtitle_filter('box')
assert 'OutlineColour=&H33000000' in style and 'Outline=4' in style
frame=root/'box80-frame.png'
subprocess.run([ffmpeg,'-v','error','-y','-ss','3','-i',str(result),'-frames:v','1',str(frame)],check=True)
subprocess.run([ffmpeg,'-v','error','-i',str(result),'-f','null','-'],check=True)
probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_format','-show_streams','-of','json',str(result)]))
assert abs(float(probe['format']['duration'])-8)<0.2
assert any(stream['codec_type']=='audio' for stream in probe['streams'])
old=np.asarray(Image.open(root/'box-frame.png').convert('L'),dtype=float)
new=np.asarray(Image.open(frame).convert('L'),dtype=float)
roi=(slice(610,710),slice(128,1152))
metrics={'box60_mean':float(old[roi].mean()),'box80_mean':float(new[roi].mean()),
    'further_darkened_pixels_over_20':int(((old[roi]-new[roi])>20).sum()),
    'upper_frame_mean_absolute_delta':float(abs(old[:500]-new[:500]).mean())}
assert metrics['box80_mean']<metrics['box60_mean']-3,metrics
assert metrics['further_darkened_pixels_over_20']>5000,metrics
assert metrics['upper_frame_mean_absolute_delta']<1,metrics
# Measure actual glyph-to-box insets on a plain frame, preserving the same text/font/filter.
import shutil
shutil.copyfile(srt,out/'subtitle.srt')
padding=out/'padding-proof.png'
subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','color=c=0x445968:s=1280x720:r=24',
    '-ss','3','-vf',style,'-frames:v','1',str(padding)],cwd=out,check=True)
a=np.asarray(Image.open(padding).convert('L'))
def bounds(mask):
    yy,xx=np.where(mask)
    return [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
glyph=bounds(a>200); box=bounds(a<40)
insets={'left':glyph[0]-box[0],'top':glyph[1]-box[1],'right':box[2]-glyph[2],'bottom':box[3]-glyph[3]}
height=glyph[3]-glyph[1]
assert all(height*0.35<=n<=height*0.70 for n in insets.values()), (glyph,box,insets)
report={'video':str(result),'frame':str(frame),'source':str(video),'subtitle':str(srt),
    'opacity':80,'filter':style,'brightness':metrics,'glyph_bounds':glyph,'box_bounds':box,
    'glyph_height':height,'padding_pixels':insets,'duration':float(probe['format']['duration'])}
(root/'box80-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))

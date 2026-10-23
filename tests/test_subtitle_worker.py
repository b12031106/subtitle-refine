import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import subtitle_worker as worker
from subtitle_refine import VideoSubtitleProcessor

SRT = '1\n00:00:01,000 --> 00:00:02,000\nHello\n'


class WorkerTests(unittest.TestCase):
    def test_cli_stdout_is_jsonl_even_with_native_and_child_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = Path(tmp) / 'options.json'
            options.write_text('{}')
            code = """
import os, sys
import subtitle_worker as w
def execute(video, output, options, emit):
    print('python diagnostic', flush=True)
    os.write(1, b'native diagnostic\\n')
    w.run_command([sys.executable, '-c', "print('child diagnostic')"])
    emit({'type':'result','manifest':'/test/manifest.json'})
w.execute = execute
sys.exit(w.main())
"""
            result = subprocess.run([sys.executable, '-c', code, '--input', 'unused',
                                     '--options-json', str(options), '--output-dir', tmp],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([json.loads(line) for line in result.stdout.splitlines()],
                             [{'type': 'result', 'manifest': '/test/manifest.json'}])
            for label in ('python diagnostic', 'native diagnostic', 'child diagnostic'):
                self.assertIn(label, result.stderr)

    @unittest.skipIf(os.name == 'nt', 'POSIX process-group cancellation test')
    def test_sigterm_stops_media_child_and_does_not_emit_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = Path(tmp) / 'options.json'
            options.write_text('{}')
            child_pid = Path(tmp) / 'child.pid'
            code = """
import sys
import subtitle_worker as w
def execute(video, output, options, emit):
    w.run_command([sys.executable, '-c', 'import os,time,pathlib; pathlib.Path(' + repr(video) + ').write_text(str(os.getpid())); time.sleep(30)'])
w.execute = execute
sys.exit(w.main())
"""
            process = subprocess.Popen([sys.executable, '-c', code, '--input', str(child_pid),
                                        '--options-json', str(options), '--output-dir', tmp],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while not child_pid.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(child_pid.exists())
                pid = int(child_pid.read_text())
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 143, stderr)
                self.assertEqual(stdout, '')
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    state = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
                    if not state or state.startswith('Z'):
                        break
                    time.sleep(0.05)
                self.assertTrue(not state or state.startswith('Z'), state)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def test_options_reject_secrets_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'options.json'
            for data in ({'api_key': 'secret'}, {'burn_in': 'false'}, {'max_workers': True}, {'provider': 'bad'}, []):
                file.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    worker.options_from_file(file)
            file.write_text('{}')
            self.assertEqual(worker.options_from_file(file)['provider'], 'local-runtime')

    def test_vtt_changes_only_timestamp_commas(self):
        result = worker.to_vtt(SRT.replace('Hello', 'Hello, world'))
        self.assertTrue(result.startswith('WEBVTT\n\n00:00:01.000 --> 00:00:02.000'))
        self.assertIn('Hello, world', result)

    def test_pipeline_artifacts_resume_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'video.mp4'
            video.write_bytes(b'fake input')
            output = root / 'output'
            processor = VideoSubtitleProcessor()
            processor.client = Mock()
            processor.correct_subtitle_with_llm = Mock(return_value=SRT)
            commands, events = [], []
            def run(argv, cwd=None):
                commands.append(argv)
                if 'whisper' in argv:
                    Path(argv[argv.index('--output_dir') + 1], 'audio.srt').write_text(SRT)
                else:
                    Path(argv[-1]).write_bytes(b'fake media')
            options = worker.DEFAULTS | {'burn_in': True}
            with patch.object(worker, 'processor_for', return_value=processor), patch.object(worker, 'run_command', side_effect=run), patch.object(worker, 'get_ffmpeg_path', return_value='/usr/bin/ffmpeg'):
                manifest = worker.execute(video, output, options, events.append)
                result = json.loads(manifest.read_text())
                self.assertEqual({x['kind'] for x in result['artifacts']}, {'srt', 'vtt', 'corrected_srt', 'burned_video'})
                self.assertEqual(events[-1], {'type': 'result', 'manifest': str(manifest)})
                self.assertEqual(processor.correct_subtitle_with_llm.call_count, 2)
                count = len(commands)
                worker.execute(video, output, options, events.append)
                self.assertEqual(len(commands), count)
                self.assertEqual(processor.correct_subtitle_with_llm.call_count, 2)
                (output / 'corrected.srt').write_text('corrupt')
                worker.execute(video, output, options, events.append)
                self.assertEqual(processor.correct_subtitle_with_llm.call_count, 4)
                self.assertEqual(len(commands), count + 1)  # only burn reruns, not transcription
                # Changing translation settings preserves the expensive ASR stage.
                asr_count = sum('whisper' in command for command in commands)
                worker.execute(video, output, options | {'context': 'updated glossary'}, events.append)
                self.assertEqual(sum('whisper' in command for command in commands), asr_count)
                worker.execute(video, output, options | {'source_language': 'en'}, events.append)
                self.assertEqual(sum('whisper' in command for command in commands), asr_count + 1)

    def test_failure_never_publishes_manifest_or_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'video.mp4'
            video.write_bytes(b'fake')
            output = root / 'out'
            output.mkdir()
            (output / 'manifest.json').write_text('{}')
            events = []
            processor = VideoSubtitleProcessor()
            processor.client = Mock()
            with patch.object(worker, 'processor_for', return_value=processor), patch.object(worker, 'get_ffmpeg_path', return_value='/usr/bin/ffmpeg'), patch.object(worker, 'run_command', side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):
                    worker.execute(video, output, worker.DEFAULTS, events.append)
            self.assertFalse((output / 'manifest.json').exists())
            self.assertFalse(any(e['type'] == 'result' for e in events))


class CaptionInputTests(unittest.TestCase):
    def test_import_skips_whisper_exports_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'video.mp4'
            video.write_bytes(b'video')
            caption = root / 'source.vtt'
            caption.write_text('WEBVTT\n\n00:01.000 --> 00:02.000 align:start\n<c>Hello</c>\n')
            options = worker.DEFAULTS | dict(subtitle_source='youtube', youtube_subtitle_language='en',
                                             youtube_subtitle_kind='manual', burn_in=True)
            processor = VideoSubtitleProcessor()
            processor.client = Mock()
            processor.correct_subtitle_with_llm = Mock(side_effect=lambda text, **kwargs: text)
            commands, events = [], []
            def run(argv, cwd=None):
                commands.append(argv)
                self.assertNotIn('whisper', argv)
                self.assertIn('subtitles=subtitle.srt', argv)
                Path(argv[-1]).write_bytes(b'burned')
            with patch.object(worker, 'processor_for', return_value=processor), patch.object(worker, 'get_ffmpeg_path', return_value='/usr/bin/ffmpeg'), patch.object(worker, 'run_command', side_effect=run):
                def execute(): return worker.execute(video, root/'out', options, events.append, subtitle_input=caption)
                manifest = json.loads(execute().read_text())
                self.assertEqual({a['kind'] for a in manifest['artifacts']}, {'srt','vtt','corrected_srt','burned_video'})
                self.assertEqual((root/'out/transcribed.srt').read_text(), SRT)
                execute()
                self.assertEqual(len(commands), 1)
                self.assertEqual(processor.correct_subtitle_with_llm.call_count, 2)
                # Changing input bytes invalidates import, correction, translation and burn.
                caption.write_text(caption.read_text().replace('Hello', 'Changed'))
                execute()
                self.assertEqual(len(commands), 2)
                self.assertIn('Changed', (root/'out/subtitle.srt').read_text())
                options['youtube_subtitle_kind'] = 'auto'
                execute()
                self.assertEqual(len(commands), 3)
                self.assertTrue(any(e.get('stage')=='importing_captions' for e in events))
                self.assertFalse(any(e.get('stage')=='transcribing' for e in events))
                with self.assertRaises(ValueError):
                    worker.execute(video, root/'out', options, events.append)
                self.assertFalse((root/'out/manifest.json').exists())

    def test_vtt_empty_display_cues_are_skipped_but_empty_track_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.vtt'
            source.write_text('WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n \n \n\n00:00:01.000 --> 00:00:02.000\n \nHello\n')
            self.assertEqual(worker.normalize_subtitle(source), SRT)
            source.write_text('WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n \n')
            with self.assertRaises(ValueError):
                worker.normalize_subtitle(source)

    def test_caption_format_and_timeline_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.srt'
            for text in ['', 'not subtitles', SRT.replace('00:00:02,000','00:00:00,000'), SRT.replace('00:00:01,000','00:61:01,000'), SRT + '\n2\n00:00:00,000 --> 00:00:01,000\nBackwards\n']:
                source.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError): worker.normalize_subtitle(source)
            source.write_text(SRT)
            self.assertEqual(worker.normalize_subtitle(source), SRT)
            with self.assertRaises(ValueError): worker.normalize_subtitle('https://example.com/captions.vtt')

    def test_switch_between_whisper_and_import_does_not_share_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root/'video.mp4'
            video.write_bytes(b'video')
            caption = root/'source.srt'
            caption.write_text(SRT.replace('Hello','Imported'))
            processor = VideoSubtitleProcessor()
            processor.client = Mock()
            processor.correct_subtitle_with_llm = Mock(side_effect=lambda text, **kwargs: text)
            commands = []
            def run(argv, cwd=None):
                commands.append(argv)
                if 'whisper' in argv:
                    Path(argv[argv.index('--output_dir')+1], 'audio.srt').write_text(SRT)
                else:
                    Path(argv[-1]).write_bytes(b'audio')
            with patch.object(worker, 'processor_for', return_value=processor), patch.object(worker, 'get_ffmpeg_path', return_value='/usr/bin/ffmpeg'), patch.object(worker, 'run_command', side_effect=run):
                worker.execute(video,root/'out',worker.DEFAULTS,lambda e:None)
                imported=worker.DEFAULTS | dict(subtitle_source='youtube',youtube_subtitle_language='en')
                worker.execute(video,root/'out',imported,lambda e:None,subtitle_input=caption)
                self.assertEqual(sum('whisper' in c for c in commands),1)
                self.assertIn('Imported',(root/'out/subtitle.srt').read_text())
                worker.execute(video,root/'out',worker.DEFAULTS,lambda e:None)
                self.assertEqual(sum('whisper' in c for c in commands),2)
                self.assertIn('Hello',(root/'out/subtitle.srt').read_text())


if __name__ == '__main__':
    unittest.main()

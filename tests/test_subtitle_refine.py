import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from subtitle_refine import VideoSubtitleProcessor, SubtitleOutputError

SRT = '7\n00:00:01,000 --> 00:00:02,000\nHello\n\n8\n00:00:02,000 --> 00:00:03,000\nworld'


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        self.p = VideoSubtitleProcessor()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def test_alignment_accepts_translation(self):
        self.p._validate_subtitle_alignment(SRT, SRT.replace('Hello', '你好'))

    def test_alignment_rejects_missing_added_reordered_or_retimed_cues(self):
        for result in (SRT.split('\n\n')[0], SRT + '\n\n' + SRT,
                       '\n\n'.join(reversed(SRT.split('\n\n'))),
                       SRT.replace('00:00:01,000', '00:00:01,001')):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.p._validate_subtitle_alignment(SRT, result)

    def test_empty_and_invalid_srt_rejected(self):
        for text in ('', 'hello', '1\n00:00:01,000 --> 00:00:02,000\n'):
            with self.assertRaises(ValueError):
                self.p._srt_entries(text)

    def test_merge_preserves_non_one_start(self):
        self.assertEqual(self.p._merge_subtitle_chunks(SRT.split('\n\n')), SRT)

    def test_runtime_request_uses_supported_fields(self):
        p = VideoSubtitleProcessor(provider='local-runtime', openai_model='gpt-6-luna')
        p.openai_client = Mock()
        p.openai_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps([{'id':'7','text':'Hello'},{'id':'8','text':'world'}])))], usage=None)
        self.assertEqual(p._process_single_chunk(SRT, 1, 1, '')[1], SRT)
        self.assertEqual(set(p.openai_client.chat.completions.create.call_args.kwargs), {'model', 'messages'})
        p.openai_client.chat.completions.create.return_value.choices[0].finish_reason = 'content_filter'
        with self.assertRaises(Exception):
            p._process_single_chunk(SRT, 1, 1, '')

    def test_runtime_json_preserves_source_times_and_reorders_by_id(self):
        result = self.p._runtime_text_result(SRT, '[{"id":"8","text":"世界"},{"id":"7","text":"你好"}]')
        self.p._validate_subtitle_alignment(SRT, result)
        self.assertIn("00:00:01,000 --> 00:00:02,000\n你好", result)
        for bad in ('[]', 'not json', '[{"id":"7","text":"x"},{"id":"7","text":"y"}]',
                    '[{"id":"7","text":"x"},{"id":"9","text":"y"}]'):
            with self.assertRaises(SubtitleOutputError):
                self.p._runtime_text_result(SRT, bad)

    def test_runtime_missing_cues_split_and_retry(self):
        p = VideoSubtitleProcessor(provider='local-runtime')
        p.openai_client = Mock()
        def answer(**kwargs):
            prompt = kwargs['messages'][0]['content']
            cues = json.loads(prompt.split('不輸出時間軸或參考前後文。\n')[1])
            content = '[]' if len(cues) > 1 else json.dumps(cues)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=content))], usage=None)
        p.openai_client.chat.completions.create.side_effect = answer
        self.assertEqual(p._process_single_chunk(SRT, 1, 1, '')[1], SRT)
        self.assertEqual(p.openai_client.chat.completions.create.call_count, 3)

    def test_context_windows_and_crlf(self):
        self.p.client = Mock()
        calls = []
        def process(chunk, index, total, prompt, *args):
            calls.append(prompt)
            return index, chunk
        with patch.object(self.p, '_process_single_chunk', side_effect=process), patch('subtitle_refine.time.sleep'):
            self.assertEqual(self.p.correct_subtitle_with_llm(SRT.replace('\n', '\r\n'), max_chars=60), SRT)
        self.assertEqual(len(calls), 2)
        self.assertIn('world', calls[0])
        self.assertIn('Hello', calls[1])

    def test_pipeline_corrects_before_translation_and_skip_only_skips_correction(self):
        for skip in (False, True):
            with self.subTest(skip=skip), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'source.srt'
                source.write_text(SRT)
                video = Path(tmp) / 'video.mp4'
                video.touch()
                self.p.client = Mock()
                with patch.object(self.p, 'correct_subtitle_with_llm', return_value=SRT) as ai, patch.object(self.p, 'embed_subtitle_to_video', return_value='output.mp4'):
                    self.p.process(str(video), is_youtube=False, existing_subtitle=str(source),
                                   skip_correction=skip, target_language='zh-TW')
                self.assertEqual(ai.call_count, 1 if skip else 2)
                self.assertEqual(ai.call_args.kwargs['target_language'], 'zh-TW')
                if not skip:
                    self.assertNotIn('target_language', ai.call_args_list[0].kwargs)
                    self.assertTrue((Path(tmp) / 'source_corrected.srt').exists())


if __name__ == '__main__':
    unittest.main()

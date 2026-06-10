import os
import sys
import subprocess
import argparse
from pathlib import Path
from google import genai
from google.genai import types
from typing import Optional
import asyncio
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(**kwargs): pass

try:
    from openai import OpenAI as _OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 設定 FFmpeg 路徑（使用 static-ffmpeg）
FFMPEG_CMD = "ffmpeg"  # 預設值
FFMPEG_AVAILABLE = False

def get_ffmpeg_path():
    """取得 FFmpeg 路徑"""
    global FFMPEG_CMD, FFMPEG_AVAILABLE

    try:
        from static_ffmpeg import run
        # 取得 static-ffmpeg 提供的 ffmpeg 路徑
        ffmpeg_path, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()
        FFMPEG_CMD = ffmpeg_path
        FFMPEG_AVAILABLE = True
        return ffmpeg_path
    except ImportError:
        # 如果沒有安裝 static-ffmpeg，嘗試使用系統的 ffmpeg
        pass
    except Exception as e:
        # 其他錯誤，記錄但不中斷
        print(f"⚠️  static-ffmpeg 初始化警告: {e}")

    # 嘗試使用系統的 ffmpeg
    try:
        import shutil
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            FFMPEG_CMD = system_ffmpeg
            FFMPEG_AVAILABLE = True
            return system_ffmpeg
    except:
        pass

    return "ffmpeg"  # 降級到預設值

class VideoSubtitleProcessor:
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        whisper_model: str = "base",
        source_language: str = "zh",
        gemini_model: str = "gemini-2.5-flash",
        provider: str = "gemini",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o"
    ):
        """
        初始化影片字幕處理器

        Args:
            gemini_api_key: Gemini API 金鑰（only-embed 模式可以不提供）
            whisper_model: Whisper 模型大小 (tiny, base, small, medium, large)
            source_language: 影片語言代碼 (zh, en, ja, ko, es, fr 等，或 auto 自動偵測)
            gemini_model: Gemini 模型名稱
            provider: AI 提供商 (gemini 或 openai)
            openai_api_key: OpenAI API 金鑰
            openai_model: OpenAI 模型名稱
        """
        self.whisper_model = whisper_model
        self.source_language = source_language
        self.provider = provider
        self.gemini_model = gemini_model
        self.openai_model = openai_model
        self.client = None
        self.openai_client = None

        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("請安裝 openai 套件: pip install openai")
            if openai_api_key:
                self.openai_client = _OpenAI(api_key=openai_api_key)
        else:
            # 只在有 API key 時才初始化 Gemini client
            if gemini_api_key:
                os.environ['GEMINI_API_KEY'] = gemini_api_key
                self.client = genai.Client(api_key=gemini_api_key)
        
    def download_youtube_video(self, url: str, output_dir: str = "./downloads") -> str:
        """
        使用 yt-dlp 下載 YouTube 影片

        Args:
            url: YouTube 影片網址
            output_dir: 輸出目錄

        Returns:
            下載的影片檔案路徑
        """
        print(f"📥 正在下載 YouTube 影片: {url}")

        os.makedirs(output_dir, exist_ok=True)
        output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

        # 下載最高畫質的 mp4 格式（影片+音訊）
        # 注意：需要保持 yt-dlp 為最新版本以避免 YouTube 403 錯誤
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", output_template,
            url
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # 從輸出中找到下載的檔案名稱
            for line in result.stdout.split('\n'):
                if 'Merging formats into' in line or 'has already been downloaded' in line:
                    video_path = line.split('"')[1]
                    print(f"✅ 影片下載完成: {video_path}")
                    return video_path
                    
            # 如果找不到，嘗試在目錄中找最新的檔案
            video_files = list(Path(output_dir).glob("*.mp4"))
            if video_files:
                latest_file = max(video_files, key=lambda p: p.stat().st_mtime)
                print(f"✅ 影片下載完成: {latest_file}")
                return str(latest_file)
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 下載失敗: {e.stderr}")
            raise
            
        raise Exception("無法找到下載的影片檔案")

    def list_youtube_subtitles(self, url: str) -> dict:
        """
        列出 YouTube 影片可用的字幕（CC 字幕和自動生成的字幕）

        Args:
            url: YouTube 影片網址

        Returns:
            字典包含 'manual' 和 'auto' 兩個鍵，每個鍵對應一個語言代碼列表
            例如: {'manual': ['en', 'zh-Hant'], 'auto': ['ja', 'ko']}
        """
        print(f"🔍 正在檢查 YouTube 影片的可用字幕...")

        cmd = ["yt-dlp", "--list-subs", url]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            output = result.stdout

            manual_subs = []
            auto_subs = []

            # 解析輸出
            in_manual_section = False
            in_auto_section = False

            for line in output.split('\n'):
                original_line = line
                line = line.strip()

                # 檢測區段標題（注意 yt-dlp 的實際輸出格式）
                if '[info] Available subtitles for' in original_line:
                    # 手動字幕（CC 字幕）
                    in_manual_section = True
                    in_auto_section = False
                    continue
                elif '[info] Available automatic captions for' in original_line:
                    # 自動生成字幕
                    in_auto_section = True
                    in_manual_section = False
                    continue

                # 跳過標題行（包含 "Language"、"Name"、"Formats" 的行）
                if 'Language' in line and 'Formats' in line:
                    continue

                # 解析語言代碼
                # YouTube 使用的語言代碼格式：en, zh-Hant, zh-Hans, pt-PT, en-US 等
                # 格式：語言代碼 + 空格 + 語言名稱（可選）+ 空格 + 格式列表
                if in_manual_section or in_auto_section:
                    # 匹配語言代碼（支援更多格式）
                    # 例如：zh-Hant, en, pt-PT, zh-Hans-CN
                    match = re.match(r'^([a-zA-Z]{2,3}(?:-[a-zA-Z]{2,4}(?:-[a-zA-Z]{2})?)?)\s+', line)
                    if match:
                        lang_code = match.group(1)
                        if in_manual_section:
                            if lang_code not in manual_subs:  # 避免重複
                                manual_subs.append(lang_code)
                        elif in_auto_section:
                            if lang_code not in auto_subs:  # 避免重複
                                auto_subs.append(lang_code)

            result_dict = {
                'manual': manual_subs,
                'auto': auto_subs
            }

            # 顯示找到的字幕
            if manual_subs or auto_subs:
                print(f"✅ 找到可用字幕:")
                if manual_subs:
                    print(f"   📝 手動字幕 (CC): {', '.join(manual_subs)}")
                if auto_subs:
                    print(f"   🤖 自動生成: {', '.join(auto_subs)}")
            else:
                print(f"ℹ️  未找到可用字幕")

            return result_dict

        except subprocess.CalledProcessError as e:
            print(f"⚠️  無法列出字幕: {e.stderr}")
            return {'manual': [], 'auto': []}

    def download_youtube_subtitle(self, url: str, lang: str, is_auto: bool = False, output_dir: str = "./subtitles") -> str:
        """
        下載 YouTube 影片的字幕

        Args:
            url: YouTube 影片網址
            lang: 字幕語言代碼（例如: 'en', 'zh-TW'）
            is_auto: 是否為自動生成的字幕
            output_dir: 輸出目錄

        Returns:
            下載的字幕檔案路徑
        """
        subtitle_type = "自動生成字幕" if is_auto else "手動字幕 (CC)"
        print(f"📥 正在下載 {subtitle_type} ({lang})...")

        os.makedirs(output_dir, exist_ok=True)
        output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "--skip-download",  # 不下載影片
            "--write-subs" if not is_auto else "--write-auto-subs",  # 下載字幕
            "--sub-lang", lang,
            "--sub-format", "srt",  # 強制使用 SRT 格式
            "--convert-subs", "srt",  # 轉換為 SRT 格式
            "-o", output_template,
            url
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # 尋找下載的字幕檔案
            # yt-dlp 會生成類似 "影片標題.語言代碼.srt" 的檔案
            subtitle_files = list(Path(output_dir).glob(f"*.{lang}.srt"))

            if not subtitle_files:
                # 嘗試其他可能的格式
                subtitle_files = list(Path(output_dir).glob("*.srt"))

            if subtitle_files:
                # 取得最新的字幕檔案
                latest_file = max(subtitle_files, key=lambda p: p.stat().st_mtime)
                subtitle_path = str(latest_file)
                print(f"✅ 字幕下載完成: {subtitle_path}")
                return subtitle_path
            else:
                raise Exception(f"無法找到下載的字幕檔案（語言: {lang}）")

        except subprocess.CalledProcessError as e:
            print(f"❌ 字幕下載失敗: {e.stderr}")
            raise

    def transcribe_video(self, video_path: str, output_dir: str = "./subtitles") -> str:
        """
        使用 Whisper 轉錄影片字幕

        Args:
            video_path: 影片檔案路徑
            output_dir: 字幕輸出目錄

        Returns:
            字幕檔案路徑 (.srt)
        """
        lang_display = "自動偵測" if self.source_language == "auto" else self.source_language
        print(f"🎤 正在使用 Whisper 轉錄字幕 (模型: {self.whisper_model}, 語言: {lang_display})...")

        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        subtitle_path = os.path.join(output_dir, f"{video_name}.srt")

        cmd = [
            "whisper",
            video_path,
            "--model", self.whisper_model,
            "--output_format", "srt",
            "--output_dir", output_dir
        ]

        # 如果不是自動偵測，則指定語言
        if self.source_language != "auto":
            cmd.extend(["--language", self.source_language])

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ 字幕轉錄完成: {subtitle_path}")
            return subtitle_path
        except subprocess.CalledProcessError as e:
            print(f"❌ 轉錄失敗: {e}")
            raise
    
    def extract_audio_from_video(self, video_path: str, output_dir: str = "./audio") -> str:
        """
        從影片中提取音訊

        Args:
            video_path: 影片檔案路徑
            output_dir: 音訊輸出目錄

        Returns:
            音訊檔案路徑 (.mp3)
        """
        print(f"🎵 正在從影片提取音訊...")

        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        audio_path = os.path.join(output_dir, f"{video_name}.mp3")

        cmd = [
            FFMPEG_CMD,
            "-i", video_path,
            "-vn",  # 不處理影片
            "-acodec", "libmp3lame",  # 使用 MP3 編碼
            "-b:a", "128k",  # 音訊位元率
            "-y",  # 覆蓋現有檔案
            audio_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ 音訊提取完成: {audio_path}")
            return audio_path
        except subprocess.CalledProcessError as e:
            print(f"❌ 音訊提取失敗: {e.stderr}")
            raise

    def extract_audio_segment(
        self,
        video_path: str,
        start_time: str,
        end_time: str,
        output_path: str
    ) -> str:
        """
        從影片中提取特定時間範圍的音訊片段

        Args:
            video_path: 影片檔案路徑
            start_time: 開始時間 (SRT 格式: HH:MM:SS,mmm)
            end_time: 結束時間 (SRT 格式: HH:MM:SS,mmm)
            output_path: 輸出音訊檔案路徑

        Returns:
            音訊片段檔案路徑
        """
        # 將 SRT 時間格式轉換為 FFmpeg 格式 (HH:MM:SS.mmm)
        start_ffmpeg = start_time.replace(',', '.')
        end_ffmpeg = end_time.replace(',', '.')

        # 計算持續時間
        from datetime import datetime, timedelta

        def parse_time(time_str):
            """解析 HH:MM:SS.mmm 格式的時間"""
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_parts = parts[2].split('.')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
            return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

        start_td = parse_time(start_ffmpeg)
        end_td = parse_time(end_ffmpeg)
        duration = end_td - start_td

        # 轉換回字串格式
        duration_seconds = duration.total_seconds()

        cmd = [
            FFMPEG_CMD,
            "-i", video_path,
            "-ss", start_ffmpeg,  # 開始時間
            "-t", str(duration_seconds),  # 持續時間（秒）
            "-vn",  # 不處理影片
            "-acodec", "libmp3lame",  # 使用 MP3 編碼
            "-b:a", "128k",  # 音訊位元率
            "-y",  # 覆蓋現有檔案
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ 音訊片段提取失敗 ({start_time} - {end_time}): {e.stderr}")
            raise

    def read_subtitle_file(self, subtitle_path: str) -> str:
        """讀取字幕檔案內容"""
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _parse_srt_chunk_timerange(self, srt_chunk: str) -> tuple:
        """
        解析 SRT 字幕片段，取得開始和結束時間

        Args:
            srt_chunk: SRT 字幕片段內容

        Returns:
            (start_time, end_time): 開始和結束時間（SRT 格式）
        """
        lines = srt_chunk.strip().split('\n')

        # 找到第一個和最後一個時間軸
        first_timestamp = None
        last_timestamp = None

        for line in lines:
            if '-->' in line:
                parts = line.split('-->')
                if len(parts) == 2:
                    if first_timestamp is None:
                        first_timestamp = parts[0].strip()
                    last_timestamp = parts[1].strip()

        return (first_timestamp, last_timestamp)

    def _create_audio_segments_for_chunks(
        self,
        video_path: str,
        subtitle_chunks: list,
        output_dir: str = "./audio/segments"
    ) -> list:
        """
        根據字幕片段創建對應的音訊片段

        Args:
            video_path: 影片檔案路徑
            subtitle_chunks: 字幕片段列表
            output_dir: 音訊片段輸出目錄

        Returns:
            音訊片段路徑列表（與 subtitle_chunks 順序對應）
        """
        print(f"🎵 正在為 {len(subtitle_chunks)} 個字幕片段創建音訊片段...")

        os.makedirs(output_dir, exist_ok=True)
        audio_segments = []

        for i, chunk in enumerate(subtitle_chunks, 1):
            # 解析時間範圍
            start_time, end_time = self._parse_srt_chunk_timerange(chunk)

            if not start_time or not end_time:
                print(f"  ⚠️  第 {i} 段無法解析時間範圍，跳過音訊提取")
                audio_segments.append(None)
                continue

            # 創建音訊片段路徑
            audio_segment_path = os.path.join(output_dir, f"segment_{i:04d}.mp3")

            print(f"  📄 第 {i}/{len(subtitle_chunks)} 段: {start_time} → {end_time}")

            try:
                # 提取音訊片段
                self.extract_audio_segment(
                    video_path,
                    start_time,
                    end_time,
                    audio_segment_path
                )
                audio_segments.append(audio_segment_path)
            except Exception as e:
                print(f"  ⚠️  第 {i} 段音訊提取失敗: {e}")
                audio_segments.append(None)

        successful_count = sum(1 for seg in audio_segments if seg is not None)
        print(f"✅ 成功創建 {successful_count}/{len(subtitle_chunks)} 個音訊片段")

        return audio_segments

    def _split_subtitle_into_chunks(self, subtitle_content: str, max_chars: int = 15000) -> list:
        """
        將字幕分割成多個片段（基於字元數量而非條目數）

        Args:
            subtitle_content: 完整的字幕內容
            max_chars: 每個片段最多包含的字元數（預設 15000）

        Returns:
            字幕片段列表
        """
        # 使用正則表達式分割字幕條目（每個條目包含：序號、時間軸、文字、空行）
        # SRT 格式：序號\n時間軸\n文字\n空行
        entries = re.split(r'\n\n+', subtitle_content.strip())

        chunks = []
        current_chunk = []
        current_chars = 0

        for entry in entries:
            entry_length = len(entry)

            # 如果加入這個條目會超過限制，且當前 chunk 不為空，則開始新的 chunk
            if current_chars + entry_length > max_chars and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_chars = 0

            # 加入當前條目
            current_chunk.append(entry)
            current_chars += entry_length + 2  # +2 for '\n\n'

        # 加入最後一個 chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _merge_subtitle_chunks(self, chunks: list) -> str:
        """
        合併字幕片段並重新編號

        Args:
            chunks: 字幕片段列表

        Returns:
            合併後的完整字幕
        """
        all_entries = []
        for chunk in chunks:
            # 分割每個片段中的條目
            entries = re.split(r'\n\n+', chunk.strip())
            all_entries.extend(entries)

        # 重新編號
        renumbered_entries = []
        counter = 1
        for entry in all_entries:
            lines = entry.strip().split('\n')
            if len(lines) >= 2:  # 至少要有序號和時間軸
                # 替換第一行的序號
                lines[0] = str(counter)
                renumbered_entries.append('\n'.join(lines))
                counter += 1

        return '\n\n'.join(renumbered_entries)

    def _fix_srt_timestamp(self, timestamp: str) -> str:
        """
        修正 SRT 時間戳格式

        Args:
            timestamp: 時間戳字串（例如："0007:43,000" 或 "00:07:43,000"）

        Returns:
            修正後的時間戳（例如："00:07:43,000"）
        """
        # 移除所有空格
        timestamp = timestamp.strip()

        # 正確的格式應該是 HH:MM:SS,mmm
        # 嘗試匹配並修正各種錯誤格式

        # 情況1: 0007:43,000 -> 00:07:43,000
        match = re.match(r'^(\d{2})(\d{2}):(\d{2}),(\d{3})$', timestamp)
        if match:
            return f"{match.group(1)}:{match.group(2)}:{match.group(3)},{match.group(4)}"

        # 情況2: 已經是正確格式 00:07:43,000
        if re.match(r'^\d{2}:\d{2}:\d{2},\d{3}$', timestamp):
            return timestamp

        # 情況3: 缺少前導零，例如 7:43,000 -> 00:07:43,000
        match = re.match(r'^(\d{1,2}):(\d{2}),(\d{3})$', timestamp)
        if match:
            return f"00:{match.group(1).zfill(2)}:{match.group(2)},{match.group(3)}"

        # 如果無法匹配，返回原始值
        return timestamp

    def _validate_and_fix_srt(self, srt_content: str) -> str:
        """
        驗證並修正 SRT 格式

        Args:
            srt_content: SRT 字幕內容

        Returns:
            修正後的 SRT 內容
        """
        lines = srt_content.split('\n')
        fixed_lines = []

        for line in lines:
            # 檢查是否是時間軸行（包含 -->）
            if '-->' in line:
                parts = line.split('-->')
                if len(parts) == 2:
                    start_time = self._fix_srt_timestamp(parts[0])
                    end_time = self._fix_srt_timestamp(parts[1])
                    fixed_lines.append(f"{start_time} --> {end_time}")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _get_rate_limit_for_model(self, model_name: str) -> tuple:
        """
        根據模型名稱返回 rate limit 設定

        Returns:
            (rpm, tpm): requests per minute, tokens per minute
        """
        rate_limits = {
            # Gemini 模型
            "gemini-2.5-flash": (10, 250000),
            "gemini-2.5-pro": (5, 125000),
            "gemini-2.5-flash-lite": (15, 250000),
            "gemini-2.0-flash": (10, 250000),
            # OpenAI 模型
            "gpt-4o": (10, 128000),
            "gpt-4o-mini": (15, 128000),
            "gpt-4.1": (10, 128000),
            "gpt-4.1-mini": (15, 128000),
            "o3": (5, 100000),
            "o4-mini": (10, 128000),
        }
        return rate_limits.get(model_name, (10, 128000))  # 預設使用保守限制

    def _process_single_chunk(
        self,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
        base_prompt_template: str,
        target_language: Optional[str] = None,
        retry_count: int = 0,
        max_retries: int = 2,
        audio_path: Optional[str] = None
    ) -> tuple:
        """
        處理單個字幕片段（支援自動重試和音訊輸入）

        Args:
            audio_path: 音訊片段路徑（可選，用於多模態處理）

        Returns:
            (chunk_index, processed_chunk): 索引和處理後的內容
        """
        if total_chunks > 1:
            retry_suffix = f" (重試 {retry_count}/{max_retries})" if retry_count > 0 else ""
            audio_suffix = " [含音訊]" if audio_path else ""
            print(f"  📄 處理第 {chunk_index}/{total_chunks} 段{retry_suffix}{audio_suffix}...")

        chunk_prompt = base_prompt_template + f"字幕內容：\n\n{chunk}"

        # 顯示發送的資料量
        chunk_chars = len(chunk)
        prompt_chars = len(chunk_prompt)
        print(f"     📊 字幕片段大小: {chunk_chars:,} 字元")
        print(f"     📊 完整提示詞大小: {prompt_chars:,} 字元")
        if audio_path:
            import os
            audio_size = os.path.getsize(audio_path)
            print(f"     🎵 音訊檔案大小: {audio_size:,} bytes ({audio_size / 1024 / 1024:.2f} MB)")

        try:
            result_text = None

            if self.provider == "openai":
                print(f"  ⏳ 正在呼叫 OpenAI API ({self.openai_model})...")
                # 各模型 max output tokens 上限
                _openai_max_tokens = {
                    "gpt-4o": 16384, "gpt-4o-mini": 16384,
                    "gpt-4.1": 32768, "gpt-4.1-mini": 32768,
                    "o3": 100000, "o4-mini": 65536,
                }.get(self.openai_model, 16384)
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": chunk_prompt}],
                    max_tokens=_openai_max_tokens,
                    temperature=0.3,
                )
                finish_reason = response.choices[0].finish_reason
                print(f"  🔍 檢查 API 回應...")
                print(f"     - finish_reason: {finish_reason}")

                if finish_reason == "length":
                    if retry_count < max_retries:
                        print(f"     🔄 將此片段切成兩半後重試...")
                        entries = re.split(r'\n\n+', chunk.strip())
                        mid = len(entries) // 2
                        if mid > 0:
                            chunk1 = '\n\n'.join(entries[:mid])
                            chunk2 = '\n\n'.join(entries[mid:])
                            print(f"     ✂️  分割為兩個子片段: {len(chunk1):,} 和 {len(chunk2):,} 字元")
                            _, processed1 = self._process_single_chunk(
                                chunk1, chunk_index, total_chunks, base_prompt_template,
                                target_language, retry_count + 1, max_retries, None
                            )
                            _, processed2 = self._process_single_chunk(
                                chunk2, chunk_index, total_chunks, base_prompt_template,
                                target_language, retry_count + 1, max_retries, None
                            )
                            return (chunk_index, processed1 + '\n\n' + processed2)
                    error_msg = f"❌ 輸出超過 token 限制！\n"
                    error_msg += f"     - 此片段字元數: {chunk_chars:,}\n"
                    error_msg += f"     - 建議：減少 max_chars 參數（目前可能需要設為 {chunk_chars // 2:,} 以下）\n"
                    raise ValueError(error_msg)

                if response.usage:
                    usage = response.usage
                    print(f"     📈 Token 使用統計:")
                    print(f"        - 輸入 tokens: {usage.prompt_tokens:,}")
                    print(f"        - 輸出 tokens: {usage.completion_tokens:,}")
                    print(f"        - 總計 tokens: {usage.total_tokens:,}")

                result_text = response.choices[0].message.content

            else:
                print(f"  ⏳ 正在呼叫 Gemini API ({self.gemini_model})...")

                # 準備內容
                if audio_path:
                    # 使用多模態模式：上傳音訊並與文字一起發送
                    print(f"     📤 正在上傳音訊檔案...")
                    audio_file = self.client.files.upload(file=audio_path)
                    print(f"     ✅ 音訊已上傳: {audio_file.name}")
                    contents = [audio_file, chunk_prompt]
                else:
                    contents = chunk_prompt

                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=65536,
                        temperature=0.3,
                    )
                )

                print(f"  🔍 檢查 API 回應...")
                if hasattr(response, 'candidates') and response.candidates:
                    finish_reason = response.candidates[0].finish_reason if hasattr(response.candidates[0], 'finish_reason') else None
                    print(f"     - finish_reason: {finish_reason}")

                    if finish_reason and 'MAX_TOKENS' in str(finish_reason):
                        if hasattr(response, 'usage_metadata'):
                            usage = response.usage_metadata
                            print(f"     ⚠️  超過 token 限制詳情:")
                            if hasattr(usage, 'thoughts_token_count'):
                                print(f"        - 思考 tokens: {usage.thoughts_token_count:,}")
                            if hasattr(usage, 'candidates_token_count'):
                                print(f"        - 輸出 tokens: {usage.candidates_token_count:,}")
                            if hasattr(usage, 'total_token_count'):
                                print(f"        - 總計: {usage.total_token_count:,} / 65,536")

                        if retry_count < max_retries:
                            print(f"     🔄 將此片段切成兩半後重試...")
                            entries = re.split(r'\n\n+', chunk.strip())
                            mid = len(entries) // 2
                            if mid > 0:
                                chunk1 = '\n\n'.join(entries[:mid])
                                chunk2 = '\n\n'.join(entries[mid:])
                                print(f"     ✂️  分割為兩個子片段: {len(chunk1):,} 和 {len(chunk2):,} 字元")
                                _, processed1 = self._process_single_chunk(
                                    chunk1, chunk_index, total_chunks, base_prompt_template,
                                    target_language, retry_count + 1, max_retries, None
                                )
                                _, processed2 = self._process_single_chunk(
                                    chunk2, chunk_index, total_chunks, base_prompt_template,
                                    target_language, retry_count + 1, max_retries, None
                                )
                                return (chunk_index, processed1 + '\n\n' + processed2)

                        error_msg = f"❌ 輸出超過 token 限制！\n"
                        error_msg += f"     - 此片段字元數: {chunk_chars:,}\n"
                        error_msg += f"     - 建議：減少 max_chars 參數（目前可能需要設為 {chunk_chars // 2:,} 以下）\n"
                        if hasattr(response, 'usage_metadata'):
                            error_msg += f"     - Token 使用情況: {response.usage_metadata}\n"
                        raise ValueError(error_msg)

                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    print(f"     📈 Token 使用統計:")
                    if hasattr(usage, 'prompt_token_count'):
                        print(f"        - 輸入 tokens: {usage.prompt_token_count:,}")
                    if hasattr(usage, 'candidates_token_count'):
                        print(f"        - 輸出 tokens: {usage.candidates_token_count:,}")
                    if hasattr(usage, 'thoughts_token_count') and usage.thoughts_token_count:
                        print(f"        - 思考 tokens: {usage.thoughts_token_count:,}")
                    if hasattr(usage, 'total_token_count'):
                        print(f"        - 總計 tokens: {usage.total_token_count:,}")
                    if hasattr(usage, 'prompt_token_count') and prompt_chars > 0:
                        char_per_token = prompt_chars / usage.prompt_token_count
                        print(f"        - 字元/Token 比率: {char_per_token:.2f}")

                result_text = response.text

            # 檢查回應是否有效
            if not result_text:
                raise ValueError("API 回應為空")

            print(f"  ✅ 成功接收到回應 (長度: {len(result_text):,} 字元)")

            # 清理回應
            cleaned_chunk = self._clean_llm_response(result_text)

            # 驗證並修正 SRT 格式
            cleaned_chunk = self._validate_and_fix_srt(cleaned_chunk)

            if total_chunks > 1:
                print(f"  ✔️  第 {chunk_index}/{total_chunks} 段處理完成")

            return (chunk_index, cleaned_chunk)

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # 檢查是否為可重試的錯誤（503, 429, 500 等）
            is_retryable = False
            if 'ServerError' in error_type or '503' in error_msg or 'UNAVAILABLE' in error_msg:
                is_retryable = True
            elif '429' in error_msg or 'RESOURCE_EXHAUSTED' in error_msg or 'RateLimitError' in error_type:
                is_retryable = True
            elif '500' in error_msg or 'INTERNAL' in error_msg or 'APIStatusError' in error_type:
                is_retryable = True

            # 如果是可重試的錯誤且還有重試次數
            if is_retryable and retry_count < max_retries:
                wait_time = (2 ** retry_count) * 5  # 指數退避：5秒, 10秒, 20秒...
                print(f"  ⚠️  第 {chunk_index}/{total_chunks} 段遇到暫時性錯誤")
                print(f"     錯誤類型: {error_type}")
                print(f"     錯誤訊息: {error_msg}")
                print(f"     🔄 將在 {wait_time} 秒後重試 ({retry_count + 1}/{max_retries})...")

                time.sleep(wait_time)

                # 遞迴重試
                return self._process_single_chunk(
                    chunk, chunk_index, total_chunks, base_prompt_template,
                    target_language, retry_count + 1, max_retries, audio_path
                )

            # 不可重試或已達最大重試次數
            print(f"  ❌ 第 {chunk_index}/{total_chunks} 段處理失敗")
            print(f"     錯誤類型: {error_type}")
            print(f"     錯誤訊息: {error_msg}")
            if is_retryable:
                print(f"     ⚠️  已達最大重試次數 ({max_retries})")
            print(f"\n  🛑 由於處理失敗，中斷整個{'翻譯' if target_language else '校正'}流程")
            raise Exception(f"第 {chunk_index}/{total_chunks} 段處理失敗: {error_msg}")

    def correct_subtitle_with_llm(
        self,
        subtitle_content: str,
        custom_prompt: Optional[str] = None,
        context: Optional[str] = None,
        target_language: Optional[str] = None,
        max_chars: int = 15000,
        use_audio_context: bool = False,
        video_path: Optional[str] = None
    ) -> str:
        """
        使用 Gemini LLM 校正字幕（並可選擇翻譯）

        Args:
            subtitle_content: 原始字幕內容
            custom_prompt: 自定義提示詞
            context: 額外的上下文資訊
            target_language: 目標翻譯語言（如果需要翻譯）
            max_chars: 每個片段的最大字元數（預設 15000，平衡處理速度與 token 限制）
            use_audio_context: 是否使用音訊上下文進行多模態處理
            video_path: 影片檔案路徑（當 use_audio_context=True 時必須提供）

        Returns:
            校正後（或翻譯後）的字幕內容
        """
        if self.provider == "openai":
            model_label = self.openai_model
            provider_label = "OpenAI"
        else:
            model_label = self.gemini_model
            provider_label = "Gemini AI"

        if use_audio_context:
            if target_language:
                print(f"🤖 正在使用 {provider_label} ({model_label}) 以多模態方式（音訊+文字）校正並翻譯字幕為 {target_language}...")
            else:
                print(f"🤖 正在使用 {provider_label} ({model_label}) 以多模態方式（音訊+文字）校正字幕...")
        else:
            if target_language:
                print(f"🤖 正在使用 {provider_label} ({model_label}) 校正並翻譯字幕為 {target_language}...")
            else:
                print(f"🤖 正在使用 {provider_label} ({model_label}) 校正字幕...")

        # 檢查是否有可用的 client
        if self.provider == "openai":
            if not self.openai_client:
                raise ValueError("需要 OpenAI API key 才能使用 AI 校正/翻譯功能")
            if use_audio_context:
                raise ValueError("--use-audio-context 僅支援 Gemini，不支援 OpenAI")
        else:
            if not self.client:
                raise ValueError("需要 Gemini API key 才能使用 AI 校正/翻譯功能")

        # 檢查音訊上下文參數
        if use_audio_context and not video_path:
            raise ValueError("使用音訊上下文時必須提供 video_path 參數")

        # 分割字幕為多個片段（基於字元數）
        chunks = self._split_subtitle_into_chunks(subtitle_content, max_chars=max_chars)
        total_chunks = len(chunks)

        if total_chunks > 1:
            print(f"📝 字幕較長，將分成 {total_chunks} 段處理（每段最多 {max_chars:,} 字元）...")
            # 顯示每個 chunk 的實際大小
            for i, chunk in enumerate(chunks, 1):
                print(f"   - 第 {i} 段: {len(chunk):,} 字元")

        processed_chunks = []

        # 語言對照表
        language_names = {
            "zh": "繁體中文",
            "zh-TW": "繁體中文",
            "zh-CN": "簡體中文",
            "en": "英文",
            "ja": "日文",
            "ko": "韓文",
            "es": "西班牙文",
            "fr": "法文",
            "de": "德文",
            "it": "義大利文",
            "pt": "葡萄牙文",
            "ru": "俄文",
            "ar": "阿拉伯文",
            "th": "泰文",
            "vi": "越南文"
        }

        # 準備基礎提示詞
        if target_language:
            # 翻譯模式
            target_lang_name = language_names.get(target_language, target_language)

            # 判斷目標語言是否為中文
            is_chinese_target = target_language in ["zh", "zh-TW", "zh-CN"]

            base_prompt_template = f"""請將以下 SRT 格式的字幕檔案翻譯成{target_lang_name}。

重要規則：
1. 必須保持完整的 SRT 格式（序號、時間軸、字幕文字）
2. 不要改變字幕的時間軸
3. 翻譯要自然流暢，符合目標語言的表達習慣
4. 專有名詞保持原文或使用通用譯名
5. 根據上下文準確翻譯
6. 必須處理完所有字幕條目，不要截斷
7. 直接輸出 SRT 格式內容，不要加任何說明文字或前言
"""

            # 如果目標語言是中文，加入標點符號規範
            if is_chinese_target:
                base_prompt_template += """8. 【重要】中文字幕標點符號規範：
   - 句尾通常不加標點符號（句號、逗號等）
   - 只有在疑問句時才加問號（？）
   - 只有在感嘆句時才加驚嘆號（！）
   - 句中可以使用逗號（，）來分隔語意
   - 例如：「你好嗎」而不是「你好嗎。」
   - 例如：「這是什麼？」（疑問句保留問號）
   - 例如：「太棒了！」（感嘆句保留驚嘆號）

"""
            else:
                base_prompt_template += "\n"

        else:
            # 校正模式
            base_prompt_template = """請校正以下 SRT 格式的字幕檔案。

重要規則：
1. 必須保持完整的 SRT 格式（序號、時間軸、字幕文字）
2. 不要改變字幕的時間軸
3. 修正錯別字、標點符號錯誤
4. 根據上下文調整斷句
5. 修正語音辨識錯誤（如同音字）
6. 保持原意，不要過度潤飾
7. 必須處理完所有字幕條目，不要截斷
8. 直接輸出 SRT 格式內容，不要加任何說明文字或前言
9. 【重要】中文字幕標點符號規範：
   - 句尾通常不加標點符號（句號、逗號等）
   - 只有在疑問句時才加問號（？）
   - 只有在感嘆句時才加驚嘆號（！）
   - 句中可以使用逗號（，）來分隔語意
   - 例如：「你好嗎」而不是「你好嗎。」
   - 例如：「這是什麼？」（疑問句保留問號）
   - 例如：「太棒了！」（感嘆句保留驚嘆號）

"""

        if context:
            base_prompt_template += f"額外的上下文資訊：{context}\n\n"

        if custom_prompt:
            base_prompt_template += f"{custom_prompt}\n\n"

        # 如果使用音訊上下文，添加相應的提示詞說明
        if use_audio_context:
            base_prompt_template = "你將收到音訊檔案和對應的字幕文字。請仔細聆聽音訊，並根據實際發音來校正字幕中的錯誤。\n\n" + base_prompt_template

        # 創建音訊片段（如果需要）
        audio_segments = []
        if use_audio_context:
            audio_segments = self._create_audio_segments_for_chunks(
                video_path,
                chunks
            )

        # 取得 rate limit（依照 provider 選用對應模型）
        active_model = self.openai_model if self.provider == "openai" else self.gemini_model
        rpm, tpm = self._get_rate_limit_for_model(active_model)

        # 計算每個請求之間需要等待的時間（秒）
        delay_between_requests = 60.0 / rpm if total_chunks > 1 else 0

        if total_chunks > 1:
            print(f"⚡ 使用平行處理加速（考慮 API rate limit: {rpm} RPM）")
            if delay_between_requests > 0:
                print(f"   每個請求間隔: {delay_between_requests:.1f} 秒")

        # 使用 ThreadPoolExecutor 平行處理，但限制同時執行的數量
        # 音訊上下文模式時降低併發數，因為需要上傳檔案
        if use_audio_context:
            max_workers = min(total_chunks, max(1, rpm // 2))  # 降低一半併發數
        else:
            max_workers = min(total_chunks, rpm)  # 不超過 RPM 限制
        processed_chunks_dict = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務，並加入延遲避免超過 rate limit
            futures = {}
            for i, chunk in enumerate(chunks, 1):
                # 延遲提交以符合 rate limit
                if i > 1 and delay_between_requests > 0:
                    time.sleep(delay_between_requests)

                # 取得對應的音訊片段（如果有）
                audio_path = None
                if use_audio_context and audio_segments and i <= len(audio_segments):
                    audio_path = audio_segments[i - 1]  # audio_segments 是 0-indexed

                future = executor.submit(
                    self._process_single_chunk,
                    chunk,
                    i,
                    total_chunks,
                    base_prompt_template,
                    target_language,
                    0,  # retry_count
                    2,  # max_retries
                    audio_path  # 音訊路徑
                )
                futures[future] = i

            # 收集結果
            for future in as_completed(futures):
                chunk_index, processed_chunk = future.result()
                processed_chunks_dict[chunk_index] = processed_chunk

        # 按照原始順序排列處理後的片段
        processed_chunks = [processed_chunks_dict[i] for i in sorted(processed_chunks_dict.keys())]

        # 合併所有處理過的片段
        if total_chunks > 1:
            print(f"🔗 正在合併 {total_chunks} 個片段並重新編號...")
            final_subtitle = self._merge_subtitle_chunks(processed_chunks)
        else:
            final_subtitle = processed_chunks[0] if processed_chunks else subtitle_content

        print(f"✅ 字幕{'翻譯' if target_language else '校正'}完成")
        return final_subtitle
    
    def _clean_llm_response(self, text: str) -> str:
        """
        清理 LLM 回應，移除可能的說明文字和格式標記

        Args:
            text: LLM 的原始回應

        Returns:
            清理後的 SRT 內容
        """
        import re

        # 確保 text 不是 None
        if not text:
            return ""

        # 移除開頭可能的說明文字（例如："以下是翻譯後的內容："）
        lines = text.strip().split('\n')
        
        # 找到第一個 SRT 序號（純數字）的位置
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().isdigit():
                start_idx = i
                break
        
        # 從第一個序號開始取內容
        cleaned_lines = lines[start_idx:]
        
        # 移除可能的 markdown 代碼塊標記
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'^```srt\n', '', result)
        result = re.sub(r'^```\n', '', result)
        result = re.sub(r'\n```$', '', result)

        return result.strip()

    def save_corrected_subtitle(self, content: str, original_path: str, target_language: Optional[str] = None) -> str:
        """
        儲存校正後的字幕
        
        Args:
            content: 校正後的字幕內容
            original_path: 原始字幕路徑
            target_language: 如果有翻譯，標註目標語言
            
        Returns:
            校正後字幕的檔案路徑
        """
        if target_language:
            corrected_path = original_path.replace('.srt', f'_{target_language}.srt')
        else:
            corrected_path = original_path.replace('.srt', '_corrected.srt')
        
        with open(corrected_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"💾 {'翻譯' if target_language else '校正'}後的字幕已儲存: {corrected_path}")
        return corrected_path
    
    def embed_subtitle_to_video(
        self, 
        video_path: str, 
        subtitle_path: str, 
        output_dir: str = "./output"
    ) -> str:
        """
        使用 FFmpeg 將字幕嵌入影片
        
        Args:
            video_path: 影片檔案路徑
            subtitle_path: 字幕檔案路徑
            output_dir: 輸出目錄
            
        Returns:
            帶字幕的影片檔案路徑
        """
        print("🎬 正在將字幕嵌入影片...")
        
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}_with_subtitles.mp4")
        
        # 將路徑轉換為絕對路徑
        subtitle_path_abs = os.path.abspath(subtitle_path)
        video_path_abs = os.path.abspath(video_path)
        
        import shutil
        import tempfile
        
        # 創建一個臨時工作目錄
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 方法：使用 ASS 格式（更可靠）+ 複製到簡單路徑
            temp_video = os.path.join(temp_dir, "input.mp4")
            temp_subtitle_srt = os.path.join(temp_dir, "subtitle.srt")
            temp_subtitle_ass = os.path.join(temp_dir, "subtitle.ass")
            
            # 複製影片和字幕到臨時目錄
            print("📋 複製檔案到臨時目錄...")
            shutil.copy2(video_path_abs, temp_video)
            shutil.copy2(subtitle_path_abs, temp_subtitle_srt)
            
            # 先將 SRT 轉換為 ASS 格式（更可靠）
            print("🔄 轉換字幕格式...")
            convert_cmd = [
                FFMPEG_CMD,
                "-i", temp_subtitle_srt,
                "-y",
                temp_subtitle_ass
            ]
            
            try:
                subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"⚠️  字幕轉換失敗，使用原始 SRT: {e.stderr}")
                # 如果轉換失敗，直接使用 SRT
                temp_subtitle_ass = temp_subtitle_srt
            
            # 使用 ass filter（比 subtitles filter 更穩定）
            print("🎨 嵌入字幕到影片...")
            
            if temp_subtitle_ass.endswith('.ass'):
                # 使用 ass filter
                cmd = [
                    FFMPEG_CMD,
                    "-i", temp_video,
                    "-vf", f"ass={temp_subtitle_ass}",
                    "-c:a", "copy",
                    "-y",
                    output_path
                ]
            else:
                # 使用 subtitles filter
                cmd = [
                    FFMPEG_CMD,
                    "-i", temp_video,
                    "-vf", f"subtitles={temp_subtitle_srt}",
                    "-c:a", "copy",
                    "-y",
                    output_path
                ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            print(f"✅ 影片處理完成: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 影片處理失敗")
            print(f"錯誤訊息: {e.stderr}")
            
            # 如果上面的方法都失敗，嘗試最後一招：使用軟字幕（不燒錄）
            print("⚠️  嘗試使用軟字幕方式（字幕不會燒錄到影片中，但會作為獨立軌道）...")
            
            try:
                soft_sub_cmd = [
                    FFMPEG_CMD,
                    "-i", video_path_abs,
                    "-i", subtitle_path_abs,
                    "-c", "copy",
                    "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=chi",
                    "-y",
                    output_path
                ]
                
                subprocess.run(soft_sub_cmd, check=True, capture_output=True, text=True)
                print(f"✅ 影片處理完成（使用軟字幕）: {output_path}")
                print("ℹ️  注意：字幕以獨立軌道形式存在，需要播放器支援才能顯示")
                return output_path
                
            except subprocess.CalledProcessError as e2:
                print(f"❌ 所有方法都失敗")
                print(f"軟字幕錯誤: {e2.stderr}")
                raise
                
        finally:
            # 清理臨時目錄
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def process(
        self,
        input_source: str,
        is_youtube: bool = False,
        custom_prompt: Optional[str] = None,
        context: Optional[str] = None,
        skip_correction: bool = False,
        target_language: Optional[str] = None,
        existing_subtitle: Optional[str] = None,
        skip_download: bool = False,
        skip_transcribe: bool = False,
        only_embed: bool = False,
        use_audio_context: bool = False
    ) -> str:
        """
        完整處理流程

        Args:
            input_source: YouTube URL 或本地影片路徑
            is_youtube: 是否為 YouTube 連結
            custom_prompt: 自定義 AI 校正提示詞
            context: 額外的上下文資訊
            skip_correction: 是否跳過 AI 校正
            target_language: 目標翻譯語言（如果需要翻譯）
            existing_subtitle: 現有字幕檔案路徑（如果提供則跳過轉錄）
            skip_download: 跳過下載步驟（僅當提供本地影片路徑時）
            skip_transcribe: 跳過轉錄步驟（需要提供 existing_subtitle）
            only_embed: 只執行字幕嵌入（需要提供影片和字幕，跳過所有其他步驟）
            use_audio_context: 是否使用音訊上下文進行多模態處理

        Returns:
            最終帶字幕的影片路徑
        """
        print("=" * 60)
        print("🚀 開始處理影片字幕")
        print("=" * 60)

        # 步驟 0: 如果是 YouTube 影片且沒有提供現有字幕，檢查 CC 字幕
        use_cc_subtitle = False
        cc_subtitle_path = None
        skip_correction_for_cc = False  # 用於記錄使用者對 CC 字幕的校正選擇

        if is_youtube and not existing_subtitle and not only_embed:
            available_subs = self.list_youtube_subtitles(input_source)

            # 如果有可用的字幕，詢問使用者
            if available_subs['manual'] or available_subs['auto']:
                print("\n" + "=" * 60)
                print("💡 發現 YouTube 提供的字幕！")
                print("=" * 60)

                # 合併手動和自動字幕列表
                all_subs = []
                if available_subs['manual']:
                    for lang in available_subs['manual']:
                        all_subs.append((lang, False, '手動 (CC)'))
                if available_subs['auto']:
                    for lang in available_subs['auto']:
                        all_subs.append((lang, True, '自動生成'))

                # 顯示選項
                print("\n可用的字幕選項：")
                print("  [0] 不使用 YouTube 字幕，使用 Whisper 轉錄")
                for idx, (lang, is_auto, subtitle_type) in enumerate(all_subs, 1):
                    print(f"  [{idx}] 使用 {lang} 字幕 ({subtitle_type})")

                # 讀取使用者選擇
                while True:
                    try:
                        choice = input("\n請選擇 [0-{}]: ".format(len(all_subs)))
                        choice_num = int(choice)
                        if 0 <= choice_num <= len(all_subs):
                            break
                        else:
                            print(f"⚠️  請輸入 0 到 {len(all_subs)} 之間的數字")
                    except ValueError:
                        print("⚠️  請輸入有效的數字")
                    except KeyboardInterrupt:
                        print("\n\n❌ 使用者取消操作")
                        sys.exit(0)

                if choice_num > 0:
                    # 使用者選擇使用 CC 字幕
                    selected_lang, selected_is_auto, selected_type = all_subs[choice_num - 1]
                    print(f"\n✅ 將使用 {selected_lang} 字幕 ({selected_type})")

                    # 下載選擇的字幕
                    try:
                        cc_subtitle_path = self.download_youtube_subtitle(
                            input_source,
                            selected_lang,
                            selected_is_auto
                        )
                        use_cc_subtitle = True
                        print("✅ 已下載 YouTube 字幕，將跳過 Whisper 轉錄步驟")

                        # 詢問是否需要 AI 處理
                        _ai_label = "OpenAI" if self.provider == "openai" else "Gemini AI"
                        print("\n" + "-" * 60)
                        print(f"🤔 是否需要使用 {_ai_label} 處理這個字幕？")
                        print("-" * 60)

                        if target_language:
                            print(f"✨ 您設定了翻譯目標語言：{target_language}")
                            print("\n選項：")
                            print(f"  [1] 使用 {_ai_label} 校正 + 翻譯（推薦，確保翻譯品質）")
                            print(f"  [2] 僅使用 {_ai_label} 翻譯（跳過校正）")
                            print("  [3] 都不使用（直接使用原始字幕，不翻譯）")

                            while True:
                                try:
                                    ai_choice = input("\n請選擇 [1-3]: ")
                                    ai_choice_num = int(ai_choice)
                                    if 1 <= ai_choice_num <= 3:
                                        break
                                    else:
                                        print("⚠️  請輸入 1 到 3 之間的數字")
                                except ValueError:
                                    print("⚠️  請輸入有效的數字")
                                except KeyboardInterrupt:
                                    print("\n\n❌ 使用者取消操作")
                                    sys.exit(0)

                            if ai_choice_num == 1:
                                print(f"\n✅ 將使用 {_ai_label} 校正 + 翻譯")
                                skip_correction_for_cc = False
                            elif ai_choice_num == 2:
                                print(f"\n✅ 將僅使用 {_ai_label} 翻譯")
                                skip_correction_for_cc = False
                                # 這裡保持 skip_correction_for_cc = False，但我們會在後面只做翻譯
                            else:
                                print("\n✅ 將直接使用原始字幕（不校正、不翻譯）")
                                skip_correction_for_cc = True
                        else:
                            print("\n選項：")
                            print(f"  [1] 使用 {_ai_label} 校正字幕（修正錯誤、優化斷句）")
                            print("  [2] 直接使用原始字幕（不校正）")

                            while True:
                                try:
                                    ai_choice = input("\n請選擇 [1-2]: ")
                                    ai_choice_num = int(ai_choice)
                                    if 1 <= ai_choice_num <= 2:
                                        break
                                    else:
                                        print("⚠️  請輸入 1 到 2 之間的數字")
                                except ValueError:
                                    print("⚠️  請輸入有效的數字")
                                except KeyboardInterrupt:
                                    print("\n\n❌ 使用者取消操作")
                                    sys.exit(0)

                            if ai_choice_num == 1:
                                print(f"\n✅ 將使用 {_ai_label} 校正字幕")
                                skip_correction_for_cc = False
                            else:
                                print("\n✅ 將直接使用原始字幕")
                                skip_correction_for_cc = True

                        print("-" * 60)

                    except Exception as e:
                        print(f"❌ 下載字幕失敗: {e}")
                        print("⚠️  將改用 Whisper 轉錄")
                        use_cc_subtitle = False
                else:
                    print("\n✅ 將使用 Whisper 轉錄")

                print("=" * 60 + "\n")

        # 步驟 1: 取得影片
        if skip_download and not is_youtube:
            video_path = input_source
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"找不到影片檔案: {video_path}")
            print(f"⏭️  跳過下載，使用現有影片: {video_path}")
        elif is_youtube:
            video_path = self.download_youtube_video(input_source)
        else:
            video_path = input_source
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"找不到影片檔案: {video_path}")
        
        # 如果是純嵌入模式
        if only_embed:
            if not existing_subtitle:
                raise ValueError("純嵌入模式需要提供 --subtitle 參數")
            subtitle_path = existing_subtitle
            if not os.path.exists(subtitle_path):
                raise FileNotFoundError(f"找不到字幕檔案: {subtitle_path}")
            print(f"⏭️  僅執行字幕嵌入，使用字幕: {subtitle_path}")
            output_video_path = self.embed_subtitle_to_video(video_path, subtitle_path)
            print("=" * 60)
            print("🎉 字幕嵌入完成！")
            print(f"📹 最終影片: {output_video_path}")
            print("=" * 60)
            return output_video_path
        
        # 步驟 2: 轉錄字幕
        if use_cc_subtitle:
            # 使用下載的 CC 字幕
            subtitle_path = cc_subtitle_path
            print(f"⏭️  使用 YouTube CC 字幕: {subtitle_path}")
        elif existing_subtitle:
            subtitle_path = existing_subtitle
            if not os.path.exists(subtitle_path):
                raise FileNotFoundError(f"找不到字幕檔案: {subtitle_path}")
            print(f"⏭️  跳過轉錄，使用現有字幕: {subtitle_path}")
        elif skip_transcribe:
            raise ValueError("如果要跳過轉錄，必須提供 existing_subtitle 參數")
        else:
            subtitle_path = self.transcribe_video(video_path)
        
        # 步驟 3: AI 校正/翻譯字幕（可選）
        # 合併原本的 skip_correction 參數和 CC 字幕的使用者選擇
        should_skip_correction = skip_correction or skip_correction_for_cc

        if not should_skip_correction:
            # 在實際需要使用 AI 時，檢查對應 provider 的 client 是否存在
            _has_client = (self.provider == "openai" and self.openai_client) or (self.provider != "openai" and self.client)
            if not _has_client:
                _key_hint = "OPENAI_API_KEY / --openai-api-key" if self.provider == "openai" else "GEMINI_API_KEY / --api-key"
                raise ValueError(
                    f"需要 {self.provider.upper()} API key 才能使用 AI 校正/翻譯功能。\n"
                    f"請設定環境變數 {_key_hint}，或使用 --skip-correction 跳過 AI 處理。"
                )

            subtitle_content = self.read_subtitle_file(subtitle_path)
            corrected_content = self.correct_subtitle_with_llm(
                subtitle_content,
                custom_prompt=custom_prompt,
                context=context,
                target_language=target_language,
                use_audio_context=use_audio_context,
                video_path=video_path if use_audio_context else None
            )
            subtitle_path = self.save_corrected_subtitle(
                corrected_content,
                subtitle_path,
                target_language=target_language
            )
        else:
            if skip_correction_for_cc:
                print("⏭️  根據使用者選擇，跳過 AI 字幕處理")
            else:
                print("⏭️  跳過 AI 字幕校正")
        
        # 步驟 4: 嵌入字幕到影片
        output_video_path = self.embed_subtitle_to_video(video_path, subtitle_path)
        
        print("=" * 60)
        print("🎉 所有處理完成！")
        print(f"📹 最終影片: {output_video_path}")
        print("=" * 60)
        
        return output_video_path


def main():
    # 載入 .env 環境變數（CLI 參數優先，不覆寫已有的 env）
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="自動化影片字幕處理工具 - 支援 YouTube 下載、Whisper 轉錄、AI 校正/翻譯、FFmpeg 嵌入字幕",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 完整流程：處理 YouTube 影片（API key 從 .env 讀取）
  python subtitle_refine.py --youtube "https://www.youtube.com/watch?v=xxxxx"

  # 完整流程：處理本地英文影片並翻譯成繁體中文
  python subtitle_refine.py --video "./video.mp4" --api-key "GEMINI_KEY" --source-lang en --translate zh-TW

  # 使用 OpenAI 處理
  python subtitle_refine.py --video "./video.mp4" --provider openai --openai-api-key "sk-..." --translate zh-TW

  # 使用音訊上下文進行多模態處理（僅支援 Gemini）
  python subtitle_refine.py --video "./video.mp4" --use-audio-context --translate zh-TW

  # 跳過下載和轉錄：僅校正/翻譯現有字幕並嵌入影片
  python subtitle_refine.py --video "./video.mp4" --subtitle "./video.srt" --translate en

  # 純字幕嵌入：直接將已處理好的字幕嵌入影片（不使用 AI）
  python subtitle_refine.py --video "./video.mp4" --subtitle "./video_translated.srt" --only-embed

  # 使用自定義 Gemini 模型
  python subtitle_refine.py --video "./video.mp4" --gemini-model "gemini-2.5-pro"

  # 使用自定義 OpenAI 模型
  python subtitle_refine.py --video "./video.mp4" --provider openai --openai-model "gpt-4o-mini"

環境變數（可寫入 .env 檔案，避免每次傳入 API key）:
  GEMINI_API_KEY=your_gemini_api_key
  OPENAI_API_KEY=your_openai_api_key

支援的語言代碼:
  zh/zh-TW (繁體中文), zh-CN (簡體中文), en (英文), ja (日文), ko (韓文)
  es (西班牙文), fr (法文), de (德文), it (義大利文), pt (葡萄牙文)
  ru (俄文), ar (阿拉伯文), th (泰文), vi (越南文), auto (自動偵測)

可用的 Gemini 模型:
  gemini-2.5-flash (推薦，性價比最高)
  gemini-2.5-flash-lite (最快，成本最低)
  gemini-2.5-pro (最強大，思考能力最好)
  gemini-2.0-flash (第二代，長上下文)

可用的 OpenAI 模型:
  gpt-4o (推薦，性價比最高)
  gpt-4o-mini (最快，成本最低)
  gpt-4.1 (最新旗艦)
  o3 / o4-mini (推理模型)
        """
    )

    # 輸入來源
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--youtube", "-y", help="YouTube 影片網址")
    input_group.add_argument("--video", "-v", help="本地影片檔案路徑")

    # AI Provider 設定
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="AI 提供商 (預設: gemini)"
    )

    # Gemini API 設定
    parser.add_argument("--api-key", "-k", help="Gemini API 金鑰（也可設定環境變數 GEMINI_API_KEY）")
    parser.add_argument(
        "--gemini-model", "-gm",
        default="gemini-2.5-flash",
        help="Gemini 模型名稱 (預設: gemini-2.5-flash)"
    )

    # OpenAI API 設定
    parser.add_argument("--openai-api-key", "-oak", help="OpenAI API 金鑰（也可設定環境變數 OPENAI_API_KEY）")
    parser.add_argument(
        "--openai-model", "-om",
        default="gpt-4o",
        help="OpenAI 模型名稱 (預設: gpt-4o)"
    )

    # 語言設定
    parser.add_argument(
        "--source-lang", "-sl",
        default="zh",
        help="影片的原始語言 (預設: zh 中文，可用 auto 自動偵測)"
    )

    parser.add_argument(
        "--translate", "-t",
        help="翻譯成目標語言 (例如: en, ja, ko, zh-TW 等)"
    )

    # Whisper 設定
    parser.add_argument(
        "--whisper-model", "-w",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 模型大小 (預設: base)"
    )

    # AI 校正設定
    parser.add_argument("--prompt", "-p", help="自定義 AI 校正提示詞")
    parser.add_argument("--context", "-c", help="額外的上下文資訊")
    parser.add_argument("--skip-correction", "-s", action="store_true", help="跳過 AI 字幕校正")
    parser.add_argument(
        "--use-audio-context", "-uac",
        action="store_true",
        help="使用音訊上下文進行多模態處理（僅支援 Gemini，會同時分析音訊和字幕，提供更準確的校正）"
    )

    # 跳過步驟的選項
    parser.add_argument(
        "--subtitle", "-sub",
        help="現有字幕檔案路徑（提供此選項將跳過下載和轉錄步驟）"
    )

    parser.add_argument(
        "--only-embed", "-oe",
        action="store_true",
        help="僅執行字幕嵌入（需要同時提供 --video 和 --subtitle，跳過所有其他步驟包括 AI 校正）"
    )

    args = parser.parse_args()

    # 解析 effective API keys（CLI 參數 > 環境變數）
    effective_gemini_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    effective_openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")

    # 檢查 only-embed 模式的必要參數
    if args.only_embed:
        if not args.video or not args.subtitle:
            parser.error("--only-embed mode requires both --video and --subtitle")
        if args.youtube:
            parser.error("--only-embed mode cannot be used with --youtube")

    # 檢查 use-audio-context 的參數衝突
    if args.use_audio_context:
        if args.only_embed:
            parser.error("--use-audio-context cannot be used with --only-embed (no AI processing in only-embed mode)")
        if args.skip_correction:
            parser.error("--use-audio-context cannot be used with --skip-correction (audio context is only used for AI correction)")
        if args.provider == "openai":
            parser.error("--use-audio-context 僅支援 Gemini，請移除 --provider openai 或移除 --use-audio-context")

    # 檢查 API key（僅在需要 AI 處理時）
    # 注意：對於 YouTube 來源，可能會使用 CC 字幕並在互動選擇中跳過 AI，所以延後檢查
    if not args.only_embed and not args.skip_correction:
        if args.provider == "openai":
            if not effective_openai_key and not args.youtube:
                parser.error("使用 OpenAI 時需要 --openai-api-key 或在 .env 中設定 OPENAI_API_KEY")
        else:
            if not effective_gemini_key and not args.youtube:
                parser.error("使用 Gemini 時需要 --api-key 或在 .env 中設定 GEMINI_API_KEY")

    # 檢查必要工具
    missing_tools = []

    # 檢查並初始化 ffmpeg
    print("🔍 檢查 FFmpeg...")
    ffmpeg_path = get_ffmpeg_path()

    try:
        result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True, timeout=5)
        print(f"✅ FFmpeg 已就緒: {ffmpeg_path}")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"❌ FFmpeg 檢查失敗: {e}")
        missing_tools.append("ffmpeg")

    # 檢查 yt-dlp（僅在需要下載 YouTube 影片時）
    if args.youtube and not args.only_embed:
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            missing_tools.append("yt-dlp")

    # 檢查 whisper（僅在需要轉錄時）
    if not args.subtitle and not args.only_embed:
        try:
            import whisper
        except ImportError:
            missing_tools.append("whisper")

    if missing_tools:
        print("❌ 缺少必要工具，請先安裝:")
        for tool in missing_tools:
            if tool == "yt-dlp":
                print(f"  pip install yt-dlp")
            elif tool == "whisper":
                print(f"  pip install openai-whisper")
            elif tool == "ffmpeg":
                print(f"  pip install static-ffmpeg")
                print(f"  # 或者手動安裝: brew install ffmpeg (macOS)")
        sys.exit(1)

    # 初始化處理器
    processor = VideoSubtitleProcessor(
        gemini_api_key=effective_gemini_key,
        whisper_model=args.whisper_model,
        source_language=args.source_lang,
        gemini_model=args.gemini_model,
        provider=args.provider,
        openai_api_key=effective_openai_key,
        openai_model=args.openai_model
    )
    
    # 開始處理
    try:
        input_source = args.youtube if args.youtube else args.video
        is_youtube = args.youtube is not None
        
        processor.process(
            input_source=input_source,
            is_youtube=is_youtube,
            custom_prompt=args.prompt,
            context=args.context,
            skip_correction=args.skip_correction,
            target_language=args.translate,
            existing_subtitle=args.subtitle,
            skip_download=args.subtitle is not None and not is_youtube,
            skip_transcribe=args.subtitle is not None,
            only_embed=args.only_embed,
            use_audio_context=args.use_audio_context
        )
        
    except Exception as e:
        print(f"\n❌ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

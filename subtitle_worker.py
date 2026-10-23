"""Noninteractive, local-file subtitle engine for background workers.

Stdout is JSONL only. Credentials stay in local configuration/environment.
"""
import argparse
import hashlib
import json
import os
import re
import html
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

from subtitle_refine import VideoSubtitleProcessor, get_ffmpeg_path, load_dotenv

DEFAULTS = dict(target_language="zh-TW", provider="local-runtime", model=None,
                context="", burn_in=False, source_language="auto", whisper_model="base",
                skip_correction=False, max_workers=2, subtitle_source="whisper",
                youtube_subtitle_language=None, youtube_subtitle_kind="manual")
CHILD = None


def options_from_file(path):
    options = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(options, dict) or set(options) - set(DEFAULTS):
        raise ValueError("Unknown options; credentials and command arguments are not accepted")
    options = DEFAULTS | options
    for key in ("context", "source_language", "whisper_model", "provider"):
        if not isinstance(options[key], str) or (key != "context" and not options[key].strip()):
            raise ValueError(f"Invalid {key}")
    for key in ("target_language", "model"):
        if options[key] is not None and (not isinstance(options[key], str) or not options[key].strip()):
            raise ValueError(f"Invalid {key}")
    for key in ("burn_in", "skip_correction"):
        if type(options[key]) is not bool:
            raise ValueError(f"Invalid {key}")
    if type(options["max_workers"]) is not int or not 1 <= options["max_workers"] <= 4:
        raise ValueError("max_workers must be between 1 and 4")
    if options["provider"] not in ("local-runtime", "gemini", "openai"):
        raise ValueError("Invalid provider")
    if options["whisper_model"] not in ("tiny", "base", "small", "medium", "large"):
        raise ValueError("Invalid whisper_model")
    if options["subtitle_source"] not in ("whisper", "youtube"):
        raise ValueError("Invalid subtitle_source")
    if options["youtube_subtitle_kind"] not in ("manual", "auto"):
        raise ValueError("Invalid youtube_subtitle_kind")
    language = options["youtube_subtitle_language"]
    if language is not None and (not isinstance(language, str) or len(language) > 40
                                or not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language)):
        raise ValueError("Invalid youtube_subtitle_language")
    if options["subtitle_source"] == "youtube" and not language:
        raise ValueError("YouTube captions require an explicit language")
    return options


def processor_for(options):
    provider = options["provider"]
    key, url = None, None
    model = options["model"]
    if provider == "local-runtime":
        root = (Path(os.environ["LOCALAPPDATA"]) / "Personal AI Runtime" if sys.platform == "win32"
                else Path.home() / "Library/Application Support/personal-ai-runtime")
        config = json.loads(Path(os.environ.get("SUBTITLE_RUNTIME_CONFIG", root / "connection.json")).expanduser().read_text())
        key, url = config["api_key"], config["base_url"]
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Runtime URL must be local HTTP")
        if not isinstance(key, str) or not key:
            raise ValueError("Missing runtime key")
        model = model or "gpt-6-luna"
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        model = model or "gpt-4o"
    else:
        model = model or "gemini-2.5-flash"
    return VideoSubtitleProcessor(
        provider=provider, openai_api_key=key, openai_base_url=url,
        openai_model=model, gemini_api_key=os.environ.get("GEMINI_API_KEY") if provider == "gemini" else None,
        gemini_model=model, source_language=(options["youtube_subtitle_language"] if options["subtitle_source"] == "youtube" else options["source_language"]),
        whisper_model=options["whisper_model"], max_workers=options["max_workers"],
    )


def run_command(argv, cwd=None):
    global CHILD
    CHILD = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                             stdout=sys.stderr, stderr=sys.stderr)
    try:
        code = CHILD.wait()
        if code:
            raise RuntimeError(f"Media subprocess failed with exit code {code}")
    finally:
        CHILD = None


def cancel(signum, frame):
    if CHILD is not None and CHILD.poll() is None:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(CHILD.pid), "/T", "/F"],
                               stdout=sys.stderr, stderr=sys.stderr, timeout=5)
            else:
                CHILD.kill()
        except (OSError, subprocess.TimeoutExpired):
            pass
    # Do not wait for an in-flight SDK thread or publish a partial result.
    os._exit(128 + signum)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def to_vtt(content):
    entries = VideoSubtitleProcessor._srt_entries(content)
    return "WEBVTT\n\n" + "\n\n".join(
        lines[1].replace(",", ".") + "\n" + "\n".join(lines[2:]) for lines in entries
    ) + "\n"


def normalize_subtitle(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Subtitle input must be a bounded local file")
    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    is_vtt = bool(re.match(r"^WEBVTT(?:[ \t].*)?(?:\n|$)", content.lstrip()))
    cues = []
    last_start = -1
    def timestamp(value):
        if not re.fullmatch(r"(?:\d{2,}:)?[0-5]\d:[0-5]\d[.,]\d{3}", value):
            raise ValueError("Invalid caption timestamp")
        fields = value.replace(",", ".").split(":")
        if len(fields) == 2:
            fields.insert(0, "00")
        h, m, sec = fields
        ms = int(h) * 3600000 + int(m) * 60000 + round(float(sec) * 1000)
        return ms, f"{int(h):02}:{m}:{sec.replace('.', ',')}"
    # In WebVTT, a space-only payload line is not a cue separator.
    separator = r"\n{2,}" if is_vtt else r"\n[ \t]*\n"
    for block in re.split(separator, content.strip()):
        lines = block.splitlines()
        if not lines:
            continue
        if is_vtt and (lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION"))):
            continue
        index = 0 if "-->" in lines[0] else 1
        if index >= len(lines) or "-->" not in lines[index] or (not is_vtt and not lines[0].isdigit()):
            raise ValueError("Invalid caption cue")
        timing = lines[index].split()
        if len(timing) < 3 or timing[1] != "-->":
            raise ValueError("Invalid caption timing")
        start, start_text = timestamp(timing[0])
        end, end_text = timestamp(timing[2])
        if end <= start or start < last_start:
            raise ValueError("Caption timeline must be ordered with positive durations")
        last_start = start
        text = html.unescape(re.sub(r"<[^>]*>", "", "\n".join(lines[index + 1:]))).strip()
        if not text:
            if is_vtt:
                # YouTube rolling captions include empty cues that clear the display.
                # They carry no text for correction/translation; retain timing validation.
                continue
            raise ValueError("Empty caption cue")
        cues.append(f"{len(cues)+1}\n{start_text} --> {end_text}\n{text}")
    result = "\n\n".join(cues) + "\n"
    VideoSubtitleProcessor._srt_entries(result)
    return result


def execute(video, output, options, emit, subtitle_input=None):
    video, output = Path(video).resolve(), Path(output).resolve()
    if not video.is_file():
        raise ValueError("Input video does not exist")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    manifest.unlink(missing_ok=True)
    options = DEFAULTS | options
    imported = options["subtitle_source"] == "youtube"
    if imported != bool(subtitle_input):
        raise ValueError("YouTube source requires --subtitle-input; Whisper must not receive one")
    try:
        subtitle_content = normalize_subtitle(subtitle_input) if imported else None
    except (ValueError, UnicodeError, OSError):
        emit(dict(type="error", message="來源字幕格式或時軸無效，請確認非空 SRT/VTT、起訖時間與順序"))
        raise
    subtitle_hash = digest(subtitle_input) if imported else None
    processor = processor_for(options)
    if not options["skip_correction"] or options["target_language"]:
        if not (processor.client or processor.openai_client):
            raise ValueError("Missing local provider credentials")
    emit(dict(type="progress", stage="importing_captions" if imported else "transcribing", progress=0, message="檢查來源與階段快取"))
    source_hash = digest(video)
    transcription_signature = hashlib.sha256(json.dumps({
        "video": source_hash, "source_language": options["source_language"],
        "whisper_model": options["whisper_model"], "transcription_version": 1,
    }, sort_keys=True).encode()).hexdigest()
    if imported:
        transcription_signature = hashlib.sha256(json.dumps({
            "video": source_hash, "subtitle_hash": subtitle_hash, "import_version": 1,
            "source": options["subtitle_source"], "language": options["youtube_subtitle_language"],
            "kind": options["youtube_subtitle_kind"],
        }, sort_keys=True).encode()).hexdigest()
    signature = hashlib.sha256(json.dumps({
        "video": source_hash, "options": options, "subtitle_hash": subtitle_hash,
        "engine": digest(Path(__file__)),
        "refiner": digest(Path(__file__).with_name("subtitle_refine.py")),
    }, sort_keys=True).encode()).hexdigest()
    checkpoint_path = output / "checkpoint.json"
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        elif checkpoint.get("signature") != signature:
            stages = checkpoint.get("stages", {})
            transcript_hash = stages.get("transcribed.srt") if isinstance(stages, dict) else None
            keep_transcript = checkpoint.get("transcription_signature") == transcription_signature
            checkpoint = {"stages": {"transcribed.srt": transcript_hash} if keep_transcript and transcript_hash else {}}
    except (OSError, ValueError):
        checkpoint = {}
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    checkpoint.setdefault("signature", signature)
    checkpoint["transcription_signature"] = transcription_signature
    if not isinstance(checkpoint.get("stages"), dict):
        checkpoint["stages"] = {}

    def cached(name):
        path = output / name
        return path.is_file() and checkpoint["stages"].get(name) == digest(path)

    def remember(name):
        order = ["transcribed.srt", "corrected.srt", "subtitle.srt", "subtitled.mp4"]
        for later in order[order.index(name) + 1:]:
            checkpoint["stages"].pop(later, None)
        checkpoint["stages"][name] = digest(output / name)
        write_json(checkpoint_path, checkpoint)

    ffmpeg = get_ffmpeg_path()
    # Whisper starts ffmpeg by name, so make the resolved binary discoverable.
    os.environ["PATH"] = str(Path(ffmpeg).resolve().parent) + os.pathsep + os.environ.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix=".work-", dir=output) as work:
        work = Path(work)
        source = output / "transcribed.srt"
        emit(dict(type="progress", stage="importing_captions" if imported else "transcribing", progress=5,
                  message=("匯入 YouTube 手動 CC" if options["youtube_subtitle_kind"] == "manual" else "匯入 YouTube 自動字幕") if imported else "語音轉錄"))
        if not cached(source.name):
            if imported:
                content = subtitle_content
            else:
                audio = work / "audio.wav"
                run_command([ffmpeg, "-nostdin", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(audio)])
                command = [sys.executable, "-m", "whisper", str(audio), "--model", options["whisper_model"],
                           "--output_format", "srt", "--output_dir", str(work), "--verbose", "False"]
                if options["source_language"] != "auto":
                    command += ["--language", options["source_language"]]
                run_command(command)
                content = (work / "audio.srt").read_text(encoding="utf-8-sig")
            processor._srt_entries(content)
            source.write_text(content, encoding="utf-8")
            remember(source.name)
        content = source.read_text(encoding="utf-8")
        artifacts = []
        if not options["skip_correction"]:
            emit(dict(type="progress", stage="correcting", progress=40, message="校正辨識文字"))
            corrected = output / "corrected.srt"
            if not cached(corrected.name):
                result = processor.correct_subtitle_with_llm(content, context=options["context"])
                corrected.write_text(result, encoding="utf-8")
                remember(corrected.name)
            corrected_content = corrected.read_text(encoding="utf-8")
            processor._validate_subtitle_alignment(content, corrected_content)
            content = corrected_content
            artifacts.append(dict(kind="corrected_srt", path=corrected.name))
        final = output / "subtitle.srt"
        if options["target_language"]:
            emit(dict(type="progress", stage="translating", progress=65, message="翻譯字幕"))
            if not cached(final.name):
                result = processor.correct_subtitle_with_llm(content, context=options["context"],
                                                            target_language=options["target_language"])
                final.write_text(result, encoding="utf-8")
                remember(final.name)
            processor._validate_subtitle_alignment(content, final.read_text(encoding="utf-8"))
        else:
            final.write_text(content, encoding="utf-8")
        emit(dict(type="progress", stage="exporting", progress=85, message="匯出 SRT 與 VTT"))
        vtt = output / "subtitle.vtt"
        vtt.write_text(to_vtt(final.read_text(encoding="utf-8")), encoding="utf-8")
        artifacts += [dict(kind="srt", path=final.name), dict(kind="vtt", path=vtt.name)]
        if options["burn_in"]:
            emit(dict(type="progress", stage="burning", progress=90, message="燒錄字幕影片"))
            burned = output / "subtitled.mp4"
            if not cached(burned.name):
                shutil.copyfile(final, work / "subtitle.srt")
                # A fixed relative filter path avoids FFmpeg filter escaping of user paths.
                run_command([ffmpeg, "-nostdin", "-y", "-i", str(video), "-vf", "subtitles=subtitle.srt",
                             "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(burned)], cwd=work)
                remember(burned.name)
            artifacts.append(dict(kind="burned_video", path=burned.name))
    for artifact in artifacts:
        path = output / artifact["path"]
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError("Missing output artifact")
    write_json(manifest, dict(version=1, artifacts=artifacts))
    emit(dict(type="progress", stage="completed", progress=100, message="字幕處理完成"))
    emit(dict(type="result", manifest=str(manifest)))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--options-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subtitle-input")
    args = parser.parse_args()
    # Redirect the actual descriptor, including native libraries/subprocesses;
    # keep a separate descriptor exclusively for protocol records.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    load_dotenv()
    def emit(record):
        protocol.write(json.dumps(record, ensure_ascii=False) + "\n")
        protocol.flush()
    try:
        kwargs = {"subtitle_input": args.subtitle_input} if args.subtitle_input else {}
        execute(args.input, args.output_dir, options_from_file(args.options_json), emit, **kwargs)
        return 0
    except Exception as error:
        # Upstream exceptions may include request content; do not echo credentials.
        print(f"Subtitle worker failed ({type(error).__name__}); see preceding stage diagnostics", file=sys.stderr)
        return 1
    finally:
        protocol.close()


if __name__ == "__main__":
    sys.exit(main())

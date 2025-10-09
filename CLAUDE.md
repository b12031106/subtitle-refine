# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

**Subtitle Refine** - AI 驅動的字幕精煉工具

與其他字幕工具的核心差異：採用**雙重 AI 處理流程**
1. 第一步：使用 Gemini AI 校正 Whisper 語音辨識的錯別字、同音字、斷句錯誤
2. 第二步：在校正後的基礎上進行高品質翻譯

整合了 YouTube 下載、Whisper 語音轉錄、Gemini AI 校正/翻譯、以及 FFmpeg 字幕嵌入功能。核心是單一 Python 腳本 `subtitle_refine.py`（約 1000 行）。

## 常用指令

### 開發與測試

```bash
# 安裝依賴
pip install -r requirements.txt

# 基本執行（YouTube 影片轉錄 + 翻譯）
python subtitle_refine.py \
  --youtube "URL" \
  --api-key "YOUR_GEMINI_API_KEY" \
  --source-lang en \
  --translate zh-TW

# 本地影片處理
python subtitle_refine.py \
  --video "./video.mp4" \
  --api-key "KEY" \
  --translate zh-TW

# 僅嵌入字幕（不使用 AI）
python subtitle_refine.py \
  --video "./video.mp4" \
  --subtitle "./subtitle.srt" \
  --only-embed

# 跳過 AI 校正（僅轉錄）
python subtitle_refine.py \
  --video "./video.mp4" \
  --api-key "KEY" \
  --skip-correction
```

### 檢查依賴工具

```bash
# FFmpeg
ffmpeg -version

# yt-dlp
yt-dlp --version

# Whisper（Python 內部）
python -c "import whisper; print(whisper.__version__)"
```

## 架構說明

### 核心類別：VideoSubtitleProcessor

**初始化參數**：
- `gemini_api_key`: Gemini API 金鑰（only-embed 模式可省略）
- `whisper_model`: Whisper 模型大小（tiny/base/small/medium/large，預設 base）
- `source_language`: 原始語言代碼（zh/en/ja/ko 等，或 auto 自動偵測）
- `gemini_model`: Gemini 模型名稱（預設 gemini-2.5-flash）

**主要處理流程** (`process()` 方法，subtitle_refine.py:810-909)：
1. **下載/載入影片** - YouTube URL 或本地檔案
2. **語音轉錄** - 使用 Whisper 生成 SRT 字幕
3. **AI 校正/翻譯** - 使用 Gemini API（可選）
4. **嵌入字幕** - 使用 FFmpeg 將字幕燒錄到影片

### 關鍵技術細節

#### 1. 字幕分段處理 (subtitle_refine.py:166-202)
- 長字幕會根據字元數（預設 15000）切分成多個片段
- 使用正則表達式 `r'\n\n+'` 分割 SRT 條目
- 避免單次 API 呼叫超過 token 限制

#### 2. 平行處理與 Rate Limiting (subtitle_refine.py:294-307, 574-613)
- 使用 `ThreadPoolExecutor` 平行處理多個字幕片段
- 根據模型自動調整併發數：
  - gemini-2.5-flash: 10 RPM
  - gemini-2.5-pro: 5 RPM
  - gemini-2.5-flash-lite: 15 RPM
- 自動在請求間加入延遲避免超過 rate limit

#### 3. 智能重試機制 (subtitle_refine.py:356-393)
- 偵測 `MAX_TOKENS` finish_reason
- 自動將超限片段切成兩半遞迴重試
- 最多重試 2 次（`max_retries=2`）

#### 4. SRT 格式修正 (subtitle_refine.py:233-292)
- 修正錯誤的時間戳格式（如 "0007:43,000" → "00:07:43,000"）
- 使用 `_fix_srt_timestamp()` 和 `_validate_and_fix_srt()` 方法

#### 5. FFmpeg 路徑處理 (subtitle_refine.py:14-47)
- 優先使用 `static-ffmpeg` 套件提供的 FFmpeg
- 降級至系統安裝的 FFmpeg
- 全域變數 `FFMPEG_CMD` 儲存實際路徑

#### 6. 中文字幕標點符號規範 (subtitle_refine.py:530-539, 557-564)
- 句尾通常不加標點符號
- 僅疑問句保留問號（？）
- 僅感嘆句保留驚嘆號（！）
- 此規則硬編碼在 AI 提示詞中

### 目錄結構

```
downloads/   # YouTube 下載的影片（自動創建）
subtitles/   # Whisper 轉錄的字幕 + AI 校正後的字幕（自動創建）
output/      # 最終帶字幕的影片（自動創建）
```

## 支援的語言與模型

### Whisper 模型
- tiny: 最快，~1GB 記憶體
- base: 平衡選擇（預設）
- small/medium/large: 更高準確度但更慢

### Gemini 模型
- gemini-2.5-flash: 預設，性價比最高
- gemini-2.5-flash-lite: 最快，成本最低
- gemini-2.5-pro: 最強大，思考能力最好
- gemini-2.0-flash: 第二代，長上下文

### 語言代碼
zh/zh-TW (繁中), zh-CN (簡中), en, ja, ko, es, fr, de, it, pt, ru, ar, th, vi, auto (自動偵測)

## 修改注意事項

### 調整 Token 限制
如果經常遇到 `MAX_TOKENS` 錯誤，修改 `correct_subtitle_with_llm()` 的 `max_chars` 參數（subtitle_refine.py:454，預設 15000）。

### 修改 Rate Limit
在 `_get_rate_limit_for_model()` 方法（subtitle_refine.py:294-307）調整 RPM/TPM。

### 修改 AI 提示詞
在 `correct_subtitle_with_llm()` 方法的 `base_prompt_template` 區域（subtitle_refine.py:517-572）。

### 新增支援的 Gemini 模型
在 `_get_rate_limit_for_model()` 的 `rate_limits` 字典中新增模型及其限制。

## 常見問題

### FFmpeg 相關
- 使用 `static-ffmpeg` 套件確保跨平台可用性
- 嵌入字幕時會先嘗試 ASS 格式，失敗則降級至 SRT
- 最後降級方案是軟字幕（不燒錄）

### API 配額
- Gemini API 有免費額度限制
- 使用 `--skip-correction` 可跳過 AI 處理
- 使用 `--only-embed` 完全不使用 API

### 處理長影片
- 自動分段處理避免 token 限制
- 使用平行處理加速（但受 API rate limit 限制）
- 最終會重新合併片段並重新編號 SRT 序號

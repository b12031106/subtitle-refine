# Video Subtitle Tool

自動化影片字幕處理工具 - 支援 YouTube 下載、Whisper 轉錄、Gemini AI 校正/翻譯、FFmpeg 嵌入字幕

## 功能特色

- 🎬 **YouTube 影片下載** - 使用 yt-dlp 下載 YouTube 影片
- 🎤 **自動字幕轉錄** - 使用 OpenAI Whisper 進行語音轉文字
- 🤖 **AI 智能校正/翻譯** - 使用 Google Gemini AI 校正錯別字或翻譯字幕
- ⚡ **平行處理** - 自動分段並平行處理，大幅提升速度
- 🔧 **自動格式修正** - 自動修正 SRT 時間戳格式錯誤
- 📊 **詳細統計資訊** - 顯示 token 使用量、處理進度等
- 🎯 **智能重試機制** - 遇到 token 限制自動切分重試

## 安裝

### 1. 克隆倉庫

```bash
git clone https://github.com/yourusername/video-subtitle.git
cd video-subtitle
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

所有依賴（包括 FFmpeg）都會自動安裝，無需手動安裝額外軟體！

### 3. 取得 Gemini API Key

前往 [Google AI Studio](https://makersuite.google.com/app/apikey) 取得免費的 API key

## 使用方式

### 完整流程：處理 YouTube 影片並翻譯

```bash
python video_subtitle_tool.py \
  --youtube "https://www.youtube.com/watch?v=xxxxx" \
  --api-key "YOUR_GEMINI_API_KEY" \
  --translate zh-TW
```

### 處理本地影片

```bash
python video_subtitle_tool.py \
  --video "./video.mp4" \
  --api-key "YOUR_API_KEY" \
  --source-lang en \
  --translate zh-TW
```

### 僅校正現有字幕並嵌入影片

```bash
python video_subtitle_tool.py \
  --video "./video.mp4" \
  --subtitle "./video.srt" \
  --api-key "YOUR_API_KEY"
```

### 純字幕嵌入（不使用 AI）

```bash
python video_subtitle_tool.py \
  --video "./video.mp4" \
  --subtitle "./video_translated.srt" \
  --only-embed
```

## 命令列參數

### 輸入來源
- `--youtube, -y` - YouTube 影片網址
- `--video, -v` - 本地影片檔案路徑

### API 設定
- `--api-key, -k` - Gemini API 金鑰（使用 --only-embed 時不需要）
- `--gemini-model, -gm` - Gemini 模型名稱（預設: gemini-2.5-flash）

### 語言設定
- `--source-lang, -sl` - 影片的原始語言（預設: zh，可用 auto 自動偵測）
- `--translate, -t` - 翻譯成目標語言（例如: en, ja, ko, zh-TW）

### Whisper 設定
- `--whisper-model, -w` - Whisper 模型大小（tiny/base/small/medium/large，預設: base）

### AI 校正設定
- `--prompt, -p` - 自定義 AI 校正提示詞
- `--context, -c` - 額外的上下文資訊
- `--skip-correction, -s` - 跳過 AI 字幕校正

### 其他選項
- `--subtitle, -sub` - 現有字幕檔案路徑（提供此選項將跳過下載和轉錄步驟）
- `--only-embed, -oe` - 僅執行字幕嵌入（需要同時提供 --video 和 --subtitle）

## 支援的語言代碼

- `zh` / `zh-TW` - 繁體中文
- `zh-CN` - 簡體中文
- `en` - 英文
- `ja` - 日文
- `ko` - 韓文
- `es` - 西班牙文
- `fr` - 法文
- `de` - 德文
- `it` - 義大利文
- `pt` - 葡萄牙文
- `ru` - 俄文
- `ar` - 阿拉伯文
- `th` - 泰文
- `vi` - 越南文
- `auto` - 自動偵測

## 可用的 Gemini 模型

- `gemini-2.5-flash` - 推薦，性價比最高（預設）
- `gemini-2.5-flash-lite` - 最快，成本最低
- `gemini-2.5-pro` - 最強大，思考能力最好
- `gemini-2.0-flash` - 第二代，長上下文

## 進階功能

### 平行處理

工具會自動根據模型的 rate limit 進行平行處理：
- gemini-2.5-flash: 10 RPM
- gemini-2.5-pro: 5 RPM
- gemini-2.5-flash-lite: 15 RPM

### 智能重試

當遇到 token 限制時，會自動將片段切分成更小的部分並重試，確保處理成功。

### 中文字幕標點符號規範

翻譯成中文時，會自動遵守中文字幕習慣：
- 句尾通常不加標點符號
- 疑問句保留問號（？）
- 感嘆句保留驚嘆號（！）

## 專案結構

```
video-subtitle/
├── video_subtitle_tool.py  # 主程式
├── requirements.txt        # Python 依賴
├── .gitignore             # Git 忽略規則
├── README.md              # 說明文件
├── downloads/             # YouTube 下載目錄（自動創建）
├── subtitles/             # 字幕輸出目錄（自動創建）
└── output/                # 最終影片輸出目錄（自動創建）
```

## 注意事項

1. **API 配額** - Gemini API 有免費額度限制，大量使用請注意配額
2. **處理時間** - 長影片處理時間較長，請耐心等待
3. **磁碟空間** - 處理影片需要足夠的磁碟空間

## 授權

MIT License

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 更新日誌

### v1.0.0
- 初始版本發布
- 支援 YouTube 下載、Whisper 轉錄、Gemini AI 校正/翻譯
- 平行處理和智能重試機制
- 自動 SRT 格式修正

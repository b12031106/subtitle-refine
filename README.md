# Subtitle Refine

AI 驅動的字幕精煉工具 - 自動校正語音辨識錯誤並翻譯字幕

支援 YouTube 下載、Whisper 轉錄、Gemini AI 校正/翻譯、FFmpeg 嵌入字幕


## 本機 ChatGPT Runtime 與字幕品質改善

支援 `gemini`、官方 `openai`、`local-runtime` 三種提供商。`local-runtime` 讀取本機
`~/Library/Application Support/personal-ai-runtime/connection.json`（Windows 為
`%LOCALAPPDATA%\Personal AI Runtime\connection.json`），不需要複製金鑰到此專案。
請先依 local-chatgpt-runtime 的說明啟動服務；預設採用設定檔中的模型，可明確指定 Luna：

```sh
python subtitle_refine.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID" \
  --provider local-runtime --openai-model gpt-6-luna \
  --source-lang en --translate zh-TW \
  --context "技術教學；Cloudflare Workers 保留原文；queue 譯為佇列"
```

使用現有影片與字幕測試，可避免再次下載或轉錄：

```sh
python subtitle_refine.py --video ./video.mp4 --subtitle ./video.srt \
  --provider local-runtime --translate zh-TW --max-workers 2
```

- `--runtime-config PATH`：指定另一份本機 connection.json；Runtime 金鑰和 URL 均以此檔為準，URL 限本機 HTTP。
- `--openai-model MODEL`：覆寫模型名稱；可用模型以 runtime 的 `models` 指令回傳為準。
- `--openai-base-url URL` / `OPENAI_BASE_URL`：供 `--provider openai` 使用的相容 API Base URL。
- `--max-workers N`：並行上限，預設 4；這是本工具限制，不代表供應商的實際配額。
- `--skip-correction --translate zh-TW`：略過校正但仍翻譯，適合已有人工字幕。
- `--use-audio-context`：僅 Gemini 校正階段可用；runtime 使用文字，不會聽到原音訊。

指定翻譯時現在會先校正、儲存 `_corrected.srt`，再翻譯並儲存
`_corrected_zh-TW.srt`。因此較舊版增加一次模型處理；可以用中間檔判斷錯誤是在
轉錄、校正還是翻譯階段發生。`--prompt` 作用於校正；`--context`（主題／術語對照）
作用於兩個階段。每個分段附前後各三條字幕作為參考，輸出逐條檢查序號、時間軸與數量；
Runtime 分段上限為 4,000 字元，模型只回傳字幕 ID 與文字，時間軸由程式從原文組回；
漏句或 ID 不符時縮小片段重試，仍不完整才報錯，不會直接嵌入影片。
結構檢查無法保證翻譯語意正確。

實作驗證：`python -m unittest discover -s tests -v`。2026-10-23 本機 Luna 的三條合成字幕
已通過實際校正與翻譯：修正 Cloud flare → Cloudflare，保留否定與 30／13 的數字差異，
序號與時間軸皆一致。完整自動測試目前 19 項通過；另已完成真實長片的 Whisper 與 YouTube 原始字幕兩種來源、Luna 校正／繁中翻譯及燒錄。尚未與 Gemini 做盲測，不能據此判定 Luna 品質優於 Gemini。

## 背景引擎協議

無互動入口（所有路徑建議使用絕對路徑）：

```sh
/path/to/python subtitle_worker.py \
  --input /path/to/video.mp4 \
  --options-json /path/to/options.json \
  --output-dir /path/to/job-output
```

Python 環境需要本專案依賴與 Whisper；FFmpeg 需包含 `subtitles` filter／libass，
中文字型需可用。這台 Mac 已驗證可用的 Python 是
`/Users/justinhsu/.pyenv/versions/3.12.0/bin/python`。

`options.json` 範例：

```json
{
  "provider": "local-runtime",
  "model": "gpt-6-luna",
  "target_language": "zh-TW",
  "source_language": "auto",
  "whisper_model": "base",
  "context": "queue 譯為佇列",
  "burn_in": false,
  "skip_correction": false,
  "max_workers": 2
}
```

`target_language: null` 表示不翻譯。省略 model 時依 provider 使用 Luna／gpt-4o／
gemini-2.5-flash；未知欄位、金鑰與自訂命令不接受。
Runtime 使用本機標準 `connection.json`，可用本機環境變數 `SUBTITLE_RUNTIME_CONFIG`
指定另一份設定；官方 OpenAI／Gemini 使用本機環境金鑰。

stdout 僅輸出 JSONL；Python／原生函式庫／媒體子程序診斷皆送 stderr：

```json
{"type":"progress","stage":"transcribing","progress":5,"message":"語音轉錄"}
{"type":"result","manifest":"/path/to/job-output/manifest.json"}
```

`progress` 是階段估計的 0–100 整體百分比，不是逐秒轉錄進度。stage 為
`transcribing`、`correcting`、`translating`、`exporting`、`burning`、`completed`。
成功 manifest 格式：

```json
{"version":1,"artifacts":[{"kind":"srt","path":"subtitle.srt"},{"kind":"vtt","path":"subtitle.vtt"}]}
```

啟用校正另含 `corrected_srt`；啟用燒錄另含 `burned_video`。路徑相對 output-dir。
燒錄失敗會讓任務失敗，不降級成軟字幕。消費端須同時檢查退出碼 0、result、manifest
與檔案完整性，不能只看到 completed 事件就發布。

同一任務重試沿用 output-dir，可依 `checkpoint.json` 重用已完成階段；來源內容、選項
或程式版本改變會使 AI 階段快取失效，產物雜湊不符也會重跑該階段與後續階段。
Whisper 轉錄有獨立快取，僅依音訊來源、來源語言、Whisper 模型與轉錄流程版本判定，
因此修正 AI 流程或修改術語不用重新轉錄。
不要讓多個程序同時寫同一 output-dir，呼叫端需持有任務鎖。

SIGTERM 退出 143，不發布成功 result；引擎停止直接媒體子程序。
呼叫端必須為引擎建立獨立 process group，取消時對整個 group 發 SIGTERM，限時後
再發 SIGKILL 清理所有 descendants。媒體子程序沿用引擎 group。
已送到模型供應商的請求不保證能撤銷。

2026-10-23 已用約 8 秒合成英文影片完成真實 Whisper base → Luna 校正 → 繁中翻譯 →
SRT/VTT → H.264/AAC 燒錄版測試；JSONL、manifest 與影片檢查通過，抽幀確認繁中字形可讀。
13 項測試涵蓋字幕對齊、快取重用／失效、失敗不發布、stdout 隔離與 SIGTERM。
另外使用 media-cellar 的隔離 SQLite 與真實 SubtitleWorker 呼叫本引擎，
成功登記四種成品，原片媒體數維持一個。此結果不代表長片翻譯品質或正式部署已驗收。

## 與 Media Cellar 的整合狀態與後續方向

首版已在兩個專案的本機工作樹實作：沿用 Media Cellar 的收件介面與雲端派工，
本專案提供獨立字幕引擎。尚未部署或重啟正式服務。Media Cellar 的 25 項 Python 與
12 項 Cloud 測試、字幕引擎 13 項測試已由兩個 pane 分工後重跑驗證。
雲端提供字幕狀態與取消／重試，成品由下載節點的 owner 登入頁取得，尚未上傳 Drive。
現有 Media Cellar 已有 URL 收件、YouTube 下載、節點路由、SQLite 租約／重試、雲端狀態同步；
因此不需要為字幕再建立一套登入與公開 Web server。

建議流程：貼上 YouTube URL → 雲端保存任務 → 本機下載原片 → 獨立字幕 worker
轉錄／校正／翻譯／產生字幕版 → 回報階段狀態與成品。

整合接點與後續擴充方向（首版限制以 Media Cellar README 為準）：

- `cellar/cloud.py` / `cloud/src/dispatch.js`：延伸既有同步與路由，加入字幕選項與節點能力，先固定派給這台能轉錄的主機。
- `cellar/worker.py` / `cellar/db.py`：下載完成後建立持久化字幕子任務；用獨立 worker、租約及 heartbeat，避免長片阻塞一般下載。字幕失敗不應讓原片變成下載失敗。
- 子任務保存來源媒體 ID、目標語言、模型、術語、設定版本與階段 checkpoint；同一原片可有不同翻譯版本，不能只靠 URL 去重。重試只重跑失敗階段。
- 在同一台機器執行下載與字幕處理，先避免跨主機傳輸大檔；Runtime Key 只存在本機，worker 連線 localhost 的 runtime。
- 目前 `inventory()` 只識別圖片／影片；需另加 SRT/VTT 字幕附件及成品關聯，避免字幕版影片被誤算成來源多下載一個檔案。
- 第一版保留原片與 SRT/VTT，播放器載入可切換字幕；另提供燒錄字幕版選項，減少不必要的重新編碼。
- 原本字幕 CLI 的 YouTube 流程仍有互動選擇；背景整合使用上方 `subtitle_worker.py`，由 Media Cellar 負責下載。

品質評估優先順序：選取有代表性的錯譯片段，固定來源字幕／術語／提示詞，比較 Gemini
與 Luna 的專名、否定、數字、漏譯及臺灣口語自然度；另抽查 Whisper 原始轉錄。
目前 Whisper 預設是 `base`，可用相同音訊比較較大模型，但更大的模型仍須實測速度和準確度。

## 核心特色

### 🎯 雙重 AI 處理流程（與其他工具的差異）
- **第一步：AI 校正** - 修正 Whisper 語音辨識的錯別字、同音字、斷句錯誤
- **第二步：AI 翻譯** - 在校正後的基礎上進行高品質翻譯

### ⚡ 其他功能
- 🎬 **YouTube 影片下載** - 使用 yt-dlp 下載 YouTube 影片
- 🎤 **自動字幕轉錄** - 使用 OpenAI Whisper 進行語音轉文字
- 📊 **平行處理** - 自動分段並平行處理，大幅提升速度
- 🔧 **自動格式修正** - 自動修正 SRT 時間戳格式錯誤
- 📈 **詳細統計資訊** - 顯示 token 使用量、處理進度等
- 🔄 **智能重試機制** - 遇到 token 限制自動切分重試

## 安裝

### 1. 克隆倉庫

```bash
git clone https://github.com/yourusername/subtitle-refine.git
cd subtitle-refine
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
python subtitle_refine.py \
  --youtube "https://www.youtube.com/watch?v=xxxxx" \
  --api-key "YOUR_GEMINI_API_KEY" \
  --translate zh-TW
```

### 處理本地影片

```bash
python subtitle_refine.py \
  --video "./video.mp4" \
  --api-key "YOUR_API_KEY" \
  --source-lang en \
  --translate zh-TW
```

### 僅校正現有字幕並嵌入影片

```bash
python subtitle_refine.py \
  --video "./video.mp4" \
  --subtitle "./video.srt" \
  --api-key "YOUR_API_KEY"
```

### 純字幕嵌入（不使用 AI）

```bash
python subtitle_refine.py \
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
subtitle-refine/
├── subtitle_refine.py     # 主程式
├── requirements.txt       # Python 依賴
├── .gitignore            # Git 忽略規則
├── README.md             # 說明文件
├── USAGE_EXAMPLES.md     # 使用範例
├── downloads/            # YouTube 下載目錄（自動創建）
├── subtitles/            # 字幕輸出目錄（自動創建）
└── output/               # 最終影片輸出目錄（自動創建）
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

### 背景 worker：使用現有 YouTube 字幕

media-cellar 下載節點負責取得 YouTube 字幕；`subtitle_worker.py` 不查網路、不使用互動選單。新增可選 argv `--subtitle-input ABS_LOCAL_FILE`，只接受 UTF-8 SRT 或 WEBVTT（上限 10 MiB）。options 新增 `subtitle_source`（`whisper` 預設／`youtube`）、`youtube_subtitle_language`（預設 null，YouTube 必填精確語言碼）、`youtube_subtitle_kind`（`manual` 預設／`auto`）。不得在 options 放路徑或 URL；路徑只由本機 adapter 傳入。Whisper 不接受字幕輸入，YouTube 未給字幕檔則失敗，沒有 fallback。

匯入會正規化 SRT 索引、時間格式、VTT 標記及 HTML entity，檢查非空、正長度與起始時間順序；YouTube VTT 的空白清屏段落會略過，含文字段落的空白行不會被誤當分隔。整份沒有文字仍拒絕；允許原字幕重疊區間，不擅自改時間或合併 rolling cues。匯入時進度 stage 為 `importing_captions`，message 區分手動 CC／自動字幕。其後校正、翻譯、匯出及燒錄契約不變；無效格式／時軸另發固定文字的 `error` 訊息並非零退出，不發布 manifest。

輸入字幕 bytes SHA256、來源語言及類型納入轉錄／匯入 checkpoint signature，來源切換不共用快取；不改變舊 Whisper transcription signature，變更 AI 設定後仍可沿用已完成 Whisper。`transcribed.srt` 檔名保留，CC 模式下存的是正規化匯入字幕，來源由 options／checkpoint 判別。既有舊 options 繼續預設 Whisper。

驗證採本機 SRT/VTT fixture、假 AI／媒體處理與隔離 DB；確認不呼叫 Whisper、重試快取、改字幕／來源類型使後續快取失效、四種 artifact 與 media-cellar 模擬私人上傳接續。隔離測試不操作既有服務或真實授權。另於 2026-10-23 在獨立 media-cellar 測試節點完成真實英文自動字幕匯入（936 條文字字幕）、校正、繁中翻譯及約 20 分鐘影片燒錄；上傳後因 YouTube 帳號未驗證的片長限制而未完成平台處理。另有短片已完成私人上傳與狀態確認。

## 共用 subtitle-engine 0.1.0 遷移

`subtitle_refine.py` 與 `subtitle_worker.py` 現在是薄相容入口，實作由已安裝的 `subtitle-engine==0.1.0` 提供。
舊 CLI argv、worker options、JSONL 和 manifest version 1 不變；`CELLAR_SUBTITLE_COMMAND` 無需改路徑。
共用核心支援三階段獨立 provider 的 Python API；舊 provider/model 仍同時控制校正與翻譯。
Runtime 只是文字 provider；Whisper 繼續是既有 ASR。這不是即時字幕插件。

本 repo 附帶 `vendor/subtitle_engine-0.1.0-py3-none-any.whl` 與 SHA256SUMS；engine 尚無公開／私有 remote，
因此不依賴相鄰 checkout 或不可取得的 Git URL。wheel 是 engine 原始碼建置的固定發行物。
完整安裝仍在本 repo 根目錄執行 `python -m pip install -r requirements.txt`。
既有已裝好模型／媒體依賴的環境，可先以**實際 worker 的 Python**執行：

```sh
python scripts/install_engine.py
python -c 'from importlib.metadata import version; assert version("subtitle-engine") == "0.1.0"'
python -m unittest discover -s tests -v
```

install_engine.py 驗證 SHA256 後以 --no-deps 安裝 bundled wheel，不連網更新其他依賴。
部署應在隔離 checkout 先安裝／驗證，再於無活躍字幕任務時切換程式；勿先替換舊入口再補依賴。
舊入口仍優先讀取自身目錄的 .env，不需要移動或重設 secrets。不得同時執行寫同 output-dir 的程序。

快取：整個 engine package 的實作 hash 取代舊單檔追蹤，第一次切換將使 AI／燒錄快取失效；
Whisper transcription signature 保持原值，已完成且 hash 正確的轉錄可重用。
YouTube 來源字幕下載與任務鎖仍由 media-cellar 管理。回滾時一併切回舊入口 commit；原 Whisper 快取仍可使用。

本次驗證是隔離 fixture／假模型與假媒體 backend，不重做真實模型品質或速度驗收。
既有長片真實驗證紀錄屬舊版歷史，不能當成新 engine 已測過即時速度。

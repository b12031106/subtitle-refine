# Subtitle Refine for Mac — 開發計畫

原生 macOS app（SwiftUI），讓使用者不用開終端機就能：貼 YouTube 連結或拖入影片 → 產生校正／翻譯字幕、翻譯畫面字 → 驗收 → 匯出。
重活全部交給既有 Python 引擎（`subtitle-engine` 0.3.1，已安裝在下方 Python），app 只負責介面、工作佇列與程序管理。

## 固定決策（不要改，除非先回報理由）

- 位置：本 repo 的 `macos/`；分支 `feat/macos-app`。**不要 commit / push**，由驗收方決定。
- 技術：SwiftUI + Swift 6，macOS 14 以上，arm64。用 `xcodegen`（`macos/project.yml` → 產生 `.xcodeproj`，產生物不進版控）。
  用 `xcodebuild` 建置、ad-hoc 簽章（`CODE_SIGN_IDENTITY=-`），不要求開發者帳號。不要用第三方 Swift 套件，除非真的必要。
- 引擎呼叫方式：**只透過子程序**，不改引擎原始碼（引擎在另一個 repo）。
  - Python：預設 `/Users/justinhsu/.pyenv/versions/3.12.0/bin/python`，設定頁可改，並提供「檢查環境」按鈕
    （檢查 `subtitle_engine` 版本、`whisper`、`yt_dlp`、`static_ffmpeg`、選用的 `torch`/`cv2`）。
  - 對白字幕：`subtitle_worker.py` 協議（見 `README.md`「背景引擎協議」），stdout JSONL 進度、`manifest.json` 成品。
  - 畫面字：`python -m subtitle_engine.overlay VIDEO --out-dir DIR ...`（`--help` 看參數）。stdout 是人類可讀的進度行，解析其中可辨識的階段即可，無法辨識的行原樣顯示在日誌。
  - YouTube 下載：`python -m yt_dlp`，輸出到該工作的資料夾。
- 程序管理：每個工作自己的 process group；取消時對整個 group SIGTERM，10 秒後 SIGKILL。同時只跑一個工作，其餘排隊。
- 資料位置：`~/Library/Application Support/SubtitleRefine/`，每個工作一個資料夾（`jobs/<uuid>/`），工作清單存 JSON，app 重開後還在；被中斷的工作可「繼續」（引擎本身有快取／續跑）。
- 金鑰：Gemini / OpenAI API key 存 Keychain，以環境變數傳給子程序；`local-runtime` 不需要金鑰。金鑰不可寫入日誌或檔案。
- 介面語言：繁體中文（台灣用語）。

## 里程碑

每完成一個里程碑：`xcodebuild` 建置成功 + 單元測試通過，然後回報（見最後一節），等驗收後再做下一個。

### M1 骨架與工作佇列
- xcodegen 專案、App 啟動、主視窗：左側工作清單，右側工作詳情（狀態、進度條、即時日誌）。
- 新增工作表單：來源（YouTube 連結 或 拖入/選擇本機影片）、提供商（local-runtime / openai / gemini）、模型（可空）、來源語言、目標語言（可「不翻譯」）、Whisper 模型、術語／背景說明、是否燒錄。
- 設定頁：Python 路徑、檢查環境、API 金鑰（Keychain）。
- `JobRunner`：啟動子程序、解析 JSONL、process group 取消、工作狀態持久化。
- 單元測試：JSONL 解析、options.json 產生（不得出現未知欄位與金鑰）、工作狀態存讀、用假的 python 腳本測取消會殺掉整個 group。

### M2 真正跑起來（對白字幕）
- 串起 YouTube 下載 → worker → 成品清單（SRT/VTT/燒錄影片），可「在 Finder 中顯示」、用 QuickTime 開啟。
- 用一支幾秒鐘的本機測試影片做端對端實測（用 `ffmpeg` 合成含英文語音的短片，或 repo 中既有的小檔），附上實際 manifest 與截圖路徑。

### M3 畫面字翻譯
- 新增工作表單加上「翻譯畫面字」區塊：遮蓋方式 blur / LaMa（LaMa 需 torch，環境檢查不通過時停用並說明）、可選「只處理片段」（起始秒數＋長度）、可帶入同工作產生的對白字幕作為 `--dialogue-srt`。
- 完成後可在 app 內開啟驗收頁（`*.review.html`，用 WKWebView 或預設瀏覽器）。
- 以 `downloads/` 中 EP.22 影片的 30 秒片段（`--start 530 --duration 30 --cover blur`）實測。

### M4 打磨
- 通知中心通知完成／失敗、Dock 圖示進度、App icon、空狀態、錯誤訊息可讀、深色模式。
- `macos/README.md`：建置、執行、架構。

## 回報方式

完成里程碑或卡住時，執行：

```sh
hc send t1h "<一行摘要>；詳情見 /Users/justinhsu/work/b12031106/subtitle-refine/macos/REPORT.md"
```

`macos/REPORT.md` 寫：做了什麼、建置與測試指令和實際輸出結果、未完成或有疑慮的地方。不要只說「應該可以」；沒有實際跑過的就寫沒跑過。

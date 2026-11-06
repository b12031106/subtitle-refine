# Subtitle Refine for Mac

原生 SwiftUI macOS App：加入本機影片或 YouTube 連結，產生校正／翻譯對白字幕、燒錄影片，或翻譯影片上的畫面字。介面使用繁體中文，支援系統深淺色。

## 執行需求

- **Apple Silicon（arm64）、macOS 14 以上**。此發行包不含 Intel 版本。
- **外部 Python 與引擎**：App 不內嵌 Python、模型或本機 AI 服務。這台機器驗證的 Python 為 `/Users/justinhsu/.pyenv/versions/3.12.0/bin/python`（Python 3.12）；其他機器請在 App 設定填入實際 Python 執行檔的絕對路徑，再按「檢查環境」。
- 該 Python 需安裝 `subtitle-engine` **0.3.1**、`openai-whisper`、`yt-dlp`、`static-ffmpeg` 與引擎依賴。repo 的 `requirements.txt`／`vendor/` 提供安裝來源，見 [根目錄 README](../README.md)。請安裝在 App 設定指向的同一個 Python：

  ```sh
  /path/to/python -m pip install -r requirements.txt
  /path/to/python -c 'from importlib.metadata import version; print(version("subtitle-engine"))'
  ```

- FFmpeg 需要 libass／`subtitles` filter，且系統需有中文字型；下載與驗證腳本透過 static_ffmpeg 取得執行檔。初次使用可能下載 FFmpeg 與 Whisper 模型，需網路與足夠儲存空間。
- **畫面字**需要 macOS Apple Vision OCR，首次使用以 `swiftc` 編譯 OCR 小工具；執行機器也要有 Xcode Command Line Tools（`xcode-select --install`）。畫面字提供商只支援 local-runtime／OpenAI；Gemini 選項會說明並停用。
- **LaMa** 另需同一個 Python 可匯入 `torch`、`cv2`，例如安裝 `torch opencv-python-headless`；需 LaMa 權重（引擎使用既有快取，缺少時下載）。預設模型快取在 `~/.cache/torch/hub/checkpoints/big-lama.pt`。App 環境檢查未通過時不可選 LaMa。
- **local-runtime**：先依 local-chatgpt-runtime 專案說明啟動並完成登入，確保 `~/Library/Application Support/personal-ai-runtime/connection.json` 可用。引擎從此檔讀取本機服務位址、金鑰與模型；不必在 App 輸入金鑰。App 不會安裝、啟動、重設或更新 Runtime。Finder 啟動的 App 不會讀取 shell 啟動檔，請優先使用上述標準設定位置。
- OpenAI／Gemini 官方服務的金鑰請儲存在 App 設定；使用 macOS Keychain，僅以環境變數傳給所選提供商的子程序，不寫進工作 options。沒有金鑰仍可使用已設定好的 local-runtime。

## 開啟與使用

Release 產物：`dist/SubtitleRefine.app` 與 `dist/SubtitleRefine-macOS-arm64.zip`。可解壓後放到 Applications，或在 repo 根目錄執行：

```sh
open macos/dist/SubtitleRefine.app
```

此包是本機 **ad-hoc 簽章**，未做 Developer ID 簽章或 notarization；跨機器下載可能需依 macOS「隱私權與安全性」確認開啟。它仍依賴外部 Python 環境。

1. 在設定確認 Python 並「檢查環境」。依提供商準備 Runtime 或 Keychain 金鑰。
2. 按「新增工作」（⌘N），拖入／選擇影片或貼上 YouTube 連結。
3. 選擇中文名稱的來源／目標語言；特殊語言可選「自行輸入語言代碼…」。來源「自動辨識」只適用來源語言，目標不可用 auto。
4. 啟用對白字幕、翻譯畫面字或兩者。關閉區塊時細項收合；內容過長可以捲動，取消／加入佇列固定在表單底部。畫面字可只處理片段、選擇 blur／LaMa，或帶入同工作對白 SRT。
5. 工作依序執行，詳情顯示日誌與成品。影片可用 QuickTime 開啟、檔案可在 Finder 顯示；畫面字的「開啟驗收頁」會使用預設瀏覽器開啟本機原圖／成品對照 HTML。

首次開始工作時，系統會詢問通知權限；完成／失敗通知不包含影片路徑或引擎診斷。可在 App 設定關閉通知，或於 macOS 系統設定調整權限；拒絕通知不影響影片處理。工作執行時 Dock 顯示進度條與百分比，取消時顯示停止中，閒置時清除。

## 中斷、取消與續跑

結束 App（⌘Q，或關閉最後一個視窗）會先停止目前工作的整個 process group：SIGTERM，最多等待 10 秒後 SIGKILL，等程序結束後再退出。正在執行及排隊的工作保留為「已中斷，可繼續」；已明確按取消的工作仍為「已取消」。

重新開啟時不會自動開始處理或消耗 API。選取工作按「繼續」，沿用同一 UUID、輸入、參數與輸出目錄，重用既有下載收據和引擎快取。畫面字快取在 `jobs/<uuid>/output/overlay/<名稱>.onscreen-work/`；續跑不清除此目錄。引擎會自行判斷哪些階段可重用，未完成或不適用的部分會重做。若異常退出留下 running／cancelling／queued 狀態，載入時也轉成中斷。請勿同時手動啟動另一個引擎程序寫入同一工作目錄。

工作清單：`~/Library/Application Support/SubtitleRefine/jobs.json`。工作檔案：同目錄下 `jobs/<uuid>/`。來源影片保留原位置；刪除或移動原片後需放回才能續跑。App 不自動刪除工作或快取。

## 建置與完整測試

開發需要 Xcode（含 macOS SDK／Swift 6）、XcodeGen；不需開發者帳號或第三方 Swift 套件。以下命令皆從 repo 根目錄執行：

```sh
xcodegen generate --spec macos/project.yml
xcodebuild -project macos/SubtitleRefine.xcodeproj -scheme SubtitleRefine \
  -configuration Debug -derivedDataPath macos/.build \
  -destination 'platform=macOS,arch=arm64' CODE_SIGN_IDENTITY=- build test
open macos/.build/Build/Products/Debug/SubtitleRefine.app
```

如需保存結果包，加上 `-resultBundlePath macos/.build/MyRun.xcresult`；每次使用新路徑。測試使用隔離暫存工作目錄與假 worker，不呼叫付費模型。環境檢查測試使用上述預設 Python；Keychain 測試若遇 macOS 禁止互動（-25308）會明確跳過，不解鎖或重設鑰匙圈。

Release 與 zip：

```sh
macos/scripts/release.sh
```

此腳本以 Release 建置、複製 App 到 `macos/dist/`、ad-hoc 簽章、驗證簽章，再產出 `SubtitleRefine-macOS-arm64.zip` 及 `SHA256SUMS`。只取代自身的生成物；不發布、不提交程式、不修改 Python 或 Runtime。`dist/`、`.build/` 與產生的 `.xcodeproj` 均忽略於 Git。

圖示為本專案繪製的字幕對話框圖形，10 種尺寸編成 `Resources/AppIcon.icns`。要重新產生：

```sh
swift macos/scripts/generate_icon.swift macos
```

## 無螢幕錄製權限的快照

```sh
macos/scripts/snapshots.sh
```

腳本重新產生專案、建置 Debug，再用 `SubtitleRefineProbe snapshots` 執行 SwiftUI ImageRenderer，於 `macos/.build/snapshots/` 產生 **28 張 PNG**（14 種情境各深／淺色）。含空清單、對白／畫面字執行與成品、設定、新增表單、Gemini／LaMa 不可用、中斷、失敗、兩區收合及短視窗固定按鈕列。

快照採固定假資料，不讀 Keychain 或真實工作，不需螢幕錄製權限。AppKit 文字輸入與 scroll 容器在 ImageRenderer 下以共用版面的唯讀內容代替；短視窗快照顯示捲動區頂端與固定頁尾，不模擬手勢捲動。測試使用 Vision OCR 檢查主內容、收合內容不可見與短視窗頁尾位置。

## 架構

| 部分 | 責任 |
| --- | --- |
| `SubtitleRefineApp`／`WorkspaceView` | 視窗、工作清單、詳情與成品操作；App delegate 等待程序停止後退出 |
| `NewJobView`／`OverlayControls`／`LanguagePicker` | 表單、能力限制與語言中文名稱；表單細項可收合、內容可捲動 |
| `Models`／`ProcessingOptions` | 工作資料、舊版載入相容、處理組合與嚴格 worker options；原子寫入 jobs JSON |
| `JobRunner` | 單工作佇列、下載 → 對白 → 畫面字、進度、取消、中斷、重啟續跑、成品 manifest |
| `GroupProcess` | posix_spawn 建立獨立 process group，stdout／stderr 讀取，TERM → KILL、金鑰遮蔽 |
| `YouTubeDownloader` | 同 Python 的 yt-dlp、static_ffmpeg、工作內下載與成功收據重用 |
| `Resources/subtitle_worker.py`／`JSONL` | 對白引擎子程序入口與背景 JSONL 協議，詳見根目錄 README |
| `OverlayPipeline` | 既有 overlay CLI argv、人類可讀進度、輸出驗證；快取由引擎管理 |
| `Artifacts` | manifest、路徑與檔案驗證，Finder／QuickTime／瀏覽器開啟 |
| `Settings`／`SettingsView` | Python 能力檢查、Keychain、通知偏好 |
| `DesktopFeedback`／`FailureGuidance` | 通知中心、Dock 進度、可操作的錯誤說明；拒絕通知不使工作失敗 |
| `SnapshotExporter`／`RenderingComponents` | 共用 SwiftUI 元件的無螢幕權限快照 |
| `SubtitleRefineProbe` | 編譯相同 Sources 的驗證工具，與 App 使用相同 Runner／成品開啟路徑 |

App 只以子程序使用既有引擎，不內嵌 Python、不修改引擎程式。對白的 worker options 不含畫面字欄位；兩階段成品最終由 App 寫入 version 1 manifest。開啟成品時再次驗證路徑在工作輸出目錄內。失敗保留原始日誌並顯示修復提示；取消／結束 App 不發送失敗通知。

M2／M3 真實 worker、YouTube 與畫面字驗證指令及成品證據保留於 [REPORT.md](REPORT.md)。M3 可重跑：

```sh
/Users/justinhsu/.pyenv/versions/3.12.0/bin/python macos/scripts/run_overlay_acceptance.py
/Users/justinhsu/.pyenv/versions/3.12.0/bin/python macos/scripts/verify_overlay.py
```

M4 真實中斷／重啟續跑驗證（另開兩個 Probe 程序；EP.22 的 start 530、duration 6、blur、local-runtime）：

```sh
/Users/justinhsu/.pyenv/versions/3.12.0/bin/python macos/scripts/run_restart_acceptance.py
```

輸出在 `macos/.build/m4-restart/`，`verification.json` 記錄同一工作的中斷／完成狀態與 clip、OCR、事件快取前後 SHA-256／mtime。此驗證確實呼叫本機 Runtime。

# M4 實作與驗證回報

日期：2026-11-06。分支：`feat/macos-app`。**未 commit、未 push、未部署。**

M4 與追加項目已完成：通知／Dock 進度、圖示、空狀態與錯誤提示、深淺色畫面、可收合且可捲動的表單、正常關閉 App 後中斷／續跑、Release App／zip、README。最後完整 build test 成功：**32 項、1 項跳過、0 失敗**；snapshots.sh 成功重新產生全部 **28 張** PNG。另以兩個獨立程序真實驗證 EP.22 畫面字中斷後續跑，clip／OCR／事件快取的 SHA-256 與修改時間都未改變。

## 交付物

- [Release SubtitleRefine.app](dist/SubtitleRefine.app)
- [SubtitleRefine-macOS-arm64.zip](dist/SubtitleRefine-macOS-arm64.zip)
- [SHA256SUMS](dist/SHA256SUMS)
- [README.md](README.md)：外部 Python／套件與 Runtime 需求、操作、建置、快照、續跑、架構
- [全部快照](.build/snapshots/)
- [打包驗證紀錄](.build/M4-package-verification.json)

版本 **0.4.0（build 4）**，macOS 14+／arm64，Release、ad-hoc 簽章。App 仍需要外部 Python、引擎及 Runtime，未做 notarization。zip 大小與雜湊：

```text
1,234,231 bytes
SHA-256: d4bc137e95ade9bdc80f82c860cebfa340af981dff0299bb90f14b3998d9c8ab
```

## 實作內容

### 表單與介面

- 對白字幕／翻譯畫面字關閉時只保留標題與開關，細項收合；原有中文語言選單、自訂代碼、Gemini 限制與 LaMa 能力檢查保留。
- 移除 820 高度的固定限制。表單最小高度 420、理想高度依螢幕可用高度決定，內容使用 ScrollView；標題與底部取消／加入佇列放在捲動區外，頁尾有分隔線，短視窗仍能操作。
- ImageRenderer 的 scroll 替代容器改成接受可用高度並裁切內容，避免快照把頁尾擠出畫面。新增 660×520 邏輯尺寸快照，OCR 測試驗證「加入佇列」位於最下方 15% 範圍。
- 空狀態補充加入影片 → 選擇處理方式 → 檢視成品的步驟；失敗詳情依 Python 套件、金鑰驗證、速率限制、連線、來源檔案、權限、空間等問題提供中文修復提示，原始原因與日誌仍保留。
- 系統深淺色沿用語意色彩；新增中斷、失敗、收合、短視窗情境各深淺色快照。

### 中斷與續跑

- App delegate 在 ⌘Q 或關閉最後一個視窗時，等待 `JobRunner.shutdown()` 停止整個 process group（TERM，10 秒後 KILL）後才退出。
- 正常結束 App 導致的停止改為「已中斷，可繼續」，排隊工作也保留為中斷；使用者明確按取消的工作仍為「已取消」。取消／中斷不發失敗通知。
- 載入時將異常退出留下的 running／cancelling／queued 狀態轉成中斷，不自動啟動工作。按繼續才重新執行，沿用原 UUID、輸入／參數、工作與輸出目錄；沒有刪除或重新命名 `onscreen-work`。
- 新增單元測試先啟動會寫 OCR 快取的子程序，結束第一個 Runner 並確認子程序 PID 已消失，重新建立 Runner，按繼續，檢查相同參數／快取、四個成品、排隊工作不誤啟動及通知狀態。另測異常儲存狀態的復原。

### 通知、Dock 與圖示

- `DesktopFeedback` 接收 Runner 狀態：完成／失敗以 UserNotifications 送通知；首次執行工作時詢問系統權限。設定可關閉通知、顯示權限狀態，拒絕或通知送出失敗不影響工作結果。
- 通知只包含通用結果提示，不顯示影片路徑或原始診斷；點擊通知會啟用 App 並選取該工作。
- Dock 在工作執行時顯示圖示上的進度條與百分比徽章，取消時顯示「停止中」，閒置後清除；狀態沒有變化時不重畫。
- 新增字幕對話框圖示與可重跑的 AppKit 產生腳本；產生 10 種尺寸的 icns。Info.plist 明確設定 `CFBundleIconFile = AppIcon.icns`，打包腳本會驗證宣告及資源存在。
- 通知／Dock 狀態以注入的 recording feedback 測試；測試確認只有完成／失敗送通知，取消／中斷不送，日誌／私有路徑不進通知，結束後 Dock 狀態清空。

## 最後建置、完整測試與快照指令

均在 repo 根目錄實際執行；工具 Xcode 27.2（27B52）。

```sh
xcodegen generate --spec macos/project.yml
xcodebuild -project macos/SubtitleRefine.xcodeproj -scheme SubtitleRefine \
  -configuration Debug -derivedDataPath macos/.build \
  -destination 'platform=macOS,arch=arm64' \
  -resultBundlePath macos/.build/M4-verified.xcresult \
  CODE_SIGN_IDENTITY=- build test
macos/scripts/snapshots.sh
macos/scripts/release.sh
```

重跑測試請改用新的 resultBundlePath。最後實際輸出：

```text
** BUILD SUCCEEDED **
Executed 32 tests, with 1 test skipped and 0 failures (0 unexpected) in 28.289 (28.302) seconds
** TEST SUCCEEDED **
```

31 項通過，1 項是既有 Keychain -25308 互動限制跳過；沒有解鎖或重設鑰匙圈。M1–M3 測試全部維持通過，新增 4 項 M4 測試；快照測試擴充至 28 張 PNG，檢查主內容、無不支援元件標記、收合後不可見的細項與短視窗頁尾位置。

- [完整建置／測試日誌](.build/M4-build-test.log)；結果包：`macos/.build/M4-verified.xcresult`
- [快照腳本日誌](.build/M4-snapshots.log)：退出碼 0，全部 28 張重新產生
- [Release 腳本日誌](.build/M4-release.log)：退出碼 0，Release BUILD SUCCEEDED

Release 驗證實際通過：`codesign --verify --deep --strict`、arm64 Mach-O、最低 macOS 14、版本與圖示 plist、App 內 icon／worker 與 source 的 byte 比對、zip CRC 與 zip 內 executable／Info.plist 和 App byte 比對。所有 `.build/` 與 `dist/` 生成物均被 Git 忽略。

## 真實畫面字中斷／重啟續跑

實際執行：

```sh
/Users/justinhsu/.pyenv/versions/3.12.0/bin/python macos/scripts/run_restart_acceptance.py
```

來源為 downloads 中 EP.22，start 530、duration 6、cover blur、provider local-runtime、目標 zh-TW。腳本先啟動 `SubtitleRefineProbe interrupt-overlay`，使用正式 JobRunner，等 OCR／事件快取產生後呼叫與 App 結束相同的 `shutdown()`，等子程序退出，整個 Probe 隨即退出；再啟動另一個 `SubtitleRefineProbe resume`，從磁碟載入同一工作並呼叫 `resume()`。

第二次程序輸出 `RECOVERED: 已中斷，可繼續`，實際完成 AI 判讀、合成與驗收頁（7 個文字事件、4 段翻譯，續跑約 0.3 分鐘）。驗證 clip.mp4、ocr.jsonl、events.json 的 SHA-256 與 mtime_ns **前後完全相同**，工作 ID 與輸出路徑一致，四個成品存在且有效。

工作 ID：`2D513F38-8B8D-45DB-A65F-4222E72768FC`。

- [中斷日誌](.build/m4-restart/interrupt.log)
- [重新啟動／續跑日誌](.build/m4-restart/resume.log)
- [快取雜湊與修改時間驗證](.build/m4-restart/verification.json)
- [腳本結果](.build/M4-restart.log)
- [實際 manifest](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m4-restart/jobs/2D513F38-8B8D-45DB-A65F-4222E72768FC/output/manifest.json>)
- [overlay_video](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m4-restart/jobs/2D513F38-8B8D-45DB-A65F-4222E72768FC/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s_onscreen_zh-TW.mp4>)
- [overlay_review](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m4-restart/jobs/2D513F38-8B8D-45DB-A65F-4222E72768FC/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s.zh-TW.review.html>)
- [overlay_ass](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m4-restart/jobs/2D513F38-8B8D-45DB-A65F-4222E72768FC/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s.zh-TW.ass>)
- [overlay_events](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m4-restart/jobs/2D513F38-8B8D-45DB-A65F-4222E72768FC/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s.zh-TW.json>)

此實測使用 App 共用程式路徑及兩個獨立程序，沒有 GUI 自動點擊或強制殺掉 App；強制終止的儲存狀態復原另由單元測試涵蓋。M3 的 30 秒 blur／6 秒 LaMa 成品證據保留於後方歷史回報。

## 全部 28 張快照

輸出：`macos/.build/snapshots/`，14 個情境各深／淺色。ImageRenderer 假資料模式、不需螢幕錄製權限；原生文字輸入／scroll／drop 容器以共用版面的唯讀內容替代，短視窗快照顯示內容頂端裁切及固定頁尾，不模擬手勢捲動。

| 情境 | 淺色 | 深色 |
| --- | --- | --- |
| 空清單 | [empty-light.png](.build/snapshots/empty-light.png) | [empty-dark.png](.build/snapshots/empty-dark.png) |
| 對白執行中 | [running-light.png](.build/snapshots/running-light.png) | [running-dark.png](.build/snapshots/running-dark.png) |
| 對白成品 | [completed-light.png](.build/snapshots/completed-light.png) | [completed-dark.png](.build/snapshots/completed-dark.png) |
| 新增工作 | [new-job-light.png](.build/snapshots/new-job-light.png) | [new-job-dark.png](.build/snapshots/new-job-dark.png) |
| 設定與通知偏好 | [settings-light.png](.build/snapshots/settings-light.png) | [settings-dark.png](.build/snapshots/settings-dark.png) |
| 畫面字表單 | [overlay-form-light.png](.build/snapshots/overlay-form-light.png) | [overlay-form-dark.png](.build/snapshots/overlay-form-dark.png) |
| 畫面字執行中 | [overlay-running-light.png](.build/snapshots/overlay-running-light.png) | [overlay-running-dark.png](.build/snapshots/overlay-running-dark.png) |
| 畫面字成品 | [overlay-completed-light.png](.build/snapshots/overlay-completed-light.png) | [overlay-completed-dark.png](.build/snapshots/overlay-completed-dark.png) |
| Gemini 不支援 | [overlay-gemini-light.png](.build/snapshots/overlay-gemini-light.png) | [overlay-gemini-dark.png](.build/snapshots/overlay-gemini-dark.png) |
| LaMa 不可用 | [overlay-lama-unavailable-light.png](.build/snapshots/overlay-lama-unavailable-light.png) | [overlay-lama-unavailable-dark.png](.build/snapshots/overlay-lama-unavailable-dark.png) |
| 已中斷，可繼續 | [interrupted-light.png](.build/snapshots/interrupted-light.png) | [interrupted-dark.png](.build/snapshots/interrupted-dark.png) |
| 失敗與修復提示 | [failed-light.png](.build/snapshots/failed-light.png) | [failed-dark.png](.build/snapshots/failed-dark.png) |
| 兩區關閉／細項收合 | [form-collapsed-light.png](.build/snapshots/form-collapsed-light.png) | [form-collapsed-dark.png](.build/snapshots/form-collapsed-dark.png) |
| 短視窗／固定底部按鈕 | [form-compact-light.png](.build/snapshots/form-compact-light.png) | [form-compact-dark.png](.build/snapshots/form-compact-dark.png) |

已實際檢視短視窗深色、收合淺色、中斷深色與 [App icon 預覽](.build/AppIcon-preview.png)。全部 PNG 也通過自動檢查。

## 驗證範圍與限制

- 無實作阻塞；要求的建置、單元測試、快照重產、Release／zip 與真實快取續跑皆已完成。
- 通知中心與 Dock 的狀態／事件已自動測試，**未以真實 GUI 人工驗收通知橫幅、點擊通知或 Dock 動態畫面**；未宣稱已看過這些系統畫面，也未變更系統通知權限。真正顯示通知需使用者允許 macOS 權限。
- 未在另一台電腦或 macOS 14 實機測試；本機編譯最低部署目標為 14.0。Release 是 ad-hoc，不含 Python／Runtime／模型。
- Keychain 的既有測試跳過原因保持明列；沒有提交金鑰、影音暫存或 Release 生成物。未 commit／push／部署。

---

# 歷史回報（M3／M2／M1；以下為當時內容）

# M3 實作與驗證回報

日期：2026-11-06。分支：`feat/macos-app`。**未 commit、未 push、未部署，未開始 M4。**

M3 已完成：畫面字翻譯、blur／LaMa、片段範圍、同工作對白上下文、驗收頁開啟，以及中文語言下拉選單與 Gemini 限制。指定 EP.22 片段已經由 App 共用的 `JobRunner` 完成兩次真實 local-runtime 流程；blur 30 秒與 LaMa 6 秒都有影片、驗收頁、ASS、事件 JSON。建置成功；28 項測試、1 項跳過、0 失敗；20 張深淺色 PNG 快照已輸出。

## 實作內容

- `ProcessingOptions`／`OverlayOptions` 將 App 的處理組合與 worker options 分開儲存；舊工作未含新欄位時仍以原有對白工作載入，不把未知欄位送給 worker。可選僅對白、僅畫面字或兩者，至少選一種。
- 新增畫面字表單：目標語言、blur／LaMa、起始秒數與長度、帶入同工作對白字幕。範圍必須成對、有限值，起始非負、長度大於零；片段範圍只作用於畫面字。勾選上下文時先完成 worker，再把該工作的 SRT 以 `--dialogue-srt` 傳給 overlay。
- LaMa 在目前 Python 的 torch／cv2 環境檢查通過後才可選；未檢查或失敗時停用並解釋。Python 路徑改變會使結果失效；真正執行 LaMa 前再次檢查，檢查中也可取消。
- 語言改成中文名稱下拉選單：來源有「自動辨識」，共同常用選項包括英文、日文、韓文、繁體中文（台灣）、簡體中文（中國）、西／法／德／泰／越／印尼文；「自行輸入語言代碼…」才顯示代碼欄位。對白及畫面字目標語言共用元件。Whisper 的來源 zh-TW／zh-CN 轉成其接受的 zh；顯示與翻譯目標仍保留地區代碼。
- Gemini 時停用畫面字選項，說明只能用 local-runtime 或 OpenAI，且對白仍可用 Gemini；程式層也拒絕 Gemini overlay，避免繞過表單送出不支援參數。
- `OverlayPipeline` 以 argv 呼叫既有 `python -m subtitle_engine.overlay`，辨識 OCR、判讀、合成、LaMa、驗收頁等人類可讀進度，其餘行原樣進入日誌。沿用工作佇列、持久化、單工作 process group、整組取消與快取目錄。
- overlay 結束後驗證影片／驗收頁／ASS／JSON 均存在且非空，產生 App 成品 manifest；有對白時合併兩階段成品。缺檔或程序失敗不會誤判完成。
- 完成頁新增「開啟驗收頁」，使用預設瀏覽器開啟本機 review HTML；影片使用 QuickTime，所有成品可在 Finder 顯示。開啟前再次驗證成品路徑。
- 未修改外部引擎、既有來源影片、API 金鑰或 runtime 設定；overlay 停用 dotenv 自動載入，沿用既有金鑰環境與日誌遮蔽。實測日誌及 jobs JSON 的 runtime key 精確值掃描通過。

## 建置與測試

在 repo 根目錄執行（均已實際執行）：

```sh
xcodegen generate --spec macos/project.yml
xcodebuild \
  -project macos/SubtitleRefine.xcodeproj \
  -scheme SubtitleRefine \
  -configuration Debug \
  -derivedDataPath macos/.build \
  -destination 'platform=macOS,arch=arm64' \
  -resultBundlePath macos/.build/M3-tests.xcresult \
  CODE_SIGN_IDENTITY=- build test
```

重跑請改用新的 resultBundlePath。實際退出碼 0：

```text
** BUILD SUCCEEDED **
Executed 28 tests, with 1 test skipped and 0 failures (0 unexpected) in 25.283 (25.296) seconds
** TEST SUCCEEDED **
```

27 項通過；1 項是既有 Keychain 互動限制（-25308），沒有解鎖或重設鑰匙圈。新增 6 項 M3 測試涵蓋語言與舊資料相容性、CLI 參數／Gemini／範圍限制、進度與必需成品、對白到 overlay 同一 SRT 串接、overlay 取消、缺少 torch 時拒絕 LaMa。既有下載、worker、整組取消等測試也通過；快照測試檢查全部 20 張 PNG 的尺寸、OCR 內容與不支援元件標記。

- 完整建置／測試日誌：[M3-build-test.log](.build/M3-build-test.log)
- 結果包：`macos/.build/M3-tests.xcresult`
- App：`macos/.build/Build/Products/Debug/SubtitleRefine.app`
- 共用程式碼驗證工具：`macos/.build/Build/Products/Debug/SubtitleRefineProbe`

`codesign --verify --deep --strict` 通過；App 內 worker adapter 與 source 以 `cmp` 比對一致。

## 真實 EP.22 端對端實測

來源：`downloads/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.mp4`。兩次皆 provider `local-runtime`、來源韓文、目標 `zh-TW`、start 530；工作隔離於 `.build/m3/`。本次兩個真實案例均為僅畫面字，沒有額外跑完整 EP.22 對白辨識。

`SubtitleRefineProbe overlay` 與 App 編譯同一份 Sources，透過正式 `JobRunner.enqueue` → 環境檢查（LaMa）→ `OverlayPipeline` → `GroupProcess` → 成品驗證／manifest → 完成狀態執行；沒有另外手寫一套 overlay shell 流程。App UI 按鈕未以 GUI 自動點擊；相同開啟處理函式已真實呼叫，詳見下方。

重跑兩種遮蓋，以及使用 static_ffmpeg 驗證／抽格：

```sh
/Users/justinhsu/.pyenv/versions/3.12.0/bin/python macos/scripts/run_overlay_acceptance.py
/Users/justinhsu/.pyenv/versions/3.12.0/bin/python macos/scripts/verify_overlay.py
```

第一支 script 自動找 downloads 中唯一 EP.22 MP4，依序經 Probe 執行 `blur 30`、`lama 6`，兩者 start 固定 530。重跑會建立新工作，`last-manifest-path.txt` 指向最新結果。第二支從 static_ffmpeg 取得 ffmpeg／ffprobe，檢查影片音軌、時長、完整解碼、成品範圍、驗收頁內嵌原圖／成品圖及翻譯事件，再以 `-ss 1 -frames:v 1` 從成品抽 PNG。

| 實測 | 指定片段 | 實際成品時長 | 畫面字事件／翻譯 | 引擎 flags | MP4 大小 |
| --- | --- | --- | --- | --- | --- |
| blur | 530 秒起，30 秒 | 30.030 秒 | 20／17 | 0 | 14,480,144 bytes |
| LaMa | 530 秒起，6 秒 | 6.006 秒 | 7／4 | 0 | 4,609,883 bytes |

兩支均為 1920×1080、H.264、約 29.97 fps，保留 AAC 音訊；完整解碼退出 0。LaMa 實際使用 MPS、修補 180 格，日誌顯示遮罩不可靠改用模糊 0 次，完成約 1.1 分鐘，並非只檢查可 import。使用既有 `~/.cache/torch/hub/checkpoints/big-lama.pt`。兩張抽格已實際檢視，可見繁中畫面字；flags 0 是引擎結果，不代表逐句人工校對過翻譯。

日誌：[blur-run.log](.build/m3/blur-run.log)、[lama-run.log](.build/m3/lama-run.log)、[acceptance.log](.build/m3/acceptance.log)、[media-check.log](.build/m3/media-check.log)。所有實際路徑、大小、SHA-256、ffprobe 結果：[verification.json](.build/m3/verification.json)。

### blur 成品與實際 manifest

- [影片](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/blur/jobs/51850BF0-3F40-46C6-B0C4-116B0DED3D5E/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+30s_onscreen_zh-TW.mp4>)
- [驗收頁](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/blur/jobs/51850BF0-3F40-46C6-B0C4-116B0DED3D5E/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+30s.zh-TW.review.html>)
- [ASS](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/blur/jobs/51850BF0-3F40-46C6-B0C4-116B0DED3D5E/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+30s.zh-TW.ass>)
- [事件 JSON](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/blur/jobs/51850BF0-3F40-46C6-B0C4-116B0DED3D5E/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+30s.zh-TW.json>)
- [抽格 PNG](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/blur-frame.png>)
- [manifest.json](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/blur/jobs/51850BF0-3F40-46C6-B0C4-116B0DED3D5E/output/manifest.json>)

```json
{"artifacts":[{"kind":"overlay_video","path":"overlay\/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+30s_onscreen_zh-TW.mp4"},{"kind":"overlay_review","path":"overlay\/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+30s.zh-TW.review.html"},{"kind":"overlay_ass","path":"overlay\/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+30s.zh-TW.ass"},{"kind":"overlay_events","path":"overlay\/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+30s.zh-TW.json"}],"version":1}
```

### lama 成品與實際 manifest

- [影片](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/lama/jobs/19B04A9E-6F04-464F-8755-DAF7DFAD5363/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s_onscreen_zh-TW_lama.mp4>)
- [驗收頁](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/lama/jobs/19B04A9E-6F04-464F-8755-DAF7DFAD5363/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s.zh-TW_lama.review.html>)
- [ASS](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/lama/jobs/19B04A9E-6F04-464F-8755-DAF7DFAD5363/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s.zh-TW.ass>)
- [事件 JSON](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/lama/jobs/19B04A9E-6F04-464F-8755-DAF7DFAD5363/output/overlay/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s.zh-TW.json>)
- [抽格 PNG](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/lama-frame.png>)
- [manifest.json](</Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/m3/lama/jobs/19B04A9E-6F04-464F-8755-DAF7DFAD5363/output/manifest.json>)

```json
{"artifacts":[{"kind":"overlay_video","path":"overlay\/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s_onscreen_zh-TW_lama.mp4"},{"kind":"overlay_review","path":"overlay\/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s.zh-TW_lama.review.html"},{"kind":"overlay_ass","path":"overlay\/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s.zh-TW.ass"},{"kind":"overlay_events","path":"overlay\/[ENG] 랜플댄 권위자 채령 vs 맑눈광 박은영ㅣ밥은영 EP.22.530s+6s.zh-TW.json"}],"version":1}
```

### 驗收頁開啟

已實際執行：

```sh
macos/.build/Build/Products/Debug/SubtitleRefineProbe open-review macos/.build/m3/blur
```

該命令載入上述真實 manifest，呼叫 App「開啟驗收頁」同一個 `ArtifactActions.open`；NSWorkspace 成功交給預設瀏覽器，退出 0，輸出 `REVIEW_OPENED`。記錄：[open-review.log](.build/m3/open-review.log)。HTML 檔案已檢查包含內嵌原畫面與成品對照圖；未宣稱以 GUI 點擊或對瀏覽器畫面做螢幕截圖。

## 可重跑的畫面快照

```sh
macos/scripts/snapshots.sh
```

已實際退出 0，共 20 張 PNG：10 個情境 × 深淺色；全部在 `macos/.build/snapshots/`。使用 SwiftUI ImageRenderer、2× scale，不需螢幕錄製權限。完整日誌：[M3-snapshots.log](.build/M3-snapshots.log)。

| 情境 | 淺色 | 深色 |
| --- | --- | --- |
| 空清單 | [empty-light.png](.build/snapshots/empty-light.png) | [empty-dark.png](.build/snapshots/empty-dark.png) |
| 對白執行中與日誌 | [running-light.png](.build/snapshots/running-light.png) | [running-dark.png](.build/snapshots/running-dark.png) |
| 對白完成成品 | [completed-light.png](.build/snapshots/completed-light.png) | [completed-dark.png](.build/snapshots/completed-dark.png) |
| 新增工作與中文語言選單 | [new-job-light.png](.build/snapshots/new-job-light.png) | [new-job-dark.png](.build/snapshots/new-job-dark.png) |
| 設定頁 | [settings-light.png](.build/snapshots/settings-light.png) | [settings-dark.png](.build/snapshots/settings-dark.png) |
| 畫面字表單、片段及對白上下文 | [overlay-form-light.png](.build/snapshots/overlay-form-light.png) | [overlay-form-dark.png](.build/snapshots/overlay-form-dark.png) |
| 畫面字執行中與日誌 | [overlay-running-light.png](.build/snapshots/overlay-running-light.png) | [overlay-running-dark.png](.build/snapshots/overlay-running-dark.png) |
| 畫面字成品與開啟驗收頁 | [overlay-completed-light.png](.build/snapshots/overlay-completed-light.png) | [overlay-completed-dark.png](.build/snapshots/overlay-completed-dark.png) |
| Gemini 不支援說明／停用 | [overlay-gemini-light.png](.build/snapshots/overlay-gemini-light.png) | [overlay-gemini-dark.png](.build/snapshots/overlay-gemini-dark.png) |
| LaMa 環境未通過／停用 | [overlay-lama-unavailable-light.png](.build/snapshots/overlay-lama-unavailable-light.png) | [overlay-lama-unavailable-dark.png](.build/snapshots/overlay-lama-unavailable-dark.png) |

快照使用固定假資料與共用 SwiftUI 畫面元件；ImageRenderer 不支援的原生輸入／scroll／drop 容器以唯讀內容顯示，同 M2。快照不代表實際輸出影片，實際成品抽格已分別列於上方。已檢視畫面字表單、深色完成成品與 Gemini 停用說明；20 張均通過自動 OCR／不支援標記檢查。

## 驗證範圍與保留事項

- 本次要求的 blur 30 秒與 LaMa 6 秒皆真實完成；沒有阻塞項目。
- 同工作對白 → overlay 的 SRT 傳遞已以正式 JobRunner 加假 Python fixture 測試通過；未再以整支 EP.22 真實執行對白辨識。真實 worker／燒錄在 M2 已完成並驗收。
- 本次未跑 OpenAI 真實服務；Gemini 是畫面字不支援情境，UI 與後端皆阻擋。
- 保留既有 Keychain 測試跳過；M4 通知、Dock、icon 與 README 留待下一里程碑。
- 所有生成影音、PNG、log、xcresult 位於被忽略的 `.build/`，未加入版控。未 commit／push／部署。

---

# 歷史回報（M2／M1；以下為當時內容）

# M2 實作與驗證回報

日期：2026-11-06。分支：`feat/macos-app`。**未 commit、未 push、未部署，未開始 M3。**

M2 已完成：YouTube 下載 → worker → 成品清單；新增 Finder 顯示與 QuickTime 開啟操作。另完成不需螢幕錄製權限的 10 張 SwiftUI PNG 快照，以及真實 local-runtime 字幕／翻譯／燒錄、指定 YouTube 影片下載實測。

## 實作內容

- `YouTubeDownloader` 與 App 的 `JobRunner` 共用下載流程：先以設定的 Python 取得 static_ffmpeg，再執行 `python -m yt_dlp`；單影片、忽略外部 yt-dlp 設定、argv 傳參、續傳、合併影片與音軌。下載存於工作的 `download/`，成功收據為 `download.json`，重試可重用已完成檔案。
- 下載程序沿用 `GroupProcess`。下載取消後不會進入 worker；各階段依序執行，取消仍為整組 SIGTERM、10 秒後 SIGKILL。
- `ArtifactManifest` 驗證版本、重複條目、非空檔案、相對路徑與 symlink 解析後的範圍；App 依 options 要求 SRT、VTT、校正字幕與燒錄影片，缺少成品即判定失敗。
- 完成工作詳情顯示成品列表。「在 Finder 中顯示」使用 `NSWorkspace.activateFileViewerSelecting`；影片明確指定 `com.apple.QuickTimePlayerX` 開啟，字幕交給系統預設 App；開啟前重新檢查路徑與檔案，錯誤顯示在詳情中。
- `SubtitleRefineProbe` 是與 App 編譯同一份 Sources 的驗證 CLI。`e2e` 直接使用正式 `JobRunner`，只注入工作儲存目錄與相同內容的 worker adapter 路徑；`download` 直接使用正式 `YouTubeDownloader`。**不是另一套 shell 下載／字幕實作。**
- 將 App 入口與可共用畫面元件分離，以 ImageRenderer 輸出固定假資料；新增表單與設定頁也共用實際 SwiftUI 版面。

## 建置與單元測試

Repo 根目錄執行：

```sh
xcodegen generate --spec macos/project.yml
xcodebuild \
  -project macos/SubtitleRefine.xcodeproj \
  -scheme SubtitleRefine \
  -configuration Debug \
  -derivedDataPath macos/.build \
  -destination 'platform=macOS,arch=arm64' \
  -resultBundlePath macos/.build/M2-final.xcresult \
  CODE_SIGN_IDENTITY=- build test
```

重跑請換新的 `-resultBundlePath`，避免 Xcode 拒絕覆寫既有結果包。

實際結果：`BUILD SUCCEEDED`、`TEST SUCCEEDED`、退出碼 0；**22 項測試、1 項跳過、0 失敗**。跳過的是 M1 已知的 Keychain 互動限制（-25308），沒有解鎖或重設鑰匙圈。

新增測試涵蓋：yt-dlp argv／URL 邊界、下載路徑越界與空檔、實際 App 的下載後 worker 串接（假 Python fixture）、下載收據重用、下載取消不得啟動 worker、缺少要求的燒錄成品時拒絕完成、10 張 PNG 的尺寸／OCR 內容／ImageRenderer 不支援元件標記檢查。既有 M1 取消整個 process group 的 10 秒測試仍通過。

結果包：`macos/.build/M2-final.xcresult`。完整日誌：`macos/.build/M2-build-test.log`。

App：`macos/.build/Build/Products/Debug/SubtitleRefine.app`。
CLI：`macos/.build/Build/Products/Debug/SubtitleRefineProbe`。

`codesign --verify --deep --strict` 實際通過；隨 App 複製的 worker adapter 與 source 經 `cmp` 確認一致。

## 可重跑的畫面快照

只需執行：

```sh
macos/scripts/snapshots.sh
```

此腳本實際跑過並退出 0，會重新產生專案、建置，再以 SwiftUI `ImageRenderer`、2× scale 輸出到 `macos/.build/snapshots/`；不啟動 GUI App、不讀使用者工作、不讀 Keychain、不申請螢幕錄製權限。

若已建置，可直接執行：

```sh
macos/.build/Build/Products/Debug/SubtitleRefineProbe snapshots macos/.build/snapshots
```

| 畫面 | 淺色 | 深色 | PNG 尺寸 |
| --- | --- | --- | --- |
| 空清單 | [empty-light.png](.build/snapshots/empty-light.png) | [empty-dark.png](.build/snapshots/empty-dark.png) | 2160×1440 |
| 執行中含日誌 | [running-light.png](.build/snapshots/running-light.png) | [running-dark.png](.build/snapshots/running-dark.png) | 2160×1440 |
| 完成含成品清單 | [completed-light.png](.build/snapshots/completed-light.png) | [completed-dark.png](.build/snapshots/completed-dark.png) | 2160×1600 |
| 新增工作 | [new-job-light.png](.build/snapshots/new-job-light.png) | [new-job-dark.png](.build/snapshots/new-job-dark.png) | 1240×1640 |
| 設定 | [settings-light.png](.build/snapshots/settings-light.png) | [settings-dark.png](.build/snapshots/settings-dark.png) | 1300×1300 |

快照採固定假資料。ImageRenderer 不支援 AppKit 的文字輸入／原生 scroll 容器／drop destination；因此快照模式沿用同一版面與內容，將互動輸入顯示為唯讀文字、直接繪製 scroll 內容，停用拖放容器。App 正常執行時仍使用原生輸入、捲動與拖放。主視窗快照以共用 sidebar row／detail 元件組合，不包含 macOS 視窗裝飾或選單。

已實際查看空清單、執行中日誌、完成清單、表單與設定 PNG，且測試會 OCR 檢查深淺色全部 10 張，不只確認檔案存在。第一輪原生容器產生的空白／黃色禁止圖示已修正，最終 PNG 沒有這些標記。

## 真實 8 秒英文短片端對端

實際命令：

```sh
/Users/justinhsu/.pyenv/versions/3.12.0/bin/python macos/scripts/make_fixture.py

macos/.build/Build/Products/Debug/SubtitleRefineProbe e2e \
  macos/.build/e2e/fixture/english-eight-seconds.mp4 \
  macos/.build/e2e/final \
  macos/Resources/subtitle_worker.py
```

兩個命令皆退出 0。fixture 腳本使用 `/usr/bin/say -v Samantha -r 180` 合成英文，透過 **static_ffmpeg 的 ffmpeg** 配上 960×540／24fps 畫面、H.264／AAC、補足至 8 秒。輸入片 ffprobe 長度為 `8.000000` 秒。

英文內容：

> Hello everyone. Today is a sunny day. We are testing automatic subtitles and translation in this short video.

實際 worker options：`provider=local-runtime`、模型留空（引擎預設）、`source_language=en`、`target_language=zh-TW`、`whisper_model=base`、`skip_correction=false`、`burn_in=true`。使用機器既有 local-runtime 連線設定；沒有修改服務、憑證或引擎。校正 context 明確要求保留英文；第一次試跑使用中文 context 曾讓校正階段也輸出中文，最終改為英文 context 後重跑完整流程，以下皆為最終結果。

日誌實際出現 `transcribing → correcting → translating → exporting → burning → completed`，App 工作狀態為 `completed`，退出碼 0。

真實 manifest：

`macos/.build/e2e/final/jobs/3E5A9281-13C4-4494-8FEF-61265F168E3C/output/manifest.json`

```json
{
  "version": 1,
  "artifacts": [
    { "kind": "corrected_srt", "path": "corrected.srt" },
    { "kind": "srt", "path": "subtitle.srt" },
    { "kind": "vtt", "path": "subtitle.vtt" },
    { "kind": "burned_video", "path": "subtitled.mp4" }
  ]
}
```

以上內容來自實際檔案（僅壓縮每個 artifact 的排版）。成品均位於同一 output 目錄：

| 成品 | 大小 |
| --- | ---: |
| [corrected.srt](.build/e2e/final/jobs/3E5A9281-13C4-4494-8FEF-61265F168E3C/output/corrected.srt) | 174 bytes |
| [subtitle.srt](.build/e2e/final/jobs/3E5A9281-13C4-4494-8FEF-61265F168E3C/output/subtitle.srt) | 168 bytes |
| [subtitle.vtt](.build/e2e/final/jobs/3E5A9281-13C4-4494-8FEF-61265F168E3C/output/subtitle.vtt) | 173 bytes |
| [subtitled.mp4](.build/e2e/final/jobs/3E5A9281-13C4-4494-8FEF-61265F168E3C/output/subtitled.mp4) | 82,069 bytes |

英文校正稿含 `Hello everyone, today is a sunny day.`；繁中字幕實際內容：

```srt
1
00:00:00,000 --> 00:00:05,440
大家好，今天是個晴朗的日子。我們正在測試自動字幕和翻譯

2
00:00:05,440 --> 00:00:06,560
就在這段短片中
```

成片 ffprobe 長度 `8.035000` 秒、H.264 + AAC；全片解碼檢查通過。已實際查看以下 ffmpeg 擷取畫格，繁中字形可讀且已燒錄在畫面內：

- [burned-1s.png](.build/e2e/final/frames/burned-1s.png)
- [burned-6s.png](.build/e2e/final/frames/burned-6s.png)

雜湊、大小與完整 ffprobe 資料：`macos/.build/e2e/final/verification.json`。

## 真實 YouTube 下載

實際命令（與 App 使用相同 `YouTubeDownloader.download`）：

```sh
macos/.build/Build/Products/Debug/SubtitleRefineProbe download \
  'https://www.youtube.com/watch?v=jNQXAC9IVRw' \
  macos/.build/e2e/youtube
```

退出碼 0，實際下載影片及音軌，合併完成：

```text
DOWNLOADED: /Users/justinhsu/work/b12031106/subtitle-refine/macos/.build/e2e/youtube/source.mp4
```

下載檔長度 `19.064000` 秒，320×240 AV1 影片 + AAC 音軌，完整解碼檢查通過；收據在同目錄 `download.json`。這支 YouTube 影片依要求只下載，沒有跑 AI。yt-dlp 曾輸出未找到 JavaScript runtime 的警告，但本次下載完整成功；沒有安裝額外 runtime，也沒有取用瀏覽器 cookies。

## 影片驗證重跑

```sh
/Users/justinhsu/.pyenv/versions/3.12.0/bin/python macos/scripts/verify_media.py \
  macos/.build/e2e/final \
  macos/.build/e2e/youtube/source.mp4
```

實際輸出摘要：

```json
{"verified": true, "duration": "8.035000", "youtube_duration": "19.064000"}
```

此腳本驗證必要成品、非空檔案、繁中內容、英文校正稿、VTT 格式、影片長度、H.264／AAC、兩部影片完整解碼，並輸出燒錄畫格與 SHA-256。

## 驗證界線

- PNG 是 ImageRenderer 繪製的共用 SwiftUI 元件與假資料，並非螢幕擷取；沒有宣稱實際點過 Finder／QuickTime 按鈕或完成 GUI 自動化。按鈕已接上正式 NSWorkspace 操作，影片格式與可解碼性已實測。
- 唯一跳過的單元測試仍為既有 Keychain 系統互動限制。未啟動任何需要真實 API key 的雲端提供商。
- 真實 E2E 與 YouTube 使用 `.build/e2e/` 隔離資料；App 正常資料位置仍為既有 Application Support。所有影音、PNG、日誌、Xcode 產物均在忽略路徑中。
- M3 畫面字翻譯尚未開始。

---

# M1 歷史回報（已驗收，保留供參考）

# M1 實作與驗證回報

日期：2026-11-06。分支：`feat/macos-app`。

M1 程式實作完成，最終 `xcodebuild build test` 退出碼為 0，建置成功；16 項測試中 15 項通過、1 項額外 Keychain 整合測試因系統工作階段限制跳過，0 項失敗。PLAN 指定的 JSONL、options、持久化與整組程序取消測試皆實際通過。視窗畫面及真實 Keychain 成功存取仍待互動式工作階段驗收，詳見下方限制。

沒有 commit、push 或部署；沒有修改 Python 引擎、重設任何金鑰、安裝套件或新增雲端資源。未開始 M2–M4。

## 已完成

- `project.yml`：XcodeGen、Swift 6 strict concurrency、macOS 14+、arm64、ad-hoc 簽章；無第三方 Swift 套件。產生的 `.xcodeproj`、`.build`、xcresult 已忽略。
- SwiftUI 主視窗：左側工作清單、右側詳情、狀態、百分比、進度條、即時日誌、取消與繼續；繁體中文介面。
- 新增工作表單：YouTube 連結驗證、本機影片選取／拖放、提供商、選填模型、來源語言、目標語言／不翻譯、Whisper 模型、術語／背景說明、燒錄選項。
- 設定：可修改 Python 路徑、非同步檢查六項依賴、OpenAI／Gemini SecureField 與 Keychain 讀寫／刪除。欄位留空不覆蓋現有金鑰；local-runtime 不讀取 Keychain。
- `JobRunner`：一次一個工作、失敗後繼續下一個、取消排隊項目、重試沿用同一 UUID／工作資料夾、關閉 App 時先取消並等待執行中的程序。
- `JobStore`：預設使用 `~/Library/Application Support/SubtitleRefine/jobs/<uuid>/`，原子寫入 `jobs.json` 與 `options.json`；重開後執行中／取消中／排隊中的工作變為可繼續的「已中斷」。清單損壞時保留原檔並停寫；寫入失敗會停止佇列。
- `GroupProcess`：`posix_spawn` 在執行 child 前建立獨立 process group，使用 argv 而非 shell；stdin 為 `/dev/null`，stdout／stderr 分開讀取。取消整個 group 先 SIGTERM，滿 10 秒後 SIGKILL；排空輸出並確認程序退出後才啟動下一項。
- 啟動時重設 child 的訊號遮罩與必要的預設處理，避免繼承 XCTest／GUI launcher 阻擋的 SIGTERM。這個問題由真實取消測試發現並修正。
- JSONL 以 bytes 緩衝，支援 UTF-8 字元跨 chunk、CRLF、最後一行沒有換行、未知事件與純文字診斷。超長行整行丟棄，日誌保留最新 2,000 行。
- `WorkerOptions` 使用固定 CodingKeys；不翻譯明確輸出 `target_language: null`；空白模型不輸出。金鑰與命令不屬於 options schema。
- 子程序環境先移除繼承的 API_KEY／TOKEN／SECRET，再僅注入所選提供商的 Keychain 金鑰；日誌在解析與寫入前遮蔽該金鑰。
- App 內附極薄的 `subtitle_worker.py` adapter，呼叫已安裝的 `subtitle_engine.worker.main`，不載入 repo 的 `.env`。本機 worker 呼叫基礎已接好；完成狀態必須同時符合退出碼 0、result 與有效 manifest／成品檔。

## 最終建置與測試

於 repo 根目錄執行：

```sh
xcodegen generate --spec macos/project.yml
xcodebuild \
  -project macos/SubtitleRefine.xcodeproj \
  -scheme SubtitleRefine \
  -configuration Debug \
  -derivedDataPath macos/.build \
  -destination 'platform=macOS,arch=arm64' \
  -resultBundlePath macos/.build/M1-acceptance.xcresult \
  CODE_SIGN_IDENTITY=- build test
```

注意：重跑時請使用新的 `-resultBundlePath`；Xcode 不接受覆蓋已存在的結果包。

實際輸出：

```text
** BUILD SUCCEEDED **
Test Case '-[SubtitleRefineTests.ProcessTests testFakePythonCancellationKillsEntireGroupAfterTenSeconds]' passed (10.100 seconds).
Executed 16 tests, with 1 test skipped and 0 failures (0 unexpected) in 15.205 (15.214) seconds
** TEST SUCCEEDED **
```

完整最終日誌：`macos/.build/M1-build-test.log`（亦保留 `/tmp/subtitle-refine-m1-acceptance.log`）。
結果包：`macos/.build/M1-acceptance.xcresult`。
App：`macos/.build/Build/Products/Debug/SubtitleRefine.app`。

核心測試涵蓋：JSONL 分段／Unicode／損壞輸入、options 精確欄位與 null、持久化及損壞檔保護、機密遮蔽、manifest 缺檔／越界、啟動前取消與無效執行檔、stdout／stderr 與非零退出、FIFO、失敗後接續、同資料夾續跑、排隊取消、關閉清理、假 worker 成功結果、YouTube URL 驗證、實際環境檢查。

取消測試建立假的 Python 腳本，parent 接收 SIGTERM，child 忽略 SIGTERM。驗證兩者在同一個獨立 group、parent 確實收到 TERM、10 秒後 KILL、兩個 PID 均消失。另有正常 SIGTERM 取消測試，最終只花 0.079 秒，沒有不必要地等待 10 秒。

測試使用 macOS 無 App host 的 logic test bundle，直接編譯相同 Sources（排除 App 入口），避免依賴本機故障的 App-host test launcher。所有工作／成品測試使用臨時資料夾，不呼叫付費服務。

簽章與架構另實際執行：

```sh
codesign --verify --deep --strict --verbose=2 macos/.build/Build/Products/Debug/SubtitleRefine.app
file macos/.build/Build/Products/Debug/SubtitleRefine.app/Contents/MacOS/SubtitleRefine
```

結果：`valid on disk`、`satisfies its Designated Requirement`，可執行檔為 `Mach-O 64-bit executable arm64`。

## 實際環境檢查

由 `AppSettings.checkEnvironment()` 的測試直接啟動預設 Python，實際輸出：

```text
subtitle_engine: 0.3.1
whisper: 20250625
yt_dlp: 2026.10.26
static_ffmpeg: 2.13
torch: 2.8.0
cv2: 可匯入
```

`cv2` 可正常 import，但目前安裝 distribution 名稱不是 `opencv-python`，因此顯示「可匯入」而不是猜測版本。此測試 4.233 秒完成。

## 限制與尚未驗證項目

1. **Keychain 成功存取仍待驗收。** 額外測試只使用隨機測試 service 與假字串，系統在新增項目時回覆 `-25308 / User interaction is not allowed`；`security show-keychain-info` 也得到相同錯誤。測試只對此特定環境錯誤使用 `XCTSkip`，其他錯誤仍會失敗。沒有解鎖、重設或修改使用者既有金鑰；不能宣稱真實 Keychain round-trip 已通過。
2. **畫面與互動式操作未完成驗證。** App 程序已實際啟動；程序取樣顯示已進入 SwiftUI → NSApplication 的主事件迴圈。但 CUA 以 App 路徑與 bundle ID 取得視窗皆回覆 `cgWindowNotFound`，因此沒有可提供的實際畫面截圖，也沒有聲稱已點測表單、拖放或設定按鈕。測試啟動的程序已清理。可在互動式桌面開啟上方 App 路徑驗收。
3. 初次 App-host `xcodebuild test` 因本機 Xcode `IDELaunchServicesLauncher` 的 `childPID > 0` 內部斷言而中止；改為無 host 的 logic tests 後最終 build/test 已成功。本機亦有 CoreSimulator 版本不一致與未使用 AppIntents 的警告；未修改／重啟任何本機服務。
4. **M2 的 YouTube 下載與真影片端對端測試尚未做。** M1 可儲存 YouTube 工作，但執行時會明確提示下載串接於 M2 提供，狀態為失敗且可繼續。沒有下載影片、跑真實轉錄／翻譯／燒錄或宣稱有真影片成品。成品列表與 Finder／QuickTime 開啟留待 M2；畫面字翻譯與打磨留待 M3/M4。

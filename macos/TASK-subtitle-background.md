# 任務：燒錄對白字幕時可選「字幕底色」

## 背景
演講類影片（投影片、畫面上有很多字）燒錄字幕時，白字黑邊會和背景糊在一起，難以閱讀。
使用者範例截圖：`/private/tmp/claude-501/-Users-justinhsu-work-b12031106-subtitle-refine/f2e34854-d146-4f0d-8cfd-d88b072c07d6/images/1.png`
（底部的繁中字幕疊在影片原有的英文字上）。需要可選的半透明底色方塊，讓字幕本身清楚可讀。

## 範圍（兩個 repo）

### 1. 引擎 `/Users/justinhsu/work/b12031106/subtitle-engine`（目前 0.3.1，乾淨的 main）
- `worker.py`：options 新增欄位（建議 `subtitle_background`: `"none"`（預設，維持現狀）｜`"box"`，以及 `subtitle_background_opacity`: 0–100，預設 60）。照現有驗證方式嚴格檢查型別與範圍；未知欄位仍要拒絕。
- 燒錄（`worker.py` 約 400 行 `subtitles=subtitle.srt`）在 `box` 時用 libass `force_style` 畫出每行一個半透明深色底塊（例如 `BorderStyle=3`、`BackColour/OutlineColour` 帶 alpha、適當 padding），白字。請實際比較 BorderStyle 3 與 4 在這版 static_ffmpeg libass 的效果，選每行底塊最乾淨的那個。
- 這兩個欄位要納入燒錄階段的 checkpoint／快取鍵：改底色後重跑只重新燒錄，不要重新轉錄或重跑 AI。
- 舊的 CLI 路徑 `processor.py` 約 1351 行如有共用燒錄，也加上同樣選項（CLI 旗標 `--subtitle-background box --subtitle-background-opacity 60`）；若不共用就在報告說明。
- 測試：options 驗證、快取鍵、以及真實燒錄一支短片後抽格，比較字幕區域背景亮度（有底 vs 無底）證明底塊真的出現。
- 版本升為 0.3.2，README 補說明；建置 wheel。**不要 commit / tag / push**。

### 2. subtitle-refine（本 repo，分支 `feat/macos-app`）
- 用 `scripts/install_engine.py` 的既有流程把 0.3.2 wheel 放進 `vendor/` 並更新 SHA256SUMS（照 0.3.1 那次的做法），並安裝到 `/Users/justinhsu/.pyenv/versions/3.12.0/bin/python`。
- Mac app：新增工作表單「將對白字幕燒錄到影片」下方加「字幕底色」開關與不透明度滑桿（燒錄未勾時停用），寫進 worker options；工作詳情顯示此設定。WorkerOptions 測試、快照更新（含開啟底色的表單）。
- 實測：拿使用者截圖那種情境——用 static_ffmpeg 合成一支背景有大量白色英文字的短片（或從 `/Users/justinhsu/work/b12031106/subtitle-refine/downloads/` 找現成影片的一小段），經 app 的 JobRunner 跑完整 worker 燒錄，`none` 與 `box` 各一次，各抽一格 PNG 放在 `macos/.build/subtitle-bg/`。
- **不要 commit / push**。

## 回報
全部做完後：`hc send t1h "<一行摘要>；詳情見 macos/REPORT.md"`，REPORT.md 最上方新增本任務段落（建置／測試實際輸出、兩張抽格路徑、引擎改動摘要）。中途不要停下來問；真的卡住才回報。

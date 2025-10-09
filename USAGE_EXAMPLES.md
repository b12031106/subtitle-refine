# Subtitle Refine - 使用範例

## 基本使用

### 1. 處理 YouTube 影片（英文轉繁體中文）

```bash
python subtitle_refine.py \
  --youtube "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --api-key "YOUR_GEMINI_API_KEY" \
  --source-lang en \
  --translate zh-TW
```

### 2. 處理本地影片（自動偵測語言並翻譯）

```bash
python subtitle_refine.py \
  --video "./my_video.mp4" \
  --api-key "YOUR_API_KEY" \
  --source-lang auto \
  --translate zh-TW
```

### 3. 只轉錄字幕，不翻譯

```bash
python subtitle_refine.py \
  --video "./video.mp4" \
  --api-key "YOUR_API_KEY" \
  --skip-correction
```

## 進階使用

### 4. 使用更強大的模型（Gemini Pro）

```bash
python subtitle_refine.py \
  --video "./video.mp4" \
  --api-key "YOUR_API_KEY" \
  --gemini-model "gemini-2.5-pro" \
  --translate zh-TW
```

### 5. 使用更精確的 Whisper 模型

```bash
python subtitle_refine.py \
  --youtube "https://www.youtube.com/watch?v=xxxxx" \
  --api-key "YOUR_API_KEY" \
  --whisper-model large \
  --translate zh-TW
```

### 6. 自定義 AI 提示詞

```bash
python subtitle_refine.py \
  --video "./video.mp4" \
  --subtitle "./video.srt" \
  --api-key "YOUR_API_KEY" \
  --prompt "請用輕鬆幽默的語氣翻譯，並保留原文的笑點" \
  --translate zh-TW
```

### 7. 提供上下文資訊協助翻譯

```bash
python subtitle_refine.py \
  --video "./tech_video.mp4" \
  --api-key "YOUR_API_KEY" \
  --context "這是一個關於 Python 程式設計的教學影片" \
  --translate zh-TW
```

## 工作流程範例

### 情境 1：從頭開始處理 YouTube 影片

```bash
# 1. 下載、轉錄、翻譯、嵌入字幕（一步完成）
python subtitle_refine.py \
  --youtube "https://www.youtube.com/watch?v=xxxxx" \
  --api-key "YOUR_API_KEY" \
  --source-lang en \
  --translate zh-TW
```

### 情境 2：已有字幕檔案，只需翻譯並嵌入

```bash
# 使用現有字幕，翻譯後嵌入影片
python subtitle_refine.py \
  --video "./video.mp4" \
  --subtitle "./video.srt" \
  --api-key "YOUR_API_KEY" \
  --translate zh-TW
```

### 情境 3：已有翻譯好的字幕，只需嵌入

```bash
# 純嵌入，不使用 AI（不需要 API key）
python subtitle_refine.py \
  --video "./video.mp4" \
  --subtitle "./video_zh-TW.srt" \
  --only-embed
```

### 情境 4：批次處理多個影片

```bash
#!/bin/bash
# batch_process.sh

VIDEOS=(
  "./video1.mp4"
  "./video2.mp4"
  "./video3.mp4"
)

for video in "${VIDEOS[@]}"; do
  echo "Processing: $video"
  python subtitle_refine.py \
    --video "$video" \
    --api-key "YOUR_API_KEY" \
    --source-lang auto \
    --translate zh-TW
done
```

## 效能優化

### 使用較小的片段大小（避免 token 限制）

如果經常遇到 `MAX_TOKENS` 錯誤，可以修改程式碼中的 `max_chars` 參數：

```python
# 在 subtitle_refine.py 中找到這一行
max_chars: int = 15000

# 改為更小的值
max_chars: int = 10000  # 或 12000
```

### 選擇合適的模型

- **快速處理**: `gemini-2.5-flash-lite`
- **平衡**: `gemini-2.5-flash`（預設）
- **高品質**: `gemini-2.5-pro`

### 選擇合適的 Whisper 模型

| 模型 | 速度 | 準確度 | 記憶體使用 |
|------|------|--------|-----------|
| tiny | 最快 | 較低 | ~1 GB |
| base | 快 | 中等 | ~1 GB |
| small | 中等 | 良好 | ~2 GB |
| medium | 慢 | 很好 | ~5 GB |
| large | 最慢 | 最好 | ~10 GB |

## 疑難排解

### 問題 1: 遇到 MAX_TOKENS 錯誤

**解決方案**: 程式會自動重試，但如果經常發生：
- 減少 `max_chars` 參數
- 使用較小的 Whisper 模型（生成較短的字幕）

### 問題 2: API 配額用完

**解決方案**:
- 使用 `--skip-correction` 跳過 AI 處理
- 等待配額重置（通常每天重置）
- 升級到付費 API

### 問題 3: 翻譯品質不佳

**解決方案**:
- 使用 `gemini-2.5-pro` 模型
- 提供 `--context` 參數
- 使用 `--prompt` 自定義翻譯風格

### 問題 4: 處理速度太慢

**解決方案**:
- 使用 `gemini-2.5-flash-lite` 模型
- 使用 `tiny` 或 `base` Whisper 模型
- 程式已自動平行處理，無需額外設定

import SwiftUI

struct OverlayControls: View {
    @Environment(\.snapshotMode) private var snapshot
    @Bindable var settings: AppSettings
    var provider: Provider
    var dialogueEnabled: Bool
    @Binding var enabled: Bool
    @Binding var cover: OverlayCover
    @Binding var target: String
    @Binding var segment: Bool
    @Binding var start: String
    @Binding var duration: String
    @Binding var useDialogue: Bool

    var body: some View {
        FormSection("翻譯畫面字") {
            FormToggle(title: "辨識並翻譯影片上的文字", isOn: $enabled).disabled(provider == .gemini)
            if provider == .gemini {
                Text("Gemini 不支援畫面字翻譯。請切換至 local-runtime 或 OpenAI；對白字幕仍可使用 Gemini。")
                    .font(.callout).foregroundStyle(.secondary)
            } else if enabled {
                LanguagePicker(title: "畫面字目標語言", code: $target)
                if snapshot {
                    HStack { Text("原文遮蓋方式"); Spacer(); Text(cover.title) }
                } else {
                    Picker("原文遮蓋方式", selection: $cover) {
                        Text(OverlayCover.blur.title).tag(OverlayCover.blur)
                        Text(OverlayCover.lama.title).tag(OverlayCover.lama).disabled(!settings.lamaAvailable)
                    }
                }
                HStack(alignment: .top) {
                    Text(settings.lamaAvailable ? "LaMa 可用：目前 Python 的 torch 與 cv2 已通過檢查。" : "LaMa 尚不可用：請先檢查目前 Python 的 torch 與 cv2；檢查未通過時只能選擇 blur。")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button(settings.checking ? "檢查中…" : "檢查環境") { Task { await settings.checkEnvironment() } }.disabled(settings.checking)
                }
                if cover == .lama { Text("LaMa 會修補原文字形，處理時間較長，可取消後繼續。").font(.caption).foregroundStyle(.secondary) }
                FormToggle(title: "只處理片段", isOn: $segment)
                if segment {
                    HStack {
                        FormTextField(title: "起始秒數", text: $start)
                        FormTextField(title: "長度（秒）", text: $duration)
                    }
                }
                FormToggle(title: "帶入同工作產生的對白字幕", isOn: $useDialogue).disabled(!dialogueEnabled)
                Text(dialogueEnabled ? "先產生對白字幕，再提供給畫面字翻譯作為上下文。片段秒數只影響畫面字處理。" : "若要帶入對白字幕，請同時啟用「產生對白字幕」。")
                    .font(.caption).foregroundStyle(.secondary)
                Text("完成後可開啟原畫面／成品對照的驗收頁。").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

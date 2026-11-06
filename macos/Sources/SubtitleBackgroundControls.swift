import SwiftUI

struct SubtitleBackgroundControls: View {
    @Environment(\.snapshotMode) private var snapshot
    @Binding var options: WorkerOptions
    private var enabled: Binding<Bool> {
        Binding(get: { options.subtitleBackground == .box }, set: { options.subtitleBackground = $0 ? .box : .none })
    }
    private var opacity: Binding<Double> {
        Binding(get: { Double(options.subtitleBackgroundOpacity) }, set: { options.subtitleBackgroundOpacity = Int($0.rounded()) })
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            FormToggle(title: "字幕底色（每行半透明黑底）", isOn: enabled)
                .disabled(!options.burnIn)
            HStack(spacing: 12) {
                Text("不透明度").font(.callout)
                if snapshot { JobProgressView(value: opacity.wrappedValue).frame(width: 180) }
                else { Slider(value: opacity, in: 0...100, step: 1).accessibilityLabel("字幕底色不透明度") }
                Text("\(options.subtitleBackgroundOpacity)%").monospacedDigit().frame(width: 45, alignment: .trailing)
            }.disabled(!options.burnIn || options.subtitleBackground != .box)
            Text(options.burnIn ? "0% 完全透明，100% 不透明；只影響燒錄影片，SRT／VTT 文字不變。" : "勾選燒錄後可設定字幕底色。")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

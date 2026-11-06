import SwiftUI
import AppKit

@MainActor enum SnapshotExporter {
    static func export(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var running = Job(source: .local("/snapshot/english-eight-seconds.mp4"))
        running.id = UUID(uuidString: "11111111-1111-1111-1111-111111111111")!
        running.status = .running
        running.options.sourceLanguage = "en"
        running.options.burnIn = true
        running.progress = 52
        running.stage = "正在翻譯為繁體中文"
        running.logs = ["[transcribing] 語音轉錄完成，共 3 段字幕", "[correcting] 已完成英文校正", "[translating] 正在翻譯為繁體中文", "提供商：local-runtime · 模型：預設"]
        var completed = running
        completed.status = .completed
        completed.progress = 100
        completed.stage = "所有成品已準備完成"
        completed.logs += ["[exporting] 已匯出 SRT 與 WebVTT", "[burning] 字幕燒錄完成", "[completed] 完成"]
        let artifacts = [JobArtifact(kind: "srt", path: "translated.srt"), JobArtifact(kind: "vtt", path: "translated.vtt"),
                         JobArtifact(kind: "corrected_srt", path: "corrected.srt"), JobArtifact(kind: "burned_video", path: "subtitled.mp4")]
        let settings = AppSettings(pythonPath: AppSettings.defaultPython)
        settings.environmentReport = "subtitle_engine: 0.3.2\nwhisper: 可匯入\nyt_dlp: 可匯入\nstatic_ffmpeg: 可匯入\ntorch / cv2: 可匯入（選用）"
        settings.checkedPython = settings.pythonPath
        settings.packageStatus = ["torch": true, "cv2": true]
        let unavailable = AppSettings(pythonPath: "/example/python-needs-environment-check")
        var overlayRunning = running
        overlayRunning.source = .local("/snapshot/밥은영 EP.22.mp4")
        overlayRunning.options.sourceLanguage = "ko"
        overlayRunning.processing = ProcessingOptions(dialogueEnabled: false,
            overlay: OverlayOptions(cover: .blur, start: 530, duration: 30))
        overlayRunning.progress = 65
        overlayRunning.stage = "翻譯與判讀畫面字"
        overlayRunning.logs = ["✂️ 擷取片段 530s +30s", "🔤 OCR（Apple Vision）", "🧩 20 個文字事件 → 13 次 LLM 判讀", "🤖 判讀 13/13"]
        var overlayComplete = overlayRunning
        overlayComplete.status = .completed
        overlayComplete.progress = 100
        overlayComplete.stage = "完成，可開啟驗收頁"
        overlayComplete.logs += ["🎨 合成 17 段畫面字翻譯（遮蓋：blur）", "📝 產生驗收頁（原畫面／成品對照）", "✅ 完成"]
        let overlayArtifacts = [JobArtifact(kind: "overlay_video", path: "overlay/EP22_onscreen_zh-TW.mp4"),
            JobArtifact(kind: "overlay_review", path: "overlay/EP22.zh-TW.review.html"),
            JobArtifact(kind: "overlay_ass", path: "overlay/EP22.zh-TW.ass"),
            JobArtifact(kind: "overlay_events", path: "overlay/EP22.zh-TW.json")]
        var backgroundComplete = completed
        backgroundComplete.options.subtitleBackground = .box
        backgroundComplete.options.subtitleBackgroundOpacity = 80
        var interrupted = overlayRunning
        interrupted.status = .interrupted
        interrupted.stage = "已中斷，可繼續；工作目錄與快取已保留"
        interrupted.logs += ["上次結束 App 時保留 onscreen-work 快取；按繼續後沿用相同工作目錄。"]
        var failed = running
        failed.status = .failed
        failed.stage = "引擎執行失敗（退出碼 1），請查看日誌。"
        failed.logs = ["ModuleNotFoundError: No module named whisper"]
        for scheme in [ColorScheme.light, .dark] {
            let suffix = scheme == .light ? "light" : "dark"
            try render(WorkspaceSnapshot(job: nil), name: "empty-\(suffix)", size: CGSize(width: 1080, height: 720), scheme: scheme, directory: directory)
            try render(WorkspaceSnapshot(job: running), name: "running-\(suffix)", size: CGSize(width: 1080, height: 720), scheme: scheme, directory: directory)
            try render(WorkspaceSnapshot(job: completed, artifacts: artifacts), name: "completed-\(suffix)", size: CGSize(width: 1080, height: 800), scheme: scheme, directory: directory)
            try render(NewJobView(sampleData: true) { _, _, _ in }, name: "new-job-\(suffix)", size: CGSize(width: 660, height: 1150), scheme: scheme, directory: directory)
            try render(SettingsView(settings: settings), name: "settings-\(suffix)", size: CGSize(width: 650, height: 780), scheme: scheme, directory: directory)
            try render(NewJobView(settings: settings, sampleData: true, sampleOverlay: true) { _, _, _ in }, name: "overlay-form-\(suffix)", size: CGSize(width: 660, height: 1370), scheme: scheme, directory: directory)
            try render(WorkspaceSnapshot(job: overlayRunning), name: "overlay-running-\(suffix)", size: CGSize(width: 1080, height: 800), scheme: scheme, directory: directory)
            try render(WorkspaceSnapshot(job: overlayComplete, artifacts: overlayArtifacts), name: "overlay-completed-\(suffix)", size: CGSize(width: 1080, height: 880), scheme: scheme, directory: directory)
            try render(OverlaySettingsSnapshot(settings: settings, provider: .gemini, enabled: false), name: "overlay-gemini-\(suffix)", size: CGSize(width: 660, height: 400), scheme: scheme, directory: directory)
            try render(NewJobView(sampleData: true, sampleBackground: true) { _, _, _ in }, name: "subtitle-bg-form-\(suffix)", size: CGSize(width: 660, height: 1150), scheme: scheme, directory: directory)
            try render(WorkspaceSnapshot(job: backgroundComplete, artifacts: artifacts), name: "subtitle-bg-completed-\(suffix)", size: CGSize(width: 1080, height: 820), scheme: scheme, directory: directory)
            try render(WorkspaceSnapshot(job: interrupted), name: "interrupted-\(suffix)", size: CGSize(width: 1080, height: 720), scheme: scheme, directory: directory)
            try render(WorkspaceSnapshot(job: failed), name: "failed-\(suffix)", size: CGSize(width: 1080, height: 780), scheme: scheme, directory: directory)
            try render(NewJobView(sampleData: true, sampleCollapsed: true) { _, _, _ in }, name: "form-collapsed-\(suffix)", size: CGSize(width: 660, height: 850), scheme: scheme, directory: directory)
            try render(NewJobView(settings: settings, sampleData: true, sampleOverlay: true) { _, _, _ in }, name: "form-compact-\(suffix)", size: CGSize(width: 660, height: 520), scheme: scheme, directory: directory)
            try render(OverlaySettingsSnapshot(settings: unavailable), name: "overlay-lama-unavailable-\(suffix)", size: CGSize(width: 660, height: 600), scheme: scheme, directory: directory)

        }
    }

    private static func render<V: View>(_ view: V, name: String, size: CGSize, scheme: ColorScheme, directory: URL) throws {
        let content = view.frame(width: size.width, height: size.height)
            .background(scheme == .dark ? Color(red: 0.12, green: 0.13, blue: 0.15) : Color.white)
            .environment(\.snapshotMode, true)
            .environment(\.colorScheme, scheme)
            .environment(\.locale, Locale(identifier: "zh_TW"))
        let renderer = ImageRenderer(content: content)
        renderer.scale = 2
        renderer.proposedSize = ProposedViewSize(size)
        guard let cgImage = renderer.cgImage,
              let data = NSBitmapImageRep(cgImage: cgImage).representation(using: .png, properties: [:]) else {
            throw AppError(message: "無法輸出快照：\(name)")
        }
        try data.write(to: directory.appendingPathComponent(name + ".png"), options: .atomic)
        print("SNAPSHOT: \(name).png \(cgImage.width)x\(cgImage.height)")
    }
}

struct WorkspaceSnapshot: View {
    let job: Job?
    var artifacts: [JobArtifact] = []
    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 22) {
                HStack { Text("工作佇列").font(.title2.bold()); Spacer(); Image(systemName: "plus") }
                if let job { JobRowView(job: job).padding(12).background(Color.accentColor.opacity(0.12), in: RoundedRectangle(cornerRadius: 8)) }
                Spacer()
                HStack { Text("\(job == nil ? 0 : 1) 個工作"); Spacer(); Image(systemName: "gearshape") }.foregroundStyle(.secondary)
            }.padding(20).frame(width: 290).background(.quaternary.opacity(0.25))
            Divider()
            if let job { JobDetailView(job: job, artifacts: artifacts) }
            else { EmptyJobsView() }
        }
    }
}

struct OverlaySettingsSnapshot: View {
    let settings: AppSettings
    var provider: Provider = .local
    @State var enabled = true
    @State var cover: OverlayCover = .blur
    @State var target = "zh-TW"
    @State var segment = true
    @State var start = "530"
    @State var duration = "30"
    @State var useDialogue = true
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("畫面字工作設定").font(.title2.bold())
            Text("提供商：\(provider.rawValue)").foregroundStyle(.secondary)
            OverlayControls(settings: settings, provider: provider, dialogueEnabled: true,
                enabled: $enabled, cover: $cover, target: $target, segment: $segment,
                start: $start, duration: $duration, useDialogue: $useDialogue)
            Spacer(minLength: 0)
        }.padding(24)
    }
}

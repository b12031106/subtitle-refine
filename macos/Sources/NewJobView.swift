import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct NewJobView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.snapshotMode) private var snapshot
    @Bindable var settings: AppSettings
    let submit: (JobSource, WorkerOptions, ProcessingOptions) -> Void
    @State private var youtube = false
    @State private var url = ""
    @State private var video: URL?
    @State private var options = WorkerOptions()
    @State private var model = ""
    @State private var dialogueEnabled = true
    @State private var translate = true
    @State private var targetLanguage = "zh-TW"
    @State private var overlayEnabled = false
    @State private var cover: OverlayCover = .blur
    @State private var overlayTarget = "zh-TW"
    @State private var segment = false
    @State private var start = "0"
    @State private var duration = "30"
    @State private var useDialogue = false
    @State private var selectingVideo = false
    @State private var dropTarget = false
    @State private var error: String?

    init(settings: AppSettings? = nil, sampleData: Bool = false, sampleOverlay: Bool = false, sampleCollapsed: Bool = false,
         submit: @escaping (JobSource, WorkerOptions, ProcessingOptions) -> Void) {
        self.settings = settings ?? AppSettings()
        self.submit = submit
        if sampleData {
            _video = State(initialValue: URL(fileURLWithPath: sampleOverlay ? "/snapshot/밥은영 EP.22.mp4" : "/snapshot/english-eight-seconds.mp4"))
            var sample = WorkerOptions()
            sample.sourceLanguage = sampleOverlay ? "ko" : "en"
            sample.burnIn = true
            sample.context = "subtitle 譯為字幕；채령 譯為彩領。"
            _options = State(initialValue: sample)
        }
        if sampleCollapsed { _dialogueEnabled = State(initialValue: false) }
        if sampleOverlay {
            _overlayEnabled = State(initialValue: true)
            _segment = State(initialValue: true)
            _start = State(initialValue: "530")
            _useDialogue = State(initialValue: true)
        }
    }

    private var processing: ProcessingOptions {
        ProcessingOptions(dialogueEnabled: dialogueEnabled, overlay: overlayEnabled ?
            OverlayOptions(cover: cover, targetLanguage: overlayTarget,
                start: segment ? Double(start) : nil, duration: segment ? Double(duration) : nil, useDialogue: useDialogue) : nil)
    }
    private var valid: Bool {
        guard youtube ? Self.validYouTubeURL(url) : video != nil else { return false }
        guard LanguageCatalog.validCode(options.sourceLanguage, allowAuto: true) else { return false }
        if dialogueEnabled && translate && !LanguageCatalog.validCode(targetLanguage, allowAuto: false) { return false }
        if overlayEnabled {
            if cover == .lama && !settings.lamaAvailable { return false }
            if segment && (Double(start) == nil || Double(duration) == nil) { return false }
        }
        return (try? processing.validate(provider: options.provider)) != nil
    }
    static func validYouTubeURL(_ value: String) -> Bool {
        guard let url = URL(string: value), ["https", "http"].contains(url.scheme?.lowercased() ?? ""),
              let host = url.host?.lowercased() else { return false }
        return host == "youtu.be" || host == "youtube.com" || host.hasSuffix(".youtube.com")
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack { Text("新增字幕工作").font(.title2.bold()); Spacer() }.padding(24).fixedSize(horizontal: false, vertical: true)
            RenderableScroll {
                VStack(alignment: .leading, spacing: 16) {
                    sourceSection
                    FormSection("處理設定") {
                        FormChoice(title: "提供商", selection: Binding(get: { options.provider.rawValue }, set: { options.provider = Provider(rawValue: $0) ?? .local }), values: Provider.allCases.map(\.rawValue))
                        LanguagePicker(title: "來源語言", code: $options.sourceLanguage, allowAuto: true)
                        FormTextField(title: "模型（留空使用預設）", text: $model)
                        FormTextField(title: "術語／背景說明", text: $options.context, multiline: true)
                    }
                    FormSection("對白字幕") {
                        FormToggle(title: "產生對白字幕", isOn: $dialogueEnabled)
                        if dialogueEnabled {
                            FormToggle(title: "翻譯對白字幕", isOn: $translate)
                            if translate { LanguagePicker(title: "對白目標語言", code: $targetLanguage) }
                            else { Text("不翻譯，保留來源語言。").font(.caption).foregroundStyle(.secondary) }
                            FormChoice(title: "Whisper 模型", selection: $options.whisperModel, values: ["tiny", "base", "small", "medium", "large"])
                            FormToggle(title: "將對白字幕燒錄到影片", isOn: $options.burnIn)
                        }
                    }
                    OverlayControls(settings: settings, provider: options.provider, dialogueEnabled: dialogueEnabled,
                        enabled: $overlayEnabled, cover: $cover, target: $overlayTarget, segment: $segment,
                        start: $start, duration: $duration, useDialogue: $useDialogue)
                    if let error { Text(error).foregroundStyle(.red) }
                }.padding(.horizontal, 20).padding(.bottom, 8)
            }
            Divider()
            HStack {
                Text("工作會依序執行，一次處理一支影片。").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("取消") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("加入佇列") {
                    options.model = model.nilIfBlank
                    options.targetLanguage = translate ? targetLanguage : nil
                    let source: JobSource = youtube ? .youtube(url) : .local(video!.path)
                    submit(source, options, processing)
                    dismiss()
                }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction).disabled(!valid)
            }.padding(20).fixedSize(horizontal: false, vertical: true)
        }
        .frame(width: 660)
        .frame(minHeight: snapshot ? nil : 420,
               idealHeight: snapshot ? nil : min(760, (NSScreen.main?.visibleFrame.height ?? 900) - 120),
               maxHeight: snapshot ? nil : .infinity)
            .onChange(of: options.provider) { _, provider in if provider == .gemini { overlayEnabled = false } }
            .onChange(of: dialogueEnabled) { _, enabled in if !enabled { useDialogue = false } }
            .fileImporter(isPresented: $selectingVideo, allowedContentTypes: [.movie, .video]) { result in
                switch result {
                case .success(let url): acceptVideo(url)
                case .failure: error = "無法開啟所選影片，請再試一次。"
                }
            }
    }
    private var sourceSection: some View {
        FormSection("影片來源") {
            if snapshot {
                HStack {
                    Text("本機影片").frame(maxWidth: .infinity).padding(6).background(Color.accentColor.opacity(0.16), in: RoundedRectangle(cornerRadius: 6))
                    Text("YouTube 連結").frame(maxWidth: .infinity).foregroundStyle(.secondary)
                }
            } else {
                Picker("來源", selection: $youtube) {
                    Text("本機影片").tag(false)
                    Text("YouTube 連結").tag(true)
                }.pickerStyle(.segmented)
            }
            if youtube {
                FormTextField(title: "YouTube 影片連結", text: $url)
                Text("會先下載這支影片，再執行所選工作。").font(.caption).foregroundStyle(.secondary)
            } else if snapshot { videoDropArea }
            else {
                videoDropArea.dropDestination(for: URL.self) { items, _ in
                    guard let first = items.first, first.isFileURL else { return false }
                    acceptVideo(first); return video == first
                } isTargeted: { dropTarget = $0 }
            }
        }
    }
    private var videoDropArea: some View {
        HStack(spacing: 12) {
            Image(systemName: "film").font(.title)
            Text(video?.lastPathComponent ?? "將影片拖到這裡").lineLimit(2).textSelection(.enabled)
            Spacer()
            Button("選擇影片…") { selectingVideo = true }
        }.frame(maxWidth: .infinity).padding(20)
            .background(dropTarget ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
    }
    private func acceptVideo(_ url: URL) {
        guard url.isFileURL, let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .contentTypeKey]),
              values.isRegularFile == true,
              values.contentType?.conforms(to: .movie) == true || values.contentType?.conforms(to: .video) == true else {
            error = "請選擇可讀取的本機影片檔案。"; return
        }
        video = url
        error = nil
    }
}

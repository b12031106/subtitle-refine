import SwiftUI

struct ContentView: View {
    @Bindable var runner: JobRunner
    @State private var selection: UUID?
    @State private var showingNewJob = false
    var body: some View {
        NavigationSplitView {
            List(runner.jobs, selection: $selection) { job in
                JobRowView(job: job).tag(job.id)
            }
            .navigationTitle("工作佇列")
            .navigationSplitViewColumnWidth(min: 240, ideal: 290)
            .safeAreaInset(edge: .bottom) {
                HStack {
                    Text("\(runner.jobs.count) 個工作").foregroundStyle(.secondary)
                    Spacer()
                    SettingsLink { Image(systemName: "gearshape") }.help("設定")
                }.padding()
            }
        } detail: {
            if let job = runner.jobs.first(where: { $0.id == selection }) {
                JobDetailView(job: job, runner: runner)
            } else {
                EmptyJobsView { showingNewJob = true }
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .selectSubtitleJob)) { notification in
            if let value = notification.object as? String, let id = UUID(uuidString: value) { selection = id }
        }
        .toolbar {
            Button { showingNewJob = true } label: { Label("新增工作", systemImage: "plus") }
                .keyboardShortcut("n")
        }
        .sheet(isPresented: $showingNewJob) {
            NewJobView(settings: runner.settings) { source, options, processing in
                selection = runner.enqueue(source: source, options: options, processing: processing)
            }
        }
        .alert("無法完成操作", isPresented: Binding(get: { runner.errorMessage != nil }, set: { if !$0 { runner.errorMessage = nil } })) {
            Button("好") { runner.errorMessage = nil }
        } message: { Text(runner.errorMessage ?? "") }
    }
}

struct JobDetailView: View {
    let job: Job
    let runner: JobRunner?
    let output: URL
    @State private var artifacts: [JobArtifact]
    @State private var artifactError: String?

    init(job: Job, runner: JobRunner? = nil, output: URL? = nil, artifacts: [JobArtifact] = []) {
        self.job = job
        self.runner = runner
        self.output = output ?? runner?.outputDirectory(for: job) ?? URL(fileURLWithPath: "/tmp/snapshot-output")
        _artifacts = State(initialValue: artifacts)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 8) {
                    Text(job.source.displayName).font(.title2.bold()).lineLimit(2)
                    Text(job.status.title).font(.headline).foregroundStyle(.secondary)
                }
                Spacer()
                if [.queued, .running].contains(job.status) {
                    Button("取消工作", role: .destructive) { runner?.cancel(job.id) }
                }
                if job.status.canResume { Button("繼續") { runner?.resume(job.id) }.buttonStyle(.borderedProminent) }
            }
            if job.status == .failed {
                let guidance = FailureGuidance.describe(stage: job.stage, logs: job.logs)
                VStack(alignment: .leading, spacing: 6) {
                    Label(guidance.title, systemImage: "exclamationmark.circle").font(.headline)
                    Text(guidance.recovery).font(.callout)
                }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            }
            JobProgressView(value: job.progress)
            HStack(alignment: .top) {
                Text(job.stage).textSelection(.enabled)
                Spacer()
                Text("\(Int(job.progress))%").monospacedDigit()
            }.font(.callout)
            Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 8) {
                GridRow { Text("提供商").foregroundStyle(.secondary); Text(job.options.provider.rawValue) }
                if job.request.dialogueEnabled {
                    GridRow { Text("翻譯").foregroundStyle(.secondary); Text("\(LanguageCatalog.title(job.options.sourceLanguage)) → \(job.options.targetLanguage.map(LanguageCatalog.title) ?? "不翻譯")") }
                } else {
                    GridRow { Text("來源語言").foregroundStyle(.secondary); Text(LanguageCatalog.title(job.options.sourceLanguage)) }
                }
                if job.request.dialogueEnabled { GridRow { Text("Whisper").foregroundStyle(.secondary); Text(job.options.whisperModel) } }
                if job.request.dialogueEnabled && job.options.burnIn {
                    GridRow {
                        Text("字幕底色").foregroundStyle(.secondary)
                        Text(job.options.subtitleBackground == .box ? "半透明黑底（不透明度 \(job.options.subtitleBackgroundOpacity)%）" : "無底色")
                    }
                }
                if let overlay = job.request.overlay {
                    GridRow { Text("畫面字").foregroundStyle(.secondary); Text("\(overlay.cover.title) · \(LanguageCatalog.title(overlay.targetLanguage))") }
                    if let start = overlay.start, let duration = overlay.duration {
                        GridRow { Text("片段").foregroundStyle(.secondary); Text("從 \(start.formatted()) 秒開始，長 \(duration.formatted()) 秒") }
                    }
                }
            }.font(.callout)
            if job.status == .completed {
                ArtifactListView(artifacts: artifacts) { artifact in
                    do { try ArtifactActions.reveal(artifact, output: output) }
                    catch { artifactError = error.localizedDescription }
                } open: { artifact in
                    Task {
                        do { try await ArtifactActions.open(artifact, output: output) }
                        catch { artifactError = error.localizedDescription }
                    }
                }
                if let artifactError { Text(artifactError).font(.caption).foregroundStyle(.red) }
            }
            Divider()
            Text("即時日誌").font(.headline)
            ScrollViewReader { proxy in
                RenderableScroll {
                    JobLogView(lines: job.logs)
                    Color.clear.frame(height: 1).id("end")
                }
                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                .onChange(of: job.logs) { _, _ in proxy.scrollTo("end", anchor: .bottom) }
            }
        }.padding(24).navigationTitle("工作詳情")
            .task(id: job.manifestPath) {
                artifactError = nil
                guard job.status == .completed, let path = job.manifestPath else { return }
                do { artifacts = try ArtifactManifest.load(path, output: output) }
                catch { artifacts = []; artifactError = error.localizedDescription }
            }
    }
}

struct JobRowView: View {
    let job: Job
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(job.source.displayName).lineLimit(1).font(.headline)
            HStack {
                Text(job.status.title)
                Spacer()
                Text(job.options.provider.rawValue)
            }.font(.caption).foregroundStyle(.secondary)
            if job.status == .running || job.status == .cancelling {
                JobProgressView(value: job.progress)
            }
        }.padding(.vertical, 5)
    }
}

struct EmptyJobsView: View {
    var add: () -> Void = {}
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "captions.bubble").font(.system(size: 48)).foregroundStyle(.secondary)
            Text("開始製作字幕").font(.title2.bold())
            Text("加入影片或 YouTube 連結，製作對白字幕、翻譯畫面字，完成後檢視成品。")
                .foregroundStyle(.secondary).multilineTextAlignment(.center)
            HStack(spacing: 24) {
                Label("加入影片", systemImage: "film")
                Label("選擇處理方式", systemImage: "slider.horizontal.3")
                Label("檢視成品", systemImage: "checkmark.circle")
            }.font(.caption).foregroundStyle(.secondary)
            Button("新增工作", action: add).buttonStyle(.borderedProminent)
        }.padding(32).frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct JobLogView: View {
    let lines: [String]
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if lines.isEmpty { Text("程序啟動後，進度與診斷訊息會顯示在這裡。").foregroundStyle(.secondary) }
            ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                Text(line).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }.padding(12)
    }
}

struct ArtifactListView: View {
    let artifacts: [JobArtifact]
    var reveal: (JobArtifact) -> Void = { _ in }
    var open: (JobArtifact) -> Void = { _ in }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("成品檔案").font(.headline)
            ForEach(artifacts) { artifact in
                HStack(spacing: 12) {
                    Image(systemName: artifact.isVideo ? "film" : "doc.text").foregroundStyle(Color.accentColor)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(artifact.title).font(.callout.bold())
                        Text(artifact.path).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                    Spacer()
                    Button("在 Finder 中顯示") { reveal(artifact) }
                    if artifact.isReview {
                        Button("開啟驗收頁") { open(artifact) }.buttonStyle(.borderedProminent)
                    } else {
                        Button(artifact.isVideo ? "用 QuickTime 開啟" : "開啟") { open(artifact) }
                    }
                }
            }
        }.padding(14).background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
    }
}

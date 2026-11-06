import Foundation

struct AppError: LocalizedError {
    var message: String
    var errorDescription: String? { message }
}

@MainActor @Observable final class JobRunner {
    private(set) var jobs: [Job] = []
    var errorMessage: String?
    private(set) var activeID: UUID?
    private let store: JobStore
    let settings: AppSettings
    private var process: GroupProcess?
    private var downloader: YouTubeDownloader?
    private let workerURL: URL?
    private var task: Task<Void, Never>?
    private var stopping = false
    private var interruptedID: UUID?
    var feedback: (any JobFeedback)?
    private var checkingEnvironment = false
    private var storageAvailable = true
    // Injection keeps queue tests independent from paid services and installed engine packages.
    typealias CommandBuilder = @MainActor (Job, URL, URL) throws -> ProcessCommand
    private let commandBuilder: CommandBuilder?

    init(store: JobStore = .applicationSupport, settings: AppSettings, workerURL: URL? = nil, commandBuilder: CommandBuilder? = nil) {
        self.workerURL = workerURL
        self.store = store
        self.settings = settings
        self.commandBuilder = commandBuilder
        do { jobs = try store.load(); try store.save(jobs) }
        catch {
            storageAvailable = false
            errorMessage = "無法讀取工作清單，已停止寫入以保留原始資料：\(error.localizedDescription)"
        }
    }

    @discardableResult func enqueue(source: JobSource, options: WorkerOptions, processing: ProcessingOptions = .init()) -> UUID? {
        guard storageAvailable, !stopping else { return nil }
        let job = Job(source: source, options: options, processing: processing)
        do {
            try processing.validate(provider: options.provider)
            try store.prepare(job.id)
            try store.save(jobs + [job])
            jobs.append(job)
            startNext()
            return job.id
        } catch { errorMessage = "無法建立工作：\(error.localizedDescription)"; return nil }
    }

    func cancel(_ id: UUID) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        if activeID == id {
            jobs[index].status = .cancelling
            jobs[index].stage = "正在結束程序，最長等待 10 秒"
            process?.cancel()
            downloader?.cancel()
            if checkingEnvironment { settings.cancelEnvironmentCheck() }
        } else if jobs[index].status == .queued {
            jobs[index].status = .cancelled
            jobs[index].stage = "已從佇列取消"
        }
        persist()
    }

    func resume(_ id: UUID) {
        guard !stopping, storageAvailable,
              let index = jobs.firstIndex(where: { $0.id == id }), jobs[index].status.canResume else { return }
        jobs[index].status = .queued
        jobs[index].progress = 0
        jobs[index].stage = "等待繼續"
        jobs[index].manifestPath = nil
        persist()
        startNext()
    }

    func shutdown() async {
        stopping = true
        if let id = activeID {
            // A user's explicit cancellation remains cancelled; quitting preserves resumable work.
            if jobs.first(where: { $0.id == id })?.status != .cancelling { interruptedID = id }
            cancel(id)
        }
        await task?.value
        for index in jobs.indices where jobs[index].status == .queued {
            jobs[index].status = .interrupted
            jobs[index].stage = "已中斷，可繼續；工作目錄與快取已保留"
        }
        persist()
    }

    private func persist() {
        guard storageAvailable else { return }
        feedback?.update(jobs: jobs)
        do { try store.save(jobs) }
        catch {
            storageAvailable = false
            errorMessage = "工作狀態無法儲存，佇列已停止：\(error.localizedDescription)"
            process?.cancel()
            downloader?.cancel()
            if checkingEnvironment { settings.cancelEnvironmentCheck() }
        }
    }

    private func startNext() {
        guard !stopping, storageAvailable, activeID == nil,
              let index = jobs.firstIndex(where: { $0.status == .queued }) else { return }
        let id = jobs[index].id
        activeID = id
        jobs[index].status = .running
        jobs[index].stage = "準備執行"
        persist()
        task = Task { await execute(id) }
    }

    private func mutate(_ id: UUID, _ update: (inout Job) -> Void) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        update(&jobs[index])
        persist()
    }

    private func checkCancellation(_ id: UUID) throws {
        if stopping || !storageAvailable || jobs.first(where: { $0.id == id })?.status == .cancelling {
            throw CancellationError()
        }
    }

    private func run(_ command: ProcessCommand, id: UUID, secrets: [String], stdout: (String) -> Void) async throws {
        try checkCancellation(id)
        let child = GroupProcess()
        process = child
        defer { process = nil }
        var code: Int32?
        var failure: String?
        for await event in child.run(command, secrets: secrets) {
            switch event {
            case .stdout(let line): stdout(line)
            case .stderr(let line): mutate(id) { Self.append(line, to: &$0) }
            case .failed(let message): failure = message
            case .exited(let value): code = value
            }
        }
        try checkCancellation(id)
        if let failure { throw AppError(message: failure) }
        guard code == 0 else { throw AppError(message: "引擎執行失敗（退出碼 \(code.map(String.init) ?? "未知")），請查看日誌。") }
    }

    private func execute(_ id: UUID) async {
        defer {
            process = nil
            downloader = nil
            activeID = nil
            startNext()
        }
        guard let job = jobs.first(where: { $0.id == id }) else { return }
        do {
            try checkCancellation(id)
            try job.request.validate(provider: job.options.provider)
            if job.request.overlay?.cover == .lama {
                mutate(id) { $0.stage = "檢查 LaMa 環境" }
                // Recheck at execution time as the interpreter may have changed since enqueue.
                while settings.checking {
                    try checkCancellation(id)
                    try await Task.sleep(for: .milliseconds(100))
                }
                checkingEnvironment = true
                await settings.checkEnvironment()
                checkingEnvironment = false
                try checkCancellation(id)
                guard settings.lamaAvailable else {
                    throw AppError(message: "LaMa 需要目前 Python 可匯入 torch 與 cv2，請先至設定檢查環境。")
                }
            }
            let output = outputDirectory(for: job)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            let input: String
            if commandBuilder != nil {
                input = job.source.location
            } else {
                switch job.source {
                case .local(let path): input = path
                case .youtube(let source):
                    let download = YouTubeDownloader()
                    downloader = download
                    mutate(id) { $0.stage = "下載 YouTube 影片" }
                    input = try await download.download(source: source, python: settings.pythonPath,
                        directory: store.directory(for: id).appendingPathComponent("download")) { line in
                        self.mutate(id) { Self.append(line, to: &$0); $0.stage = "下載 YouTube 影片" }
                    }.path
                    downloader = nil
                }
                guard FileManager.default.isReadableFile(atPath: input) else {
                    throw AppError(message: "無法讀取來源影片，請確認檔案仍在原位置。")
                }
            }
            try checkCancellation(id)
            var secrets: [String] = []
            var environment = AppSettings.childEnvironment()
            if commandBuilder == nil && job.options.provider != .local {
                let key = try Keychain.read(job.options.provider)
                guard !key.isEmpty else { throw AppError(message: "請先在設定儲存 \(job.options.provider.rawValue) API 金鑰。") }
                secrets.append(key)
                environment[job.options.provider == .openai ? "OPENAI_API_KEY" : "GEMINI_API_KEY"] = key
            }
            var artifacts: [JobArtifact] = []
            if job.request.dialogueEnabled {
                let optionsURL = try store.writeOptions(job)
                let command: ProcessCommand
                if let commandBuilder { command = try commandBuilder(job, optionsURL, output) }
                else {
                    guard let worker = workerURL ?? Bundle.main.url(forResource: "subtitle_worker", withExtension: "py") else {
                        throw AppError(message: "App 缺少 subtitle_worker.py，請重新建置。")
                    }
                    command = ProcessCommand(executable: settings.pythonPath,
                        arguments: [worker.path, "--input", input, "--options-json", optionsURL.path, "--output-dir", output.path], environment: environment)
                }
                var result: String?
                var engineError: String?
                try await run(command, id: id, secrets: secrets) { line in
                    switch WorkerEvent.parse(line) {
                    case .progress(let stage, let percent, let message):
                        self.mutate(id) {
                            $0.progress = job.request.overlay == nil ? percent : percent * 0.5
                            $0.stage = message
                            Self.append("[\(stage)] \(message)", to: &$0)
                        }
                    case .result(let path): result = path
                    case .error(let message): engineError = message; self.mutate(id) { Self.append(message, to: &$0) }
                    case .log(let line): self.mutate(id) { Self.append(line, to: &$0) }
                    }
                }
                if let engineError { throw AppError(message: engineError) }
                guard let result else { throw AppError(message: "引擎未回傳成品清單，無法確認完成。") }
                try Self.validateManifest(result, output: output, options: job.options)
                artifacts = try ArtifactManifest.load(result, output: output)
            }
            if let overlay = job.request.overlay {
                try checkCancellation(id)
                let directory = output.appendingPathComponent("overlay", isDirectory: true)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                let dialogue: URL?
                if overlay.useDialogue {
                    guard let subtitle = artifacts.first(where: { $0.kind == "srt" }) else {
                        throw AppError(message: "找不到同工作產生的對白字幕。")
                    }
                    dialogue = try ArtifactManifest.resolvedURL(subtitle, output: output)
                } else { dialogue = nil }
                let command = try OverlayPipeline.command(python: settings.pythonPath, input: URL(fileURLWithPath: input),
                    output: directory, options: job.options, overlay: overlay, dialogue: dialogue, environment: environment)
                mutate(id) { $0.stage = "準備畫面字翻譯" }
                try await run(command, id: id, secrets: secrets) { line in
                    self.mutate(id) {
                        Self.append(line, to: &$0)
                        if let progress = OverlayPipeline.progress(line) {
                            let value = job.request.dialogueEnabled ? 50 + progress.percent * 0.5 : progress.percent
                            $0.progress = max($0.progress, value)
                            $0.stage = progress.stage
                        }
                    }
                }
                artifacts += try OverlayPipeline.artifacts(in: directory, relativeTo: output)
            }
            try checkCancellation(id)
            let manifest = output.appendingPathComponent("manifest.json")
            try JSONEncoder().encode(ArtifactManifest(version: 1, artifacts: artifacts)).write(to: manifest, options: .atomic)
            _ = try ArtifactManifest.load(manifest.path, output: output)
            mutate(id) { $0.status = .completed; $0.progress = 100; $0.stage = "完成"; $0.manifestPath = manifest.path }
            if let completed = jobs.first(where: { $0.id == id }) { feedback?.finished(job: completed) }
        } catch is CancellationError {
            mutate(id) {
                $0.status = interruptedID == id ? .interrupted : .cancelled
                $0.stage = interruptedID == id ? "已中斷，可繼續；工作目錄與快取已保留" : "已取消"
            }
        } catch {
            mutate(id) { $0.status = .failed; $0.stage = error.localizedDescription; Self.append(error.localizedDescription, to: &$0) }
            if let failed = jobs.first(where: { $0.id == id }) { feedback?.finished(job: failed) }
        }
    }

    private static func append(_ message: String, to job: inout Job) {
        job.logs.append(String(message.prefix(16_384)))
        if job.logs.count > 2_000 { job.logs.removeFirst(job.logs.count - 2_000) }
    }

    func outputDirectory(for job: Job) -> URL {
        store.directory(for: job.id).appendingPathComponent("output", isDirectory: true)
    }

    static func validateManifest(_ path: String, output: URL, options: WorkerOptions? = nil) throws {
        let artifacts = try ArtifactManifest.load(path, output: output)
        if let options {
            var required: Set<String> = ["srt", "vtt"]
            if !options.skipCorrection { required.insert("corrected_srt") }
            if options.burnIn { required.insert("burned_video") }
            guard Set(artifacts.map(\.kind)).isSuperset(of: required) else {
                throw AppError(message: "成品清單缺少要求的字幕或燒錄影片。")
            }
        }
    }
}

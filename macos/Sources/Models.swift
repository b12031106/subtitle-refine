import Foundation

enum Provider: String, Codable, CaseIterable, Sendable {
    case local = "local-runtime", openai, gemini
}

enum SubtitleBackground: String, Codable, CaseIterable, Sendable { case none, box }

struct WorkerOptions: Codable, Equatable, Sendable {
    var provider: Provider = .local
    var model: String? = nil
    var targetLanguage: String? = "zh-TW"
    var sourceLanguage = "auto"
    var whisperModel = "base"
    var context = ""
    var burnIn = false
    var subtitleBackground: SubtitleBackground = .none
    var subtitleBackgroundOpacity = 80
    var skipCorrection = false
    var maxWorkers = 2

    enum CodingKeys: String, CodingKey {
        case provider, model, context
        case targetLanguage = "target_language", sourceLanguage = "source_language"
        case whisperModel = "whisper_model", burnIn = "burn_in"
        case subtitleBackground = "subtitle_background", subtitleBackgroundOpacity = "subtitle_background_opacity"
        case skipCorrection = "skip_correction", maxWorkers = "max_workers"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        provider = try c.decode(Provider.self, forKey: .provider)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        targetLanguage = try c.decodeIfPresent(String.self, forKey: .targetLanguage)
        sourceLanguage = try c.decode(String.self, forKey: .sourceLanguage)
        whisperModel = try c.decode(String.self, forKey: .whisperModel)
        context = try c.decode(String.self, forKey: .context)
        burnIn = try c.decode(Bool.self, forKey: .burnIn)
        skipCorrection = try c.decode(Bool.self, forKey: .skipCorrection)
        maxWorkers = try c.decode(Int.self, forKey: .maxWorkers)
        // Work created before engine 0.3.2 must remain readable and keep its old appearance.
        subtitleBackground = try c.decodeIfPresent(SubtitleBackground.self, forKey: .subtitleBackground) ?? .none
        subtitleBackgroundOpacity = try c.decodeIfPresent(Int.self, forKey: .subtitleBackgroundOpacity) ?? 80
        guard (0...100).contains(subtitleBackgroundOpacity) else {
            throw DecodingError.dataCorruptedError(forKey: .subtitleBackgroundOpacity, in: c, debugDescription: "字幕底色不透明度必須介於 0–100。")
        }
    }

    func encode(to encoder: Encoder) throws {
        guard (0...100).contains(subtitleBackgroundOpacity) else {
            throw EncodingError.invalidValue(subtitleBackgroundOpacity, .init(codingPath: encoder.codingPath, debugDescription: "字幕底色不透明度必須介於 0–100。"))
        }
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(provider, forKey: .provider)
        try c.encodeIfPresent(model?.nilIfBlank, forKey: .model)
        // Explicit null is required: omission would enable the engine's default translation.
        try c.encode(targetLanguage?.nilIfBlank, forKey: .targetLanguage)
        try c.encode(sourceLanguage, forKey: .sourceLanguage)
        try c.encode(whisperModel, forKey: .whisperModel)
        try c.encode(context, forKey: .context)
        try c.encode(burnIn, forKey: .burnIn)
        try c.encode(subtitleBackground, forKey: .subtitleBackground)
        try c.encode(subtitleBackgroundOpacity, forKey: .subtitleBackgroundOpacity)
        try c.encode(skipCorrection, forKey: .skipCorrection)
        try c.encode(maxWorkers, forKey: .maxWorkers)
    }
}

extension String {
    var nilIfBlank: String? {
        let value = trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }
}

enum JobSource: Codable, Equatable, Sendable {
    case local(String), youtube(String)
    var displayName: String {
        switch self {
        case .local(let path): URL(fileURLWithPath: path).lastPathComponent
        case .youtube(let url): url
        }
    }
    var location: String {
        switch self { case .local(let s), .youtube(let s): s }
    }
}

enum JobStatus: String, Codable, Sendable {
    case queued, running, cancelling, interrupted, cancelled, failed, completed
    var title: String {
        switch self {
        case .queued: "排隊中"
        case .running: "處理中"
        case .cancelling: "取消中"
        case .interrupted: "已中斷，可繼續"
        case .cancelled: "已取消"
        case .failed: "失敗"
        case .completed: "完成"
        }
    }
    var canResume: Bool { [.interrupted, .cancelled, .failed].contains(self) }
}

struct Job: Codable, Identifiable, Equatable, Sendable {
    var id = UUID()
    var createdAt = Date()
    var source: JobSource
    var options = WorkerOptions()
    var status: JobStatus = .queued
    var progress: Double = 0
    var stage = "等待開始"
    var logs: [String] = []
    var manifestPath: String?
    var processing: ProcessingOptions?
    var request: ProcessingOptions { processing ?? ProcessingOptions() }
}

struct JobStore: Sendable {
    let root: URL
    static var applicationSupport: JobStore {
        JobStore(root: FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SubtitleRefine", isDirectory: true))
    }
    func directory(for id: UUID) -> URL {
        root.appendingPathComponent("jobs/\(id.uuidString)", isDirectory: true)
    }
    func prepare(_ id: UUID) throws {
        try FileManager.default.createDirectory(at: directory(for: id), withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
    }
    func save(_ jobs: [Job]) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        try JSONEncoder().encode(jobs).write(to: root.appendingPathComponent("jobs.json"), options: .atomic)
    }
    func load() throws -> [Job] {
        let url = root.appendingPathComponent("jobs.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        return try JSONDecoder().decode([Job].self, from: Data(contentsOf: url)).map { original in
            var job = original
            // Nothing runs automatically after a relaunch, including jobs that were queued.
            if [.running, .cancelling, .queued].contains(job.status) {
                job.status = .interrupted
                job.stage = "上次執行已中斷，可繼續"
            }
            return job
        }
    }
    func writeOptions(_ job: Job) throws -> URL {
        try prepare(job.id)
        let url = directory(for: job.id).appendingPathComponent("options.json")
        var worker = job.options
        worker.sourceLanguage = LanguageCatalog.whisperCode(worker.sourceLanguage)
        try JSONEncoder().encode(worker).write(to: url, options: .atomic)
        return url
    }
}

import Foundation

@MainActor final class YouTubeDownloader {
    private var process: GroupProcess?
    private var cancelled = false
    func cancel() { cancelled = true; process?.cancel() }

    static func command(python: String, source: String, directory: URL, ffmpeg: String) -> ProcessCommand {
        ProcessCommand(executable: python, arguments: [
            "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-simulate", "--newline", "--no-color",
            "--continue", "--no-overwrites", "--restrict-filenames", "--ffmpeg-location", ffmpeg,
            "--format", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b", "--merge-output-format", "mp4",
            "--output", directory.appendingPathComponent("source.%(ext)s").path,
            "--print", "after_move:APP_FILE:%(filepath)j", "--progress", "--", source
        ], environment: AppSettings.childEnvironment())
    }

    func download(source: String, python: String, directory: URL,
                  onLine: @escaping (String) -> Void = { _ in }) async throws -> URL {
        guard NewJobView.validYouTubeURL(source) else { throw AppError(message: "請提供有效的 YouTube 影片連結。") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let receipt = directory.appendingPathComponent("download.json")
        if let data = try? Data(contentsOf: receipt), let cached = try? JSONDecoder().decode(Receipt.self, from: data),
           cached.source == source, let file = try? Self.validateFile(cached.path, directory: directory) {
            onLine("已重用下載完成的影片。")
            return file
        }
        let probe = ProcessCommand(executable: python, arguments: ["-c", "from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise as get; print('APP_FFMPEG:'+get()[0])"], environment: AppSettings.childEnvironment())
        var ffmpeg: String?
        try await run(probe) { line in
            if line.hasPrefix("APP_FFMPEG:") { ffmpeg = String(line.dropFirst(11)) }
            else { onLine(line) }
        }
        guard let ffmpeg else { throw AppError(message: "無法取得 static_ffmpeg 執行檔。") }
        var downloaded: String?
        try await run(Self.command(python: python, source: source, directory: directory, ffmpeg: ffmpeg)) { line in
            if line.hasPrefix("APP_FILE:"), let data = String(line.dropFirst(9)).data(using: .utf8) {
                downloaded = try? JSONDecoder().decode(String.self, from: data)
            } else { onLine(line) }
        }
        guard let downloaded else { throw AppError(message: "下載程序未回報完成檔案。") }
        let file = try Self.validateFile(downloaded, directory: directory)
        try JSONEncoder().encode(Receipt(source: source, path: file.path)).write(to: receipt, options: .atomic)
        return file
    }

    private func run(_ command: ProcessCommand, onLine: (String) -> Void) async throws {
        guard !cancelled else { throw CancellationError() }
        let child = GroupProcess()
        process = child
        defer { process = nil }
        var code: Int32?
        var failure: String?
        for await event in child.run(command) {
            switch event {
            case .stdout(let line), .stderr(let line): onLine(line)
            case .exited(let value): code = value
            case .failed(let message): failure = message
            }
        }
        guard !cancelled else { throw CancellationError() }
        if let failure { throw AppError(message: failure) }
        guard code == 0 else { throw AppError(message: "YouTube 下載失敗（退出碼 \(code.map(String.init) ?? "未知")），請查看日誌後重試。") }
    }

    static func validateFile(_ path: String, directory: URL) throws -> URL {
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let url = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
        guard url.path.hasPrefix(root.path + "/"), ["mp4", "mkv", "webm", "mov"].contains(url.pathExtension.lowercased()),
              let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              attrs[.type] as? FileAttributeType == .typeRegular, (attrs[.size] as? NSNumber)?.intValue ?? 0 > 0 else {
            throw AppError(message: "下載結果不是工作資料夾中的有效影片檔。")
        }
        return url
    }
    private struct Receipt: Codable { let source: String; let path: String }
}

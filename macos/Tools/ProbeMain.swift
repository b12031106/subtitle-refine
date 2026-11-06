import Foundation
import AppKit

@main struct ProbeMain {
    @MainActor static func main() async {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            guard let mode = args.first else { throw AppError(message: "Usage: SubtitleRefineProbe snapshots DIR | download URL DIR | e2e VIDEO DIR WORKER") }
            switch mode {
            case "snapshots":
                guard args.count == 2 else { throw AppError(message: "snapshots DIR") }
                _ = NSApplication.shared
                try SnapshotExporter.export(to: URL(fileURLWithPath: args[1]))
            case "download":
                guard args.count == 3 else { throw AppError(message: "download URL DIR") }
                let downloader = YouTubeDownloader()
                let file = try await downloader.download(source: args[1], python: AppSettings.defaultPython,
                    directory: URL(fileURLWithPath: args[2])) { print($0) }
                print("DOWNLOADED: " + file.path)
            case "open-review":
                guard args.count == 2 else { throw AppError(message: "open-review JOB_ROOT") }
                let root = URL(fileURLWithPath: args[1])
                let path = try String(contentsOf: root.appendingPathComponent("last-manifest-path.txt"), encoding: .utf8)
                let output = URL(fileURLWithPath: path).deletingLastPathComponent()
                let artifacts = try ArtifactManifest.load(path, output: output)
                guard let review = artifacts.first(where: { $0.isReview }) else { throw AppError(message: "Missing review") }
                try await ArtifactActions.open(review, output: output)
                print("REVIEW_OPENED: " + review.path)
            case "resume":
                guard args.count == 2 else { throw AppError(message: "resume JOB_ROOT") }
                let root = URL(fileURLWithPath: args[1])
                let idText = try String(contentsOf: root.appendingPathComponent("interrupted-job-id.txt"), encoding: .utf8)
                guard let id = UUID(uuidString: idText) else { throw AppError(message: "Invalid interrupted job ID") }
                let runner = JobRunner(store: JobStore(root: root), settings: AppSettings(pythonPath: AppSettings.defaultPython))
                guard let recovered = runner.jobs.first(where: { $0.id == id }), recovered.status == .interrupted,
                      runner.activeID == nil else { throw AppError(message: "Expected interrupted job on relaunch") }
                print("RECOVERED: " + recovered.status.title); fflush(nil)
                var shown = recovered.logs.count
                runner.resume(id)
                let deadline = Date().addingTimeInterval(3600)
                while runner.activeID != nil {
                    if let job = runner.jobs.first(where: { $0.id == id }) {
                        for line in job.logs.dropFirst(shown) { print(line); fflush(nil) }
                        shown = job.logs.count
                    }
                    if Date() > deadline { await runner.shutdown(); throw AppError(message: "Resume timed out") }
                    try await Task.sleep(for: .milliseconds(100))
                }
                guard let result = runner.jobs.first(where: { $0.id == id }), result.status == .completed,
                      let manifest = result.manifestPath else { throw AppError(message: runner.jobs.first?.stage ?? "Resume failed") }
                _ = try ArtifactManifest.load(manifest, output: runner.outputDirectory(for: result))
                try Data(manifest.utf8).write(to: root.appendingPathComponent("last-manifest-path.txt"))
                print("RESUMED_MANIFEST: " + manifest)
            case "overlay", "interrupt-overlay":
                guard args.count == 5, let cover = OverlayCover(rawValue: args[3]), let duration = Double(args[4]) else {
                    throw AppError(message: "overlay VIDEO DIR blur|lama DURATION")
                }
                let root = URL(fileURLWithPath: args[2])
                let runner = JobRunner(store: JobStore(root: root), settings: AppSettings(pythonPath: AppSettings.defaultPython))
                var options = WorkerOptions()
                options.sourceLanguage = "ko"
                options.context = "韓國綜藝節目 밥은영 EP.22，來賓 ITZY 彩領（채령）、主持人朴恩英（박은영）。請使用台灣繁體中文。"
                let overlay = OverlayOptions(cover: cover, targetLanguage: "zh-TW", start: 530, duration: duration)
                guard let id = runner.enqueue(source: .local(URL(fileURLWithPath: args[1]).path), options: options,
                    processing: ProcessingOptions(dialogueEnabled: false, overlay: overlay)) else {
                    throw AppError(message: runner.errorMessage ?? "Cannot enqueue")
                }
                var shown = 0
                let deadline = Date().addingTimeInterval(3600)
                while runner.activeID != nil {
                    if let job = runner.jobs.first(where: { $0.id == id }) {
                        for line in job.logs.dropFirst(shown) { print(line); fflush(nil) }
                        shown = job.logs.count
                    }
                    if mode == "interrupt-overlay", runner.jobs.first(where: { $0.id == id })?.logs.contains(where: { $0.contains("個文字事件") }) == true {
                        await runner.shutdown()
                        guard runner.jobs.first(where: { $0.id == id })?.status == .interrupted else { throw AppError(message: "Quit did not preserve interrupted state") }
                        try Data(id.uuidString.utf8).write(to: root.appendingPathComponent("interrupted-job-id.txt"))
                        print("INTERRUPTED: " + id.uuidString)
                        return
                    }
                    if Date() > deadline { await runner.shutdown(); throw AppError(message: "Overlay exceeded one hour") }
                    try await Task.sleep(for: .milliseconds(250))
                }
                guard let job = runner.jobs.first(where: { $0.id == id }), job.status == .completed, let manifest = job.manifestPath else {
                    throw AppError(message: runner.jobs.last?.stage ?? "Overlay failed")
                }
                let artifacts = try ArtifactManifest.load(manifest, output: runner.outputDirectory(for: job))
                guard Set(artifacts.map(\.kind)).isSuperset(of: ["overlay_video", "overlay_review", "overlay_ass", "overlay_events"]) else {
                    throw AppError(message: "Overlay artifacts incomplete")
                }
                try Data(manifest.utf8).write(to: root.appendingPathComponent("last-manifest-path.txt"), options: .atomic)
                print("MANIFEST: " + manifest)
                print(try String(contentsOfFile: manifest, encoding: .utf8))
            case "e2e":
                guard args.count == 4 else { throw AppError(message: "e2e VIDEO DIR WORKER") }
                let root = URL(fileURLWithPath: args[2])
                let store = JobStore(root: root)
                let runner = JobRunner(store: store, settings: AppSettings(pythonPath: AppSettings.defaultPython),
                                       workerURL: URL(fileURLWithPath: args[3]))
                var options = WorkerOptions()
                options.sourceLanguage = "en"
                options.targetLanguage = "zh-TW"
                options.burnIn = true
                options.context = "A short demonstration about automatic subtitles and translation. Preserve English during source correction."
                guard let id = runner.enqueue(source: .local(URL(fileURLWithPath: args[1]).path), options: options) else {
                    throw AppError(message: runner.errorMessage ?? "Cannot enqueue")
                }
                let deadline = Date().addingTimeInterval(900)
                var shown = 0
                while runner.activeID != nil {
                    if let job = runner.jobs.first(where: { $0.id == id }) {
                        for line in job.logs.dropFirst(shown) { print(line) }
                        shown = job.logs.count
                    }
                    if Date() > deadline {
                        await runner.shutdown()
                        throw AppError(message: "E2E exceeded 15 minutes")
                    }
                    try await Task.sleep(for: .milliseconds(200))
                }
                guard let job = runner.jobs.first(where: { $0.id == id }), job.status == .completed,
                      let path = job.manifestPath else {
                    throw AppError(message: runner.jobs.last?.stage ?? "Worker failed")
                }
                let artifacts = try ArtifactManifest.load(path, output: runner.outputDirectory(for: job))
                guard Set(artifacts.map(\.kind)).isSuperset(of: ["srt", "vtt", "corrected_srt", "burned_video"]) else {
                    throw AppError(message: "Missing required end-to-end artifacts")
                }
                try Data(path.utf8).write(to: root.appendingPathComponent("last-manifest-path.txt"), options: .atomic)
                print("MANIFEST: " + path)
                print(try String(contentsOfFile: path, encoding: .utf8))
            default: throw AppError(message: "Unknown mode: " + mode)
            }
        } catch {
            fputs("\(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}

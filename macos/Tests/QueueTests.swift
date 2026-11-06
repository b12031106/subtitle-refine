import XCTest

@MainActor final class QueueTests: XCTestCase {
    func testSerialQueueFailureContinuationAndResumeReuseDirectory() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = JobStore(root: root)
        var started: [UUID] = []
        var directories: [URL] = []
        let runner = JobRunner(store: store, settings: AppSettings()) { job, _, output in
            started.append(job.id)
            directories.append(output)
            return ProcessCommand(executable: AppSettings.defaultPython, arguments: ["-c", "import time; time.sleep(0.15); print('diagnostic'); raise SystemExit(3)"])
        }
        let first = try XCTUnwrap(runner.enqueue(source: .local("first.mp4"), options: WorkerOptions()))
        let second = try XCTUnwrap(runner.enqueue(source: .local("second.mp4"), options: WorkerOptions()))
        XCTAssertEqual(runner.jobs.map(\.status), [.running, .queued])
        try await waitUntil { runner.activeID == nil }
        XCTAssertEqual(started, [first, second])
        XCTAssertEqual(runner.jobs.map(\.status), [.failed, .failed])
        XCTAssertEqual(try store.load().map(\.status), [.failed, .failed])
        runner.resume(first)
        try await waitUntil { runner.activeID == nil }
        XCTAssertEqual(started, [first, second, first])
        XCTAssertEqual(directories.first, directories.last)
    }

    func testQueuedCancellationDoesNotLaunchAndShutdownStopsActiveJob() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        var started: [UUID] = []
        let runner = JobRunner(store: JobStore(root: root), settings: AppSettings()) { job, _, _ in
            started.append(job.id)
            return ProcessCommand(executable: AppSettings.defaultPython, arguments: ["-c", "import time; print('ready',flush=True); time.sleep(120)"])
        }
        let first = try XCTUnwrap(runner.enqueue(source: .local("one.mp4"), options: WorkerOptions()))
        let second = try XCTUnwrap(runner.enqueue(source: .local("two.mp4"), options: WorkerOptions()))
        runner.cancel(second)
        try await waitUntil { runner.jobs[0].logs.contains("ready") }
        await runner.shutdown()
        XCTAssertEqual(started, [first])
        XCTAssertNil(runner.activeID)
        XCTAssertEqual(runner.jobs.map(\.status), [.interrupted, .cancelled])
    }

    func testCorruptStoreIsPreserved() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let file = root.appendingPathComponent("jobs.json")
        let original = Data("corrupted data".utf8)
        try original.write(to: file)
        let runner = JobRunner(store: JobStore(root: root), settings: AppSettings())
        XCTAssertNotNil(runner.errorMessage)
        XCTAssertNil(runner.enqueue(source: .local("video.mp4"), options: WorkerOptions()))
        XCTAssertEqual(try Data(contentsOf: file), original)
    }

    func testSuccessfulFakeWorkerRequiresManifestAndArtifacts() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let runner = JobRunner(store: JobStore(root: root), settings: AppSettings()) { _, _, output in
            let script = """
            import pathlib,json,sys
            out=pathlib.Path(sys.argv[1])
            (out/'subtitle.srt').write_text('1\\n00:00:00,000 --> 00:00:01,000\\nHello\\n')
            manifest=out/'manifest.json'
            manifest.write_text(json.dumps({'version':1,'artifacts':[{'kind':kind,'path':'subtitle.srt'} for kind in ['srt','vtt','corrected_srt']]}))
            print(json.dumps({'type':'progress','stage':'completed','progress':100,'message':'完成'}))
            print(json.dumps({'type':'result','manifest':str(manifest)}))
            """
            return ProcessCommand(executable: AppSettings.defaultPython, arguments: ["-c", script, output.path])
        }
        _ = runner.enqueue(source: .local("fixture.mp4"), options: WorkerOptions())
        try await waitUntil { runner.activeID == nil }
        XCTAssertEqual(runner.jobs.first?.status, .completed)
        XCTAssertNotNil(runner.jobs.first?.manifestPath)
    }

    func testEnvironmentCheckReportsEveryDependency() async {
        let settings = AppSettings(pythonPath: AppSettings.defaultPython)
        await settings.checkEnvironment()
        XCTAssertFalse(settings.checking)
        for dependency in ["subtitle_engine", "whisper", "yt_dlp", "static_ffmpeg", "torch", "cv2"] {
            XCTAssertTrue(settings.environmentReport.contains(dependency + ":"), settings.environmentReport)
        }
        print("Environment probe:\n" + settings.environmentReport)
    }

    func testYouTubeURLValidation() {
        XCTAssertTrue(NewJobView.validYouTubeURL("https://youtu.be/abc"))
        XCTAssertTrue(NewJobView.validYouTubeURL("https://www.youtube.com/watch?v=abc"))
        XCTAssertFalse(NewJobView.validYouTubeURL("https://youtube.com.evil.invalid/watch?v=abc"))
        XCTAssertFalse(NewJobView.validYouTubeURL("file:///tmp/video"))
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<400 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(25))
        }
        XCTFail("Queue did not reach expected state within 10 seconds")
    }
}

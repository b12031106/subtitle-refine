import XCTest

@MainActor final class M2Tests: XCTestCase {
    func testDownloadCommandUsesArgumentArrayAndIgnoresExternalConfig() {
        let url = "https://www.youtube.com/watch?v=jNQXAC9IVRw&list=example"
        let command = YouTubeDownloader.command(python: "/custom/python", source: url,
            directory: URL(fileURLWithPath: "/tmp/a directory"), ffmpeg: "/tmp/ffmpeg")
        XCTAssertEqual(command.executable, "/custom/python")
        XCTAssertEqual(Array(command.arguments.prefix(2)), ["-m", "yt_dlp"])
        XCTAssertEqual(Array(command.arguments.suffix(2)), ["--", url])
        XCTAssertTrue(command.arguments.contains("--ignore-config"))
        XCTAssertTrue(command.arguments.contains("--no-playlist"))
        XCTAssertTrue(command.arguments.contains("/tmp/a directory/source.%(ext)s"))
        XCTAssertFalse(command.environment.keys.contains(where: { $0.contains("API_KEY") }))
    }

    func testDownloadValidationRejectsEscapesAndEmptyFiles() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let file = root.appendingPathComponent("source.mp4")
        try Data().write(to: file)
        XCTAssertThrowsError(try YouTubeDownloader.validateFile(file.path, directory: root))
        try Data("fake-video".utf8).write(to: file)
        XCTAssertEqual(try YouTubeDownloader.validateFile(file.path, directory: root), file)
        XCTAssertThrowsError(try YouTubeDownloader.validateFile(file.path, directory: root.appendingPathComponent("nested")))
        let link = root.appendingPathComponent("link.mp4")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: URL(fileURLWithPath: "/etc/hosts"))
        XCTAssertThrowsError(try YouTubeDownloader.validateFile(link.path, directory: root))
    }

    func testDownloaderAndWorkerPipelineAndCachedDownload() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let python = try fakePython(root: root)
        let store = JobStore(root: root.appendingPathComponent("store"))
        let runner = JobRunner(store: store, settings: AppSettings(pythonPath: python.path), workerURL: root.appendingPathComponent("worker.py"))
        let source = "https://youtu.be/fixture"
        let id = try XCTUnwrap(runner.enqueue(source: .youtube(source), options: WorkerOptions()))
        for _ in 0..<400 {
            if runner.activeID == nil { break }
            try await Task.sleep(for: .milliseconds(25))
        }
        XCTAssertNil(runner.activeID)
        let job = try XCTUnwrap(runner.jobs.first)
        XCTAssertEqual(job.status, .completed, job.stage)
        let download = store.directory(for: id).appendingPathComponent("download")
        XCTAssertTrue(job.logs.contains { $0.contains("WORKER_INPUT:") && $0.contains("/download/source.mp4") })
        let cached = try await YouTubeDownloader().download(source: source, python: "/does-not-exist", directory: download)
        XCTAssertEqual(cached.lastPathComponent, "source.mp4")
        XCTAssertEqual(try ArtifactManifest.load(try XCTUnwrap(job.manifestPath), output: runner.outputDirectory(for: job)).count, 3)
    }

    func testCancellingDownloadDoesNotStartWorker() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let python = try fakePython(root: root, slow: true)
        let runner = JobRunner(store: JobStore(root: root.appendingPathComponent("store")), settings: AppSettings(pythonPath: python.path), workerURL: root.appendingPathComponent("worker.py"))
        let id = try XCTUnwrap(runner.enqueue(source: .youtube("https://youtu.be/fixture"), options: WorkerOptions()))
        for _ in 0..<200 {
            if runner.jobs.first?.logs.contains("DOWNLOAD_READY") == true { break }
            try await Task.sleep(for: .milliseconds(25))
        }
        XCTAssertTrue(runner.jobs[0].logs.contains("DOWNLOAD_READY"))
        runner.cancel(id)
        await runner.shutdown()
        XCTAssertEqual(runner.jobs[0].status, .cancelled)
        XCTAssertFalse(runner.jobs[0].logs.contains { $0.contains("WORKER_INPUT:") })
    }

    func testManifestRequiresRequestedBurnedVideo() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("subtitle".utf8).write(to: root.appendingPathComponent("subtitle.srt"))
        let file = root.appendingPathComponent("manifest.json")
        try Data("{\"version\":1,\"artifacts\":[{\"kind\":\"srt\",\"path\":\"subtitle.srt\"}]}".utf8).write(to: file)
        var options = WorkerOptions()
        options.burnIn = true
        XCTAssertThrowsError(try JobRunner.validateManifest(file.path, output: root, options: options))
    }

    private func fakePython(root: URL, slow: Bool = false) throws -> URL {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let file = root.appendingPathComponent("fake python")
        let code = """
        #!\(AppSettings.defaultPython)
        import json,pathlib,sys,time
        if '-c' in sys.argv:
            print('APP_FFMPEG:/tmp/fake-ffmpeg',flush=True)
        elif '-m' in sys.argv:
            print('DOWNLOAD_READY',flush=True)
            time.sleep(\(slow ? 120 : 0))
            target=pathlib.Path(sys.argv[sys.argv.index('--output')+1].replace('%(ext)s','mp4'))
            target.write_bytes(b'fake video fixture')
            print('APP_FILE:'+json.dumps(str(target)),flush=True)
        else:
            print('WORKER_INPUT:'+sys.argv[sys.argv.index('--input')+1],flush=True)
            out=pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1])
            artifacts=[]
            for kind,name in [('srt','subtitle.srt'),('vtt','subtitle.vtt'),('corrected_srt','corrected.srt')]:
                (out/name).write_text('fixture')
                artifacts.append({'kind':kind,'path':name})
            manifest=out/'manifest.json'
            manifest.write_text(json.dumps({'version':1,'artifacts':artifacts}))
            print(json.dumps({'type':'result','manifest':str(manifest)}),flush=True)
        """
        try Data(code.utf8).write(to: file)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: file.path)
        return file
    }
}

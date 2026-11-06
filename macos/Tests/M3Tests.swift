import XCTest

@MainActor final class M3Tests: XCTestCase {
    func testLanguageCatalogAndWhisperOptionsKeepRegionOutOfWhisper() throws {
        XCTAssertEqual(LanguageCatalog.title("auto"), "自動辨識")
        XCTAssertEqual(LanguageCatalog.title("ko"), "韓文")
        XCTAssertEqual(LanguageCatalog.title("zh-TW"), "繁體中文（台灣）")
        XCTAssertTrue(LanguageCatalog.validCode("pt-BR", allowAuto: false))
        XCTAssertFalse(LanguageCatalog.validCode("auto", allowAuto: false))
        XCTAssertFalse(LanguageCatalog.validCode("../zh-TW", allowAuto: false))
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        var job = Job(source: .local("test.mp4"))
        job.options.sourceLanguage = "zh-TW"
        job.processing = ProcessingOptions(overlay: OverlayOptions())
        let store = JobStore(root: root)
        let file = try store.writeOptions(job)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        XCTAssertEqual(json["source_language"] as? String, "zh")
        XCTAssertNil(json["overlay"])
        XCTAssertNil(json["processing"])
        try store.save([job])
        XCTAssertEqual(try store.load()[0].options.sourceLanguage, "zh-TW")
        XCTAssertEqual(try store.load()[0].processing, job.processing)
        var old = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(job)) as? [String: Any])
        old.removeValue(forKey: "processing")
        XCTAssertTrue(try JSONDecoder().decode(Job.self, from: JSONSerialization.data(withJSONObject: old)).request.dialogueEnabled)
    }

    func testOverlayCommandRejectsGeminiAndInvalidRangesAndCarriesDialogue() throws {
        var options = WorkerOptions()
        options.sourceLanguage = "ko"
        options.model = "vision-model"
        options.context = "채령 = 彩領; $HOME remains literal"
        let overlay = OverlayOptions(cover: .lama, start: 530, duration: 6, useDialogue: true)
        let command = try OverlayPipeline.command(python: "/python", input: URL(fileURLWithPath: "/video with spaces.mp4"),
            output: URL(fileURLWithPath: "/output"), options: options, overlay: overlay,
            dialogue: URL(fileURLWithPath: "/job/output/subtitle.srt"), environment: [:])
        XCTAssertEqual(Array(command.arguments.prefix(3)), ["-m", "subtitle_engine.overlay", "/video with spaces.mp4"])
        XCTAssertTrue(command.arguments.contains("/job/output/subtitle.srt"))
        XCTAssertTrue(command.arguments.contains("ko-KR,en-US"))
        XCTAssertTrue(command.arguments.contains("530.0"))
        XCTAssertTrue(command.arguments.contains("6.0"))
        XCTAssertTrue(command.arguments.contains(options.context))
        XCTAssertEqual(command.environment["PYTHON_DOTENV_DISABLED"], "1")
        options.provider = .gemini
        XCTAssertThrowsError(try OverlayPipeline.command(python: "/python", input: URL(fileURLWithPath: "/video"), output: URL(fileURLWithPath: "/output"), options: options, overlay: overlay, dialogue: nil, environment: [:]))
        for invalid in [OverlayOptions(start: -1, duration: 6), OverlayOptions(start: 0, duration: 0), OverlayOptions(start: 0), OverlayOptions(start: .infinity, duration: 6), OverlayOptions(targetLanguage: "../../outside")] {
            XCTAssertThrowsError(try invalid.validate(provider: .local, dialogueEnabled: true))
        }
        XCTAssertThrowsError(try OverlayOptions(useDialogue: true).validate(provider: .local, dialogueEnabled: false))
        XCTAssertThrowsError(try ProcessingOptions(dialogueEnabled: false).validate(provider: .local))
    }

    func testOverlayProgressAndMissingReview() throws {
        XCTAssertEqual(OverlayPipeline.progress("🤖 判讀 13/13（42s）")?.percent, 65)
        XCTAssertEqual(OverlayPipeline.progress("🧽 LaMa 修補（mps）")?.stage, "LaMa 修補與合成")
        XCTAssertEqual(OverlayPipeline.progress("📝 產生驗收頁（原畫面／成品對照）")?.percent, 96)
        XCTAssertNil(OverlayPipeline.progress("diagnostic: preserve this line"))
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("overlay")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        for name in ["test.mp4", "test.ass", "test.json"] { try Data("fixture".utf8).write(to: output.appendingPathComponent(name)) }
        XCTAssertThrowsError(try OverlayPipeline.artifacts(in: output, relativeTo: root))
        try Data("<!doctype html><title>驗收</title>".utf8).write(to: output.appendingPathComponent("test.review.html"))
        let artifacts = try OverlayPipeline.artifacts(in: output, relativeTo: root)
        XCTAssertEqual(artifacts.count, 4)
        XCTAssertTrue(artifacts.contains { $0.isReview })
        XCTAssertTrue(artifacts.allSatisfy { $0.path.hasPrefix("overlay/") })
    }

    func testDialogueThenOverlayUsesSameJobsSRTAndPublishesAllArtifacts() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let python = try fakePython(root: root)
        let input = root.appendingPathComponent("input.mp4")
        try Data("fixture".utf8).write(to: input)
        let store = JobStore(root: root.appendingPathComponent("store"))
        let runner = JobRunner(store: store, settings: AppSettings(pythonPath: python.path), workerURL: root.appendingPathComponent("worker.py"))
        let processing = ProcessingOptions(overlay: OverlayOptions(start: 530, duration: 30, useDialogue: true))
        let id = try XCTUnwrap(runner.enqueue(source: .local(input.path), options: WorkerOptions(), processing: processing))
        try await waitUntil { runner.activeID == nil }
        let job = runner.jobs[0]
        XCTAssertEqual(job.status, .completed, job.logs.joined(separator: "\n"))
        let subtitle = store.directory(for: id).appendingPathComponent("output/subtitle.srt").path
        XCTAssertTrue(job.logs.contains("DIALOGUE:" + subtitle))
        XCTAssertTrue(job.logs.contains("diagnostic: preserve this line"))
        let artifacts = try ArtifactManifest.load(try XCTUnwrap(job.manifestPath), output: runner.outputDirectory(for: job))
        XCTAssertEqual(artifacts.count, 7)
        XCTAssertEqual(try store.load()[0].processing, processing)
    }

    func testOverlayCancelDoesNotPublishCompletion() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let python = try fakePython(root: root, slow: true)
        let input = root.appendingPathComponent("input.mp4")
        try Data("fixture".utf8).write(to: input)
        let runner = JobRunner(store: JobStore(root: root.appendingPathComponent("store")), settings: AppSettings(pythonPath: python.path))
        let id = try XCTUnwrap(runner.enqueue(source: .local(input.path), options: WorkerOptions(), processing: ProcessingOptions(dialogueEnabled: false, overlay: OverlayOptions())))
        try await waitUntil { runner.jobs[0].logs.contains("OVERLAY_READY") }
        runner.cancel(id)
        await runner.shutdown()
        XCTAssertEqual(runner.jobs[0].status, .cancelled)
        XCTAssertNil(runner.jobs[0].manifestPath)
    }

    func testLamaRequiresSuccessfulCurrentInterpreterProbe() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let python = try fakePython(root: root)
        let settings = AppSettings(pythonPath: python.path)
        XCTAssertFalse(settings.lamaAvailable)
        await settings.checkEnvironment()
        XCTAssertFalse(settings.lamaAvailable)
        XCTAssertEqual(settings.packageStatus["torch"], false)
        let runner = JobRunner(store: JobStore(root: root.appendingPathComponent("store")), settings: settings)
        _ = runner.enqueue(source: .local("unused.mp4"), options: WorkerOptions(), processing: ProcessingOptions(dialogueEnabled: false, overlay: OverlayOptions(cover: .lama)))
        try await waitUntil { runner.activeID == nil }
        XCTAssertEqual(runner.jobs[0].status, .failed)
        XCTAssertTrue(runner.jobs[0].stage.contains("LaMa 需要"))
        XCTAssertFalse(runner.jobs[0].logs.contains("OVERLAY_READY"))
    }

    private func waitUntil(_ predicate: () -> Bool) async throws {
        for _ in 0..<400 {
            if predicate() { return }
            try await Task.sleep(for: .milliseconds(25))
        }
        XCTFail("Timed out waiting for overlay job")
    }

    private func fakePython(root: URL, slow: Bool = false) throws -> URL {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let file = root.appendingPathComponent("fake python")
        let code = """
        #!\(AppSettings.defaultPython)
        import json,pathlib,sys,time
        if '-c' in sys.argv:
            print('torch: 無法匯入（選用）',flush=True)
            print('cv2: 可匯入',flush=True)
        elif '-m' in sys.argv:
            print('OVERLAY_READY',flush=True)
            time.sleep(\(slow ? 120 : 0))
            out=pathlib.Path(sys.argv[sys.argv.index('--out-dir')+1])
            if '--dialogue-srt' in sys.argv:
                srt=sys.argv[sys.argv.index('--dialogue-srt')+1]
                assert pathlib.Path(srt).read_text()=='fixture'
                print('DIALOGUE:'+srt,flush=True)
            print('🤖 判讀 2/2（1s）',flush=True)
            print('diagnostic: preserve this line',flush=True)
            for name in ['test.mp4','test.ass','test.json','test.review.html']:
                (out/name).write_text('fixture')
        else:
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

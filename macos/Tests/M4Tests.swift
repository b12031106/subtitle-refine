import XCTest
import Darwin

@MainActor final class M4Tests: XCTestCase {
    func testQuitRelaunchAndResumeActuallyReusesOverlayCache() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let input = root.appendingPathComponent("EP22.mp4")
        try Data("video fixture".utf8).write(to: input)
        let script = root.appendingPathComponent("fake-python")
        try Data("""
        #!\(AppSettings.defaultPython)
        import pathlib,sys,time,os,json
        out=pathlib.Path(sys.argv[sys.argv.index('--out-dir')+1])
        work=out/'EP22.530s+30s.onscreen-work'
        work.mkdir(exist_ok=True)
        cache=work/'ocr.json'
        args=work/'arguments.json'
        if not cache.exists():
            cache.write_text('preserve-completed-OCR')
            args.write_text(json.dumps(sys.argv[1:]))
            print('CACHE_READY:'+str(os.getpid()),flush=True)
            time.sleep(120)
        else:
            assert cache.read_text()=='preserve-completed-OCR'
            assert json.loads(args.read_text())==sys.argv[1:]
            print('CACHE_REUSED:'+str(work),flush=True)
            for name in ['EP22.mp4','EP22.ass','EP22.json','EP22.review.html']:
                (out/name).write_text('fixture')
        """.utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
        let store = JobStore(root: root.appendingPathComponent("store"))
        let settings = AppSettings(pythonPath: script.path)
        let first = JobRunner(store: store, settings: settings)
        let feedback = RecordingFeedback()
        first.feedback = feedback
        let request = ProcessingOptions(dialogueEnabled: false, overlay: OverlayOptions(start: 530, duration: 30))
        let id = try XCTUnwrap(first.enqueue(source: .local(input.path), options: WorkerOptions(), processing: request))
        try await waitUntil { first.jobs[0].logs.contains(where: { $0.hasPrefix("CACHE_READY:") }) }
        let pidLine = try XCTUnwrap(first.jobs[0].logs.first(where: { $0.hasPrefix("CACHE_READY:") }))
        let pid = try XCTUnwrap(Int32(pidLine.dropFirst("CACHE_READY:".count)))
        let queued = try XCTUnwrap(first.enqueue(source: .local(input.path), options: WorkerOptions(), processing: request))
        let output = first.outputDirectory(for: first.jobs[0])
        let cache = output.appendingPathComponent("overlay/EP22.530s+30s.onscreen-work/ocr.json")
        let before = try Data(contentsOf: cache)
        await first.shutdown()
        XCTAssertNil(first.activeID)
        XCTAssertEqual(first.jobs.map(\.status), [.interrupted, .interrupted])
        XCTAssertEqual(kill(pid, 0), -1, "Quit must wait for the child to exit before allowing a restart")
        XCTAssertTrue(feedback.notices.isEmpty, "Quitting is not a failure notification")
        let second = JobRunner(store: store, settings: settings)
        second.feedback = feedback
        XCTAssertEqual(second.jobs[0].status.title, "已中斷，可繼續")
        XCTAssertEqual(second.jobs[0].request, request)
        XCTAssertNil(second.activeID, "Relaunch must wait for an explicit resume")
        second.resume(id)
        try await waitUntil { second.activeID == nil }
        let resumed = second.jobs[0]
        XCTAssertEqual(resumed.id, id)
        XCTAssertEqual(second.outputDirectory(for: resumed), output)
        XCTAssertEqual(resumed.status, .completed, resumed.logs.joined(separator: "\n"))
        XCTAssertTrue(resumed.logs.contains(where: { $0.hasPrefix("CACHE_REUSED:") }))
        XCTAssertEqual(try Data(contentsOf: cache), before)
        XCTAssertEqual(try ArtifactManifest.load(try XCTUnwrap(resumed.manifestPath), output: output).count, 4)
        XCTAssertEqual(second.jobs.first(where: { $0.id == queued })?.status, .interrupted)
        XCTAssertEqual(feedback.notices.count, 1)
        XCTAssertEqual(feedback.notices.first?.title, "字幕工作完成")
        XCTAssertNil(feedback.progress.last!)
    }

    func testAbruptExitStoredStatesRecoverWithoutChangingCacheOrCompletedJobs() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = JobStore(root: root)
        let jobs = [JobStatus.running, .cancelling, .queued, .completed, .failed].map { status in
            var job = Job(source: .local("input.mp4")); job.status = status; job.progress = 52
            return job
        }
        try store.save(jobs)
        let runner = JobRunner(store: store, settings: AppSettings())
        XCTAssertEqual(runner.jobs.map(\.status), [.interrupted, .interrupted, .interrupted, .completed, .failed])
        XCTAssertEqual(runner.jobs.map(\.id), jobs.map(\.id))
        XCTAssertEqual(runner.jobs[0].progress, 52)
        XCTAssertNil(runner.activeID)
        XCTAssertTrue(runner.jobs[0].stage.contains("可繼續"))
    }

    func testDockAndNotificationStatesDoNotReportCancellationAsFailure() async throws {
        var active = Job(source: .local("private-video.mp4")); active.status = .running; active.progress = 52.9
        let queued = Job(source: .local("next.mp4"))
        XCTAssertEqual(DockProgress.current([active, queued]), DockProgress(percent: 52, queued: 1, cancelling: false))
        active.status = .cancelling
        XCTAssertEqual(DockProgress.current([active])?.label, "停止中")
        for status in [JobStatus.queued, .running, .cancelled, .interrupted] {
            active.status = status
            XCTAssertNil(CompletionNotice(job: active))
        }
        active.status = .failed
        active.stage = "secret diagnostic"
        XCTAssertNil(DockProgress.current([active]))
        let notice = try XCTUnwrap(CompletionNotice(job: active))
        XCTAssertFalse(notice.body.contains(active.stage))
        XCTAssertFalse(notice.body.contains(active.source.location))
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let feedback = RecordingFeedback()
        let runner = JobRunner(store: JobStore(root: root), settings: AppSettings()) { _, _, _ in
            ProcessCommand(executable: "/usr/bin/false", arguments: [])
        }
        runner.feedback = feedback
        _ = runner.enqueue(source: .local("fixture"), options: WorkerOptions())
        try await waitUntil { runner.activeID == nil }
        XCTAssertEqual(feedback.notices.map(\.title), ["字幕工作失敗"])
        XCTAssertNil(feedback.progress.last!)
    }

    func testFailureGuidanceOffersSpecificRecoveryAndKeepsRawDiagnosticsSeparate() {
        XCTAssertEqual(FailureGuidance.describe(stage: "退出碼 1", logs: ["ModuleNotFoundError: No module named whisper"]).title, "Python 環境尚未就緒")
        XCTAssertEqual(FailureGuidance.describe(stage: "HTTP 401", logs: []).title, "提供商驗證失敗")
        XCTAssertEqual(FailureGuidance.describe(stage: "429 rate limit", logs: []).title, "提供商暫時無法接受更多工作")
        XCTAssertEqual(FailureGuidance.describe(stage: "connection timed out", logs: []).title, "連線暫時中斷")
        XCTAssertEqual(FailureGuidance.describe(stage: "無法讀取來源影片", logs: []).title, "找不到來源檔案")
        XCTAssertTrue(FailureGuidance.describe(stage: "unknown", logs: []).recovery.contains("快取會保留"))
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<400 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(25))
        }
        XCTFail("Timed out waiting for restart fixture")
    }
}

@MainActor private final class RecordingFeedback: JobFeedback {
    var notices: [CompletionNotice] = []
    var progress: [DockProgress?] = []
    func update(jobs: [Job]) { progress.append(DockProgress.current(jobs)) }
    func finished(job: Job) { if let notice = CompletionNotice(job: job) { notices.append(notice) } }
}

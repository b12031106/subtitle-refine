import XCTest
import Darwin

final class ProcessTests: XCTestCase, @unchecked Sendable {
    let python = "/Users/justinhsu/.pyenv/versions/3.12.0/bin/python"

    func testFakePythonCancellationKillsEntireGroupAfterTenSeconds() async throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let script = root.appendingPathComponent("fake_python.py")
        // Grandchild ignores TERM; parent records TERM and reaps it after the escalation.
        try Data("""
        import os, signal, subprocess, sys, time
        signal.signal(signal.SIGTERM, lambda *_: print('parent received TERM', flush=True))
        code = "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('child '+str(os.getpid())+' '+str(os.getpgrp()),flush=True); time.sleep(120)"
        child = subprocess.Popen([sys.executable, '-c', code])
        print('parent '+str(os.getpid())+' '+str(os.getpgrp()), flush=True)
        child.wait()
        """.utf8).write(to: script)
        let process = GroupProcess()
        var pids: [pid_t] = []
        var groups: [pid_t] = []
        var began: Date?
        var sawTerm = false
        var code: Int32?
        let watchdog = Task {
            try? await Task.sleep(for: .seconds(20))
            if !Task.isCancelled { process.cancel() }
        }
        defer { watchdog.cancel() }
        for await event in process.run(ProcessCommand(executable: python, arguments: [script.path])) {
            switch event {
            case .stdout(let line):
                if line == "parent received TERM" { sawTerm = true }
                let parts = line.split(separator: " ")
                if parts.count == 3, ["parent", "child"].contains(String(parts[0])),
                   let pid = Int32(parts[1]), let group = Int32(parts[2]) {
                    pids.append(pid); groups.append(group)
                    if pids.count == 2 { began = Date(); process.cancel() }
                }
            case .exited(let value): code = value
            case .failed(let message): XCTFail(message)
            default: break
            }
        }
        XCTAssertEqual(pids.count, 2)
        XCTAssertEqual(Set(groups).count, 1)
        XCTAssertNotEqual(groups.first, getpgrp())
        XCTAssertTrue(sawTerm)
        XCTAssertEqual(code, 128 + SIGKILL)
        let elapsed = Date().timeIntervalSince(try XCTUnwrap(began))
        XCTAssertGreaterThanOrEqual(elapsed, 9.8)
        XCTAssertLessThan(elapsed, 15)
        // launchd reaps an orphaned grandchild asynchronously; allow a brief bounded interval.
        for _ in 0..<100 {
            if pids.allSatisfy({ kill($0, 0) == -1 && errno == ESRCH }) { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        for pid in pids { XCTAssertEqual(kill(pid, 0), -1); XCTAssertEqual(errno, ESRCH) }
    }

    func testStreamsRedactedOutputAndNonzeroExit() async {
        let process = GroupProcess()
        var lines: [String] = []
        var code: Int32?
        for await event in process.run(ProcessCommand(executable: python,
                    arguments: ["-c", "import sys; print('hello secret-fixture'); print('error secret-fixture',file=sys.stderr); sys.exit(7)"]),
                    secrets: ["secret-fixture"]) {
            switch event {
            case .stdout(let line), .stderr(let line): lines.append(line)
            case .exited(let value): code = value
            case .failed(let message): XCTFail(message)
            }
        }
        XCTAssertEqual(code, 7)
        XCTAssertEqual(lines.count, 2)
        XCTAssertTrue(lines.allSatisfy { !$0.contains("secret-fixture") && $0.contains("[已隱藏金鑰]") })
    }

    func testCancelBeforeSpawnAndInvalidExecutable() async {
        let cancelled = GroupProcess()
        cancelled.cancel()
        for await event in cancelled.run(ProcessCommand(executable: python, arguments: ["-c", "raise Exception('must not run')"])) {
            if case .exited(let code) = event { XCTAssertEqual(code, SIGTERM) }
            else { XCTFail("Pre-cancelled command started") }
        }
        let invalid = GroupProcess()
        var failed = false
        for await event in invalid.run(ProcessCommand(executable: "/nonexistent/python", arguments: [])) {
            if case .failed = event { failed = true }
        }
        XCTAssertTrue(failed)
    }
}

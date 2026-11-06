import XCTest
import Security

final class CoreTests: XCTestCase {
    func testJSONLChunkingUnicodeCRLFAndFinalLine() {
        let text = "{\"type\":\"progress\",\"stage\":\"transcribing\",\"progress\":5,\"message\":\"語音轉錄\"}\r\n{\"type\":\"result\",\"manifest\":\"/tmp/manifest.json\"}"
        var decoder = LineDecoder()
        var lines: [String] = []
        for byte in text.utf8 { lines += decoder.feed(Data([byte])) }
        lines += decoder.feed(Data(), final: true)
        XCTAssertEqual(lines.count, 2)
        XCTAssertEqual(WorkerEvent.parse(lines[0]), .progress(stage: "transcribing", percent: 5, message: "語音轉錄"))
        XCTAssertEqual(WorkerEvent.parse(lines[1]), .result("/tmp/manifest.json"))
    }

    func testJSONLMalformedUnknownErrorAndBounds() {
        XCTAssertEqual(WorkerEvent.parse("not JSON"), .log("not JSON"))
        XCTAssertEqual(WorkerEvent.parse("{\"type\":\"new\"}"), .log("{\"type\":\"new\"}"))
        XCTAssertEqual(WorkerEvent.parse("{\"type\":\"error\",\"message\":\"錯誤\"}"), .error("錯誤"))
        XCTAssertEqual(WorkerEvent.parse("{\"type\":\"progress\",\"stage\":\"x\",\"progress\":120}"),
                       .progress(stage: "x", percent: 100, message: "x"))
        var decoder = LineDecoder()
        XCTAssertEqual(decoder.feed(Data(repeating: 65, count: 1_048_577)), ["[略過超過 1 MiB 的日誌行]"])
        XCTAssertEqual(decoder.feed(Data("secret-suffix\nnext\n".utf8)), ["next"])
    }

    func testOptionsStrictSchemaNoCredentialsAndExplicitNull() throws {
        var options = WorkerOptions()
        options.targetLanguage = nil
        options.model = "  "
        let data = try JSONEncoder().encode(options)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(json.keys), Set(["provider", "target_language", "source_language", "whisper_model", "context", "burn_in", "skip_correction", "max_workers", "subtitle_background", "subtitle_background_opacity"]))
        XCTAssertTrue(json["target_language"] is NSNull)
        XCTAssertNil(json["api_key"])
        XCTAssertNil(json["command"])
        XCTAssertNil(json["input"])
        XCTAssertEqual(try JSONDecoder().decode(WorkerOptions.self, from: data).targetLanguage, nil)
        options.model = " gpt-test "
        let explicit = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(options)) as? [String: Any])
        XCTAssertEqual(explicit["model"] as? String, "gpt-test")
    }

    func testPersistenceRecoveryAndOptionsFile() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = JobStore(root: root)
        XCTAssertTrue(try store.load().isEmpty)
        var job = Job(source: .local("/tmp/測試.mp4"))
        job.status = .running
        job.logs = ["進度"]
        var finished = Job(source: .youtube("https://youtu.be/test"))
        finished.status = .completed
        try store.save([job, finished])
        let read = try store.load()
        XCTAssertEqual(read[0].id, job.id)
        XCTAssertEqual(read[0].status, .interrupted)
        XCTAssertEqual(read[0].logs, ["進度"])
        XCTAssertEqual(read[1], finished)
        let options = try store.writeOptions(job)
        XCTAssertTrue(options.path.contains(job.id.uuidString))
        XCTAssertEqual(try JSONDecoder().decode(WorkerOptions.self, from: Data(contentsOf: options)), job.options)
        try Data("broken".utf8).write(to: root.appendingPathComponent("jobs.json"))
        XCTAssertThrowsError(try store.load())
    }

    func testKeychainRoundTripInIsolatedService() throws {
        let service = "dev.subtitlerefine.tests.\(UUID().uuidString)"
        defer { try? Keychain.save("", for: .openai, service: service) }
        XCTAssertEqual(try Keychain.read(.openai, service: service), "")
        do {
            try Keychain.save("fixture-only", for: .openai, service: service)
        } catch let error as KeychainError where error.status == errSecInteractionNotAllowed {
            throw XCTSkip("目前 macOS 工作階段不允許 Keychain 互動（-25308）；未解鎖或更改使用者鑰匙圈。")
        }
        XCTAssertEqual(try Keychain.read(.openai, service: service), "fixture-only")
        try Keychain.save("fixture-updated", for: .openai, service: service)
        XCTAssertEqual(try Keychain.read(.openai, service: service), "fixture-updated")
        try Keychain.save("", for: .openai, service: service)
        XCTAssertEqual(try Keychain.read(.openai, service: service), "")
    }

    func testSecretRedaction() {
        let redactor = SecretRedactor(secrets: ["test-secret", "", "test-secret-long"])
        XCTAssertEqual(redactor.redact("test-secret-long / test-secret"), "[已隱藏金鑰] / [已隱藏金鑰]")
    }

    @MainActor func testManifestRejectsMissingFilesAndTraversal() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let manifest = root.appendingPathComponent("manifest.json")
        try Data("{\"version\":1,\"artifacts\":[{\"kind\":\"srt\",\"path\":\"a.srt\"}]}".utf8).write(to: manifest)
        XCTAssertThrowsError(try JobRunner.validateManifest(manifest.path, output: root))
        try Data("subtitle".utf8).write(to: root.appendingPathComponent("a.srt"))
        XCTAssertNoThrow(try JobRunner.validateManifest(manifest.path, output: root))
        try Data("{\"version\":1,\"artifacts\":[{\"kind\":\"srt\",\"path\":\"../outside.srt\"}]}".utf8).write(to: manifest)
        XCTAssertThrowsError(try JobRunner.validateManifest(manifest.path, output: root))
    }
}

func temporaryDirectory() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent("subtitle-refine-tests-\(UUID().uuidString)")
}

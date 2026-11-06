import XCTest

final class SubtitleBackgroundTests: XCTestCase {
    func testDefaultAndExplicitBackgroundOptionsRoundTrip() throws {
        var options = WorkerOptions()
        XCTAssertEqual(options.subtitleBackground, .none)
        XCTAssertEqual(options.subtitleBackgroundOpacity, 80)
        options.burnIn = true
        options.subtitleBackground = .box
        for value in [0, 60, 80, 100] {
            options.subtitleBackgroundOpacity = value
            let data = try JSONEncoder().encode(options)
            let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(json["subtitle_background"] as? String, "box")
            XCTAssertEqual(json["subtitle_background_opacity"] as? Int, value)
            XCTAssertEqual(try JSONDecoder().decode(WorkerOptions.self, from: data), options)
        }
        options.subtitleBackgroundOpacity = 101
        XCTAssertThrowsError(try JSONEncoder().encode(options))
    }

    func testOldPersistedJobLoadsWithNoBackgroundAndBadValuesAreRejected() throws {
        let job = Job(source: .local("lecture.mp4"))
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(job)) as? [String: Any])
        var options = try XCTUnwrap(json["options"] as? [String: Any])
        options.removeValue(forKey: "subtitle_background")
        options.removeValue(forKey: "subtitle_background_opacity")
        json["options"] = options
        let decoded = try JSONDecoder().decode(Job.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertEqual(decoded.id, job.id)
        XCTAssertEqual(decoded.options.subtitleBackground, .none)
        XCTAssertEqual(decoded.options.subtitleBackgroundOpacity, 80)
        for invalid: Any in [-1, 101, "60", true, 0.5] {
            var broken = options
            broken["subtitle_background_opacity"] = invalid
            XCTAssertThrowsError(try JSONDecoder().decode(WorkerOptions.self, from: JSONSerialization.data(withJSONObject: broken)))
        }
        options["subtitle_background"] = "unsupported"
        XCTAssertThrowsError(try JSONDecoder().decode(WorkerOptions.self, from: JSONSerialization.data(withJSONObject: options)))
    }

    func testJobStoreWritesBackgroundToWorkerOptions() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        var job = Job(source: .local("lecture.mp4"))
        job.options.burnIn = true
        job.options.subtitleBackground = .box
        job.options.subtitleBackgroundOpacity = 75
        let store = JobStore(root: root)
        let file = try store.writeOptions(job)
        let options = try JSONDecoder().decode(WorkerOptions.self, from: Data(contentsOf: file))
        XCTAssertEqual(options.subtitleBackground, .box)
        XCTAssertEqual(options.subtitleBackgroundOpacity, 75)
        try store.save([job])
        XCTAssertEqual(try store.load()[0].options, job.options)
    }
}

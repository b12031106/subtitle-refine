import XCTest
import AppKit
import Vision

@MainActor final class SnapshotTests: XCTestCase {
    func testSnapshotsContainActualContentWithoutUnsupportedViewMarkers() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try SnapshotExporter.export(to: root)
        let files = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        XCTAssertEqual(files.filter { $0.pathExtension == "png" }.count, 32)
        let required = ["empty": "開始製作字幕", "running": "translating", "completed": "QuickTime", "new-job": "Whisper", "settings": "API", "overlay-form": "LaMa", "overlay-running": "OCR", "overlay-completed": "驗收頁", "overlay-gemini": "Gemini", "overlay-lama-unavailable": "LaMa", "interrupted": "可繼續", "failed": "Python", "form-collapsed": "加入佇列", "form-compact": "加入佇列", "subtitle-bg-form": "80%", "subtitle-bg-completed": "半透明黑底"]
        for file in files {
            let bitmap = try XCTUnwrap(NSBitmapImageRep(data: Data(contentsOf: file)))
            XCTAssertGreaterThan(bitmap.pixelsWide, 1000)
            XCTAssertGreaterThan(bitmap.pixelsHigh, 700)
            var markers = 0
            for x in stride(from: 0, to: bitmap.pixelsWide, by: 8) {
                for y in stride(from: 0, to: bitmap.pixelsHigh, by: 8) {
                    guard let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { continue }
                    if color.redComponent > 0.95 && color.greenComponent > 0.65 && color.greenComponent < 0.9 && color.blueComponent < 0.05 { markers += 1 }
                }
            }
            XCTAssertLessThan(markers, 10, "ImageRenderer unsupported-view marker in \(file.lastPathComponent)")
            let request = VNRecognizeTextRequest()
            request.recognitionLanguages = ["zh-Hant", "en-US"]
            request.recognitionLevel = .accurate
            try VNImageRequestHandler(url: file).perform([request])
            let text = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ").replacingOccurrences(of: " ", with: "")
            let name = file.deletingPathExtension().lastPathComponent
            let screen = name.replacingOccurrences(of: "-light", with: "").replacingOccurrences(of: "-dark", with: "")
            if screen == "form-collapsed" {
                XCTAssertFalse(text.contains("Whisper"), "Disabled dialogue details should be collapsed")
                XCTAssertFalse(text.contains("LaMa"), "Disabled overlay details should be collapsed")
            }
            if screen == "form-compact" {
                XCTAssertEqual(bitmap.pixelsHigh, 1040)
                let footer = (request.results ?? []).first { $0.topCandidates(1).first?.string.replacingOccurrences(of: " ", with: "").contains("加入佇列") == true }
                XCTAssertLessThan(try XCTUnwrap(footer).boundingBox.midY, 0.15, "Actions stay at the bottom of a short form")
            }
            XCTAssertTrue(text.contains(try XCTUnwrap(required[screen])), "Missing main content in \(name): \(text)")
        }
    }
}

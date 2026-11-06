import Foundation
import AppKit

struct JobArtifact: Codable, Identifiable, Equatable, Sendable {
    var kind: String
    var path: String
    var id: String { kind + ":" + path }
    var title: String {
        switch kind {
        case "srt": "SRT 字幕"
        case "vtt": "WebVTT 字幕"
        case "corrected_srt": "校正字幕"
        case "burned_video": "已燒錄字幕影片"
        case "overlay_video": "畫面字翻譯影片"
        case "overlay_review": "畫面字驗收頁"
        case "overlay_ass": "畫面字 ASS 字幕"
        case "overlay_events": "畫面字事件資料"
        default: kind
        }
    }
    var isVideo: Bool { ["burned_video", "overlay_video"].contains(kind) }
    var isReview: Bool { kind == "overlay_review" }
}

struct ArtifactManifest: Codable {
    let version: Int
    let artifacts: [JobArtifact]

    static func load(_ path: String, output: URL) throws -> [JobArtifact] {
        let root = output.resolvingSymlinksInPath().standardizedFileURL
        let file = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
        guard file == root.appendingPathComponent("manifest.json") else {
            throw AppError(message: "成品清單位置不正確。")
        }
        let manifest = try JSONDecoder().decode(Self.self, from: Data(contentsOf: file))
        guard manifest.version == 1, !manifest.artifacts.isEmpty,
              Set(manifest.artifacts.map(\.id)).count == manifest.artifacts.count else {
            throw AppError(message: "成品清單格式不完整。")
        }
        for artifact in manifest.artifacts { _ = try resolvedURL(artifact, output: root) }
        return manifest.artifacts
    }

    static func resolvedURL(_ artifact: JobArtifact, output: URL) throws -> URL {
        let root = output.resolvingSymlinksInPath().standardizedFileURL
        let url = root.appendingPathComponent(artifact.path).resolvingSymlinksInPath().standardizedFileURL
        guard !artifact.path.hasPrefix("/"), url.path.hasPrefix(root.path + "/"),
              let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              attributes[.type] as? FileAttributeType == .typeRegular,
              (attributes[.size] as? NSNumber)?.intValue ?? 0 > 0 else {
            throw AppError(message: "成品檔案缺少、空白或位置不正確。")
        }
        return url
    }
}

@MainActor enum ArtifactActions {
    static func reveal(_ artifact: JobArtifact, output: URL) throws {
        NSWorkspace.shared.activateFileViewerSelecting([try ArtifactManifest.resolvedURL(artifact, output: output)])
    }
    static func open(_ artifact: JobArtifact, output: URL) async throws {
        let url = try ArtifactManifest.resolvedURL(artifact, output: output)
        if artifact.isReview {
            guard let browser = NSWorkspace.shared.urlForApplication(toOpen: URL(string: "https://example.com")!) else {
                throw AppError(message: "找不到預設瀏覽器，請先設定瀏覽器後再開啟驗收頁。")
            }
            _ = try await NSWorkspace.shared.open([url], withApplicationAt: browser, configuration: .init())
        } else if artifact.isVideo {
            guard let app = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.QuickTimePlayerX") else {
                throw AppError(message: "找不到 QuickTime Player，請從 Finder 開啟影片。")
            }
            _ = try await NSWorkspace.shared.open([url], withApplicationAt: app, configuration: .init())
        } else if !NSWorkspace.shared.open(url) {
            throw AppError(message: "無法開啟成品檔案，請從 Finder 選擇適合的 App。")
        }
    }
}

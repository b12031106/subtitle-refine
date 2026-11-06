import Foundation

enum WorkerEvent: Equatable, Sendable {
    case progress(stage: String, percent: Double, message: String)
    case result(String)
    case error(String)
    case log(String)

    static func parse(_ line: String) -> WorkerEvent {
        guard let data = line.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return .log(line) }
        switch type {
        case "progress":
            guard let stage = json["stage"] as? String, let value = json["progress"] as? Double,
                  value.isFinite else { return .log(line) }
            return .progress(stage: stage, percent: min(100, max(0, value)), message: json["message"] as? String ?? stage)
        case "result":
            guard let path = json["manifest"] as? String else { return .log(line) }
            return .result(path)
        case "error": return .error(json["message"] as? String ?? "引擎回報錯誤")
        default: return .log(line)
        }
    }
}

/// Buffers bytes (not Strings), so split UTF-8 characters survive pipe boundaries.
struct LineDecoder {
    private var pending = Data()
    private var discardingOversizedLine = false
    mutating func feed(_ data: Data, final: Bool = false) -> [String] {
        pending.append(data)
        var lines: [String] = []
        while let newline = pending.firstIndex(of: 10) {
            if !discardingOversizedLine {
                if pending.distance(from: pending.startIndex, to: newline) <= 1_048_576 {
                    lines.append(String(decoding: pending[..<newline], as: UTF8.self).trimmingCharacters(in: .newlines))
                } else {
                    lines.append("[略過超過 1 MiB 的日誌行]")
                }
            }
            discardingOversizedLine = false
            pending.removeSubrange(...newline)
        }
        // Bound malformed, newline-free output; never expose a partial secret to the logger.
        if pending.count > 1_048_576 {
            pending.removeAll()
            if !discardingOversizedLine { lines.append("[略過超過 1 MiB 的日誌行]") }
            discardingOversizedLine = true
        }
        if final, !pending.isEmpty {
            if !discardingOversizedLine { lines.append(String(decoding: pending, as: UTF8.self)) }
            pending.removeAll()
        }
        return lines.filter { !$0.isEmpty }
    }
}

struct SecretRedactor: Sendable {
    var secrets: [String]
    func redact(_ text: String) -> String {
        secrets.filter { !$0.isEmpty }.sorted { $0.count > $1.count }
            .reduce(text) { $0.replacingOccurrences(of: $1, with: "[已隱藏金鑰]") }
    }
}

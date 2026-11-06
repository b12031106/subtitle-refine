import Foundation

enum OverlayCover: String, Codable, CaseIterable, Sendable {
    case blur, lama
    var title: String { self == .blur ? "模糊遮蓋（blur）" : "LaMa 修補" }
}

struct OverlayOptions: Codable, Equatable, Sendable {
    var cover: OverlayCover = .blur
    var targetLanguage = "zh-TW"
    var start: Double?
    var duration: Double?
    var useDialogue = false

    func validate(provider: Provider, dialogueEnabled: Bool) throws {
        guard provider != .gemini else { throw AppError(message: "畫面字翻譯僅支援 local-runtime 與 OpenAI；Gemini 只可用於對白字幕。") }
        guard LanguageCatalog.validCode(targetLanguage, allowAuto: false) else { throw AppError(message: "請選擇畫面字的目標語言，或輸入有效語言代碼。") }
        if start != nil || duration != nil {
            guard let start, let duration, start.isFinite, duration.isFinite, start >= 0, duration > 0 else {
                throw AppError(message: "片段起始秒數須大於或等於 0，長度須大於 0，兩者皆須填寫。")
            }
        }
        guard !useDialogue || dialogueEnabled else { throw AppError(message: "帶入同工作對白字幕時，必須同時產生對白字幕。") }
    }
}

struct ProcessingOptions: Codable, Equatable, Sendable {
    var dialogueEnabled = true
    var overlay: OverlayOptions?
    func validate(provider: Provider) throws {
        guard dialogueEnabled || overlay != nil else { throw AppError(message: "請至少選擇對白字幕或畫面字翻譯。") }
        try overlay?.validate(provider: provider, dialogueEnabled: dialogueEnabled)
    }
}

enum LanguageCatalog {
    static let common: [(code: String, title: String)] = [
        ("en", "英文"), ("ja", "日文"), ("ko", "韓文"), ("zh-TW", "繁體中文（台灣）"),
        ("zh-CN", "簡體中文"), ("es", "西班牙文"), ("fr", "法文"), ("de", "德文"),
        ("th", "泰文"), ("vi", "越南文"), ("id", "印尼文")
    ]
    static func title(_ code: String) -> String {
        if code == "auto" { return "自動辨識" }
        if code == "zh" { return "中文" }
        return common.first { $0.code == code }?.title ?? "自訂（\(code)）"
    }
    static func validCode(_ code: String, allowAuto: Bool) -> Bool {
        if code == "auto" { return allowAuto }
        return code.range(of: "^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$", options: .regularExpression) != nil
    }
    static func whisperCode(_ code: String) -> String {
        // Whisper accepts ISO language names/codes, not Chinese region tags.
        ["zh-TW", "zh-CN", "zh-Hant", "zh-Hans"].contains(code) ? "zh" : code
    }
    static func ocrLanguages(_ code: String) -> String? {
        switch code {
        case "en": "en-US"
        case "ja": "ja-JP,en-US"
        case "ko": "ko-KR,en-US"
        case "zh-TW", "zh-Hant": "zh-Hant,en-US"
        case "zh-CN", "zh-Hans", "zh": "zh-Hans,en-US"
        default: nil // Let the engine use its default for auto or unsupported Vision locales.
        }
    }
}

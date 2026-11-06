import Foundation
import Security
import SwiftUI

enum Keychain {
    static let service = "dev.subtitlerefine.mac.api-keys"
    static func read(_ provider: Provider, service: String = Keychain.service) throws -> String {
        var query = base(provider, service: service)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        if status == errSecItemNotFound { return "" }
        guard status == errSecSuccess, let data = item as? Data,
              let value = String(data: data, encoding: .utf8) else { throw KeychainError(status: status) }
        return value
    }
    static func save(_ value: String, for provider: Provider, service: String = Keychain.service) throws {
        let query = base(provider, service: service)
        if value.isEmpty {
            let status = SecItemDelete(query as CFDictionary)
            guard status == errSecSuccess || status == errSecItemNotFound else { throw KeychainError(status: status) }
            return
        }
        let changes = [kSecValueData as String: Data(value.utf8)]
        var status = SecItemUpdate(query as CFDictionary, changes as CFDictionary)
        if status == errSecItemNotFound {
            var item = query.merging(changes) { _, new in new }
            item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            status = SecItemAdd(item as CFDictionary, nil)
        }
        guard status == errSecSuccess else { throw KeychainError(status: status) }
    }
    private static func base(_ provider: Provider, service: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
         kSecAttrAccount as String: provider.rawValue]
    }
}

struct KeychainError: LocalizedError {
    let status: OSStatus
    var errorDescription: String? { "Keychain 存取失敗（\(status)）。請確認登入鑰匙圈已解鎖。" }
}

@MainActor @Observable final class AppSettings {
    var notificationsEnabled: Bool {
        didSet { UserDefaults.standard.set(notificationsEnabled, forKey: "notificationsEnabled") }
    }
    var notificationStatus = "首次開始工作時會詢問通知權限；不允許仍可正常處理影片。"
    static let defaultPython = "/Users/justinhsu/.pyenv/versions/3.12.0/bin/python"
    var pythonPath: String {
        didSet {
            UserDefaults.standard.set(pythonPath, forKey: "pythonPath")
            checkedPython = nil
            packageStatus = [:]
        }
    }
    var environmentReport = "尚未檢查環境"
    var checking = false
    var checkedPython: String?
    var packageStatus: [String: Bool] = [:]
    var lamaAvailable: Bool {
        checkedPython == pythonPath && packageStatus["torch"] == true && packageStatus["cv2"] == true
    }
    private var checker: GroupProcess?
    init(pythonPath: String? = nil) {
        self.notificationsEnabled = UserDefaults.standard.object(forKey: "notificationsEnabled") as? Bool ?? true
        self.pythonPath = pythonPath ?? UserDefaults.standard.string(forKey: "pythonPath") ?? Self.defaultPython
    }

    static func childEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        for name in environment.keys where name.contains("API_KEY") || name.contains("TOKEN") || name.contains("SECRET") {
            environment.removeValue(forKey: name)
        }
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        return environment
    }

    func cancelEnvironmentCheck() { checker?.cancel() }

    func checkEnvironment() async {
        guard !checking else { return }
        checking = true
        let checkedPath = pythonPath
        packageStatus = [:]
        checkedPython = nil
        environmentReport = "正在檢查 Python 與套件…"
        let process = GroupProcess()
        checker = process
        let script = """
        import importlib, importlib.metadata
        for name, dist, optional in [('subtitle_engine','subtitle-engine',False),('whisper','openai-whisper',False),('yt_dlp','yt-dlp',False),('static_ffmpeg','static-ffmpeg',False),('torch','torch',True),('cv2','opencv-python',True)]:
            try:
                importlib.import_module(name)
                try: version = importlib.metadata.version(dist)
                except importlib.metadata.PackageNotFoundError: version = '可匯入'
                print(name + ': ' + version, flush=True)
            except Exception:
                print(name + ': 無法匯入' + ('（選用）' if optional else '（必要）'), flush=True)
        """
        var lines: [String] = []
        var success = false
        // A broken third-party import must not hang the settings window indefinitely.
        let timeout = Task {
            try? await Task.sleep(for: .seconds(60))
            if !Task.isCancelled { process.cancel() }
        }
        for await event in process.run(ProcessCommand(executable: checkedPath, arguments: ["-c", script],
                                                     environment: Self.childEnvironment())) {
            switch event {
            case .stdout(let line):
                lines.append(line)
                if let name = line.split(separator: ":").first {
                    packageStatus[String(name)] = !line.contains("無法匯入")
                }
            case .stderr: break // Import diagnostics can include arbitrary environment data.
            case .failed(let message): lines.append(message)
            case .exited(let code):
                success = code == 0
                if code != 0 { lines.append("檢查程序未完成（退出碼 \(code)）") }
            }
            environmentReport = lines.joined(separator: "\n")
        }
        timeout.cancel()
        checker = nil
        checkedPython = success ? checkedPath : nil
        checking = false
    }
}

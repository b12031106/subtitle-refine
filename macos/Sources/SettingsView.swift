import SwiftUI

struct SettingsView: View {
    @Bindable var settings: AppSettings
    @State private var openAIKey = ""
    @State private var geminiKey = ""
    @State private var notice = "金鑰儲存在 macOS Keychain，欄位留空不會變更現有金鑰。"

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            FormSection("Python 環境") {
                FormTextField(title: "Python 執行檔路徑", text: $settings.pythonPath)
                HStack {
                    Button(settings.checking ? "檢查中…" : "檢查環境") { Task { await settings.checkEnvironment() } }
                        .disabled(settings.checking)
                    Text("需安裝 subtitle-engine 與必要套件").font(.caption).foregroundStyle(.secondary)
                }
                Text(settings.environmentReport).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
            }
            FormSection("API 金鑰") {
                Text("local-runtime 使用本機 Runtime，不需要 API 金鑰。").font(.callout)
                FormTextField(title: "OpenAI API 金鑰", text: $openAIKey, secure: true)
                HStack {
                    Button("儲存 OpenAI 金鑰") { save(openAIKey, provider: .openai) }.disabled(openAIKey.nilIfBlank == nil)
                    Button("刪除 OpenAI 金鑰", role: .destructive) { save("", provider: .openai) }
                }
                FormTextField(title: "Gemini API 金鑰", text: $geminiKey, secure: true)
                HStack {
                    Button("儲存 Gemini 金鑰") { save(geminiKey, provider: .gemini) }.disabled(geminiKey.nilIfBlank == nil)
                    Button("刪除 Gemini 金鑰", role: .destructive) { save("", provider: .gemini) }
                }
                Text(notice).font(.caption).foregroundStyle(.secondary)
            }
            FormSection("通知") {
                FormToggle(title: "工作完成或失敗時通知我", isOn: $settings.notificationsEnabled)
                Text(settings.notificationStatus).font(.caption).foregroundStyle(.secondary)
            }
        }.padding(24).frame(width: 650, height: 780)
    }
    private func save(_ value: String, provider: Provider) {
        do {
            try Keychain.save(value.trimmingCharacters(in: .whitespacesAndNewlines), for: provider)
            if provider == .openai { openAIKey = "" } else { geminiKey = "" }
            notice = value.isEmpty ? "已刪除 \(provider.rawValue) 金鑰。" : "已儲存 \(provider.rawValue) 金鑰。"
        } catch { notice = error.localizedDescription }
    }
}

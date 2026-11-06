import SwiftUI

struct LanguagePicker: View {
    @Environment(\.snapshotMode) private var snapshot
    var title: String
    @Binding var code: String
    var allowAuto = false
    @State private var custom = false

    private var isKnown: Bool { (allowAuto && code == "auto") || LanguageCatalog.common.contains { $0.code == code } }
    private var isCustom: Bool { custom || !isKnown }
    private var selection: Binding<String> {
        Binding(get: { isCustom ? "custom" : code }, set: {
            custom = $0 == "custom"
            code = custom ? "" : $0
        })
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if snapshot {
                HStack { Text(title); Spacer(); Text(isCustom ? "自行輸入語言代碼" : LanguageCatalog.title(code)); Image(systemName: "chevron.down").font(.caption) }
                    .padding(8).background(.background, in: RoundedRectangle(cornerRadius: 6))
            } else {
                Picker(title, selection: selection) {
                    if allowAuto { Text("自動辨識").tag("auto") }
                    ForEach(LanguageCatalog.common, id: \.code) { language in Text(language.title).tag(language.code) }
                    Text("自行輸入語言代碼…").tag("custom")
                }
            }
            if isCustom { FormTextField(title: "自訂語言代碼（例如 pt-BR）", text: $code) }
        }
    }
}

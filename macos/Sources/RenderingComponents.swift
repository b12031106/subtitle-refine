import SwiftUI

private struct SnapshotModeKey: EnvironmentKey { static let defaultValue = false }
extension EnvironmentValues {
    var snapshotMode: Bool {
        get { self[SnapshotModeKey.self] }
        set { self[SnapshotModeKey.self] = newValue }
    }
}

/// ImageRenderer cannot draw AppKit scroll containers and text-input backing views.
/// The content/layout stays shared; snapshots substitute read-only values only for those controls.
struct RenderableScroll<Content: View>: View {
    @Environment(\.snapshotMode) private var snapshot
    @ViewBuilder var content: () -> Content
    var body: some View {
        if snapshot {
            // Match a real ScrollView's zero minimum height; oversized content must not
            // push the form header/footer outside a short window in ImageRenderer.
            GeometryReader { geometry in
                content().frame(width: geometry.size.width, alignment: .topLeading)
                    .fixedSize(horizontal: false, vertical: true)
            }.clipped()
        }
        else { ScrollView { content() } }
    }
}

struct FormSection<Content: View>: View {
    var title: String
    @ViewBuilder var content: () -> Content
    init(_ title: String, @ViewBuilder content: @escaping () -> Content) {
        self.title = title; self.content = content
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(.headline)
            content()
        }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 12))
    }
}

struct FormTextField: View {
    @Environment(\.snapshotMode) private var snapshot
    var title: String
    @Binding var text: String
    var secure = false
    var multiline = false
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            if snapshot {
                Text(text.isEmpty ? "尚未填寫" : secure ? "••••••••" : text)
                    .foregroundStyle(text.isEmpty ? .secondary : .primary)
                    .frame(maxWidth: .infinity, minHeight: multiline ? 42 : 20, alignment: .leading)
                    .padding(.horizontal, 8).padding(.vertical, 5)
                    .background(.background, in: RoundedRectangle(cornerRadius: 5))
                    .overlay(RoundedRectangle(cornerRadius: 5).stroke(.secondary.opacity(0.25)))
            } else if secure {
                SecureField(title, text: $text).labelsHidden().textFieldStyle(.roundedBorder)
            } else {
                TextField(title, text: $text, axis: multiline ? .vertical : .horizontal)
                    .labelsHidden().textFieldStyle(.roundedBorder).lineLimit(multiline ? 2...3 : 1...1)
            }
        }
    }
}

struct FormChoice: View {
    @Environment(\.snapshotMode) private var snapshot
    var title: String
    @Binding var selection: String
    var values: [String]
    var body: some View {
        if snapshot {
            HStack {
                Text(title)
                Spacer()
                Text(selection)
                Image(systemName: "chevron.up.chevron.down").font(.caption2)
            }.padding(8).background(.background, in: RoundedRectangle(cornerRadius: 5))
        } else {
            Picker(title, selection: $selection) { ForEach(values, id: \.self) { Text($0).tag($0) } }
        }
    }
}

struct FormToggle: View {
    @Environment(\.snapshotMode) private var snapshot
    var title: String
    @Binding var isOn: Bool
    var body: some View {
        if snapshot {
            HStack {
                Image(systemName: isOn ? "checkmark.square.fill" : "square")
                    .foregroundStyle(isOn ? Color.accentColor : .secondary)
                Text(title)
            }
        } else { Toggle(title, isOn: $isOn) }
    }
}

struct JobProgressView: View {
    var value: Double
    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(.secondary.opacity(0.16))
                Capsule().fill(Color.accentColor).frame(width: geometry.size.width * min(1, max(0, value / 100)))
            }
        }.frame(height: 6).accessibilityLabel("工作進度").accessibilityValue("\(Int(value))%")
    }
}

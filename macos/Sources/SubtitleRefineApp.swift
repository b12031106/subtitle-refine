import SwiftUI
import AppKit

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate {
    var runner: JobRunner?
    var feedback: DesktopFeedback?
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let runner else { return .terminateNow }
        Task {
            await runner.shutdown()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

@main struct SubtitleRefineApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var settings: AppSettings
    @State private var runner: JobRunner

    init() {
        let settings = AppSettings()
        _settings = State(initialValue: settings)
        let store: JobStore
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
            store = JobStore(root: FileManager.default.temporaryDirectory.appendingPathComponent("SubtitleRefine-test-host-\(UUID().uuidString)"))
        } else {
            store = .applicationSupport
        }
        _runner = State(initialValue: JobRunner(store: store, settings: settings))
    }
    var body: some Scene {
        WindowGroup {
            ContentView(runner: runner)
                .onAppear {
                    delegate.runner = runner
                    if delegate.feedback == nil {
                        let feedback = DesktopFeedback(settings: settings)
                        delegate.feedback = feedback
                        runner.feedback = feedback
                        feedback.update(jobs: runner.jobs)
                    }
                }
                .frame(minWidth: 900, minHeight: 600)
        }
        .defaultSize(width: 1080, height: 720)
        Settings { SettingsView(settings: settings) }
    }
}


import AppKit
import UserNotifications

@MainActor protocol JobFeedback: AnyObject {
    func update(jobs: [Job])
    func finished(job: Job)
}

struct DockProgress: Equatable {
    let percent: Int
    let queued: Int
    let cancelling: Bool
    static func current(_ jobs: [Job]) -> Self? {
        guard let active = jobs.first(where: { [.running, .cancelling].contains($0.status) }) else { return nil }
        return Self(percent: Int(min(100, max(0, active.progress))),
                    queued: jobs.filter { $0.status == .queued }.count, cancelling: active.status == .cancelling)
    }
    var label: String { cancelling ? "停止中" : "\(percent)%" }
}

struct CompletionNotice: Equatable {
    let title: String
    let body: String
    init?(job: Job) {
        guard [.completed, .failed].contains(job.status) else { return nil }
        title = job.status == .completed ? "字幕工作完成" : "字幕工作失敗"
        // Never put diagnostic logs, local paths, or provider responses on the lock screen.
        body = job.status == .completed ? "成品已準備完成，請回到 Subtitle Refine 檢視。" : "請回到 Subtitle Refine 查看原因；修正後可繼續工作。"
    }
}

@MainActor final class DesktopFeedback: NSObject, JobFeedback, UNUserNotificationCenterDelegate {
    private let center = UNUserNotificationCenter.current()
    private let settings: AppSettings
    private var previous: DockProgress?
    private var authorizationTask: Task<Bool, Never>?
    private var askedThisSession = false
    private var startedAuthorization = false
    private let dock = DockProgressView(frame: NSRect(x: 0, y: 0, width: 128, height: 128))

    init(settings: AppSettings) {
        self.settings = settings
        super.init()
        center.delegate = self
        Task { await refreshPermission() }
    }

    func refreshPermission() async {
        let status = await center.notificationSettings().authorizationStatus
        settings.notificationStatus = switch status {
        case .authorized, .provisional, .ephemeral: "已允許完成／失敗通知"
        case .denied: "通知未允許；可在系統設定 → 通知 → Subtitle Refine 開啟。"
        case .notDetermined: "首次開始工作時會詢問通知權限；不允許仍可正常處理影片。"
        @unknown default: "請在系統設定確認通知權限。"
        }
    }

    private func authorize() async -> Bool {
        if let authorizationTask { return await authorizationTask.value }
        let task = Task { @MainActor [weak self] in
            guard let self else { return false }
            do {
                let status = await center.notificationSettings().authorizationStatus
                let allowed: Bool
                if status == .notDetermined && !askedThisSession {
                    askedThisSession = true
                    allowed = try await center.requestAuthorization(options: [.alert, .sound])
                } else { allowed = status == .authorized || status == .provisional }
                await refreshPermission()
                return allowed
            } catch {
                settings.notificationStatus = "通知暫時無法使用；工作結果仍保留在 App。"
                return false
            }
        }
        authorizationTask = task
        let result = await task.value
        authorizationTask = nil
        return result
    }

    func update(jobs: [Job]) {
        let current = DockProgress.current(jobs)
        if current != previous {
            previous = current
            let tile = NSApp.dockTile
            if let current {
                dock.progress = current
                tile.contentView = dock
                tile.badgeLabel = current.label
            } else {
                tile.contentView = nil
                tile.badgeLabel = nil
            }
            tile.display()
        }
        if current != nil && settings.notificationsEnabled && !startedAuthorization {
            startedAuthorization = true
            Task { _ = await authorize() }
        }
    }

    func finished(job: Job) {
        guard settings.notificationsEnabled, let notice = CompletionNotice(job: job) else { return }
        Task {
            guard await authorize(), settings.notificationsEnabled else { return }
            let content = UNMutableNotificationContent()
            content.title = notice.title
            content.body = notice.body
            content.sound = .default
            content.userInfo = ["jobID": job.id.uuidString]
            do {
                try await center.add(UNNotificationRequest(identifier: "job-\(job.id.uuidString)", content: content, trigger: nil))
            } catch { settings.notificationStatus = "通知送出失敗；工作結果仍保留在 App。" }
        }
    }

    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter,
        willPresent notification: UNNotification) async -> UNNotificationPresentationOptions { [.banner, .sound] }

    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse) async {
        let id = response.notification.request.content.userInfo["jobID"] as? String
        await MainActor.run {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows.first(where: { $0.canBecomeMain })?.makeKeyAndOrderFront(nil)
            if let id { NotificationCenter.default.post(name: .selectSubtitleJob, object: id) }
        }
    }
}

extension Notification.Name { static let selectSubtitleJob = Notification.Name("selectSubtitleJob") }

@MainActor final class DockProgressView: NSView {
    var progress = DockProgress(percent: 0, queued: 0, cancelling: false) { didSet { needsDisplay = true } }
    override func draw(_ dirtyRect: NSRect) {
        NSApp.applicationIconImage.draw(in: bounds)
        let track = NSRect(x: 14, y: 10, width: bounds.width - 28, height: 14)
        NSColor.black.withAlphaComponent(0.75).setFill()
        NSBezierPath(roundedRect: track, xRadius: 7, yRadius: 7).fill()
        let fill = NSRect(x: track.minX + 3, y: track.minY + 3,
            width: (track.width - 6) * CGFloat(progress.percent) / 100, height: 8)
        (progress.cancelling ? NSColor.systemOrange : NSColor.systemTeal).setFill()
        NSBezierPath(roundedRect: fill, xRadius: 4, yRadius: 4).fill()
    }
}

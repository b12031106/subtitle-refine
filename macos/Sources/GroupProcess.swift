import Foundation
import Darwin

struct ProcessCommand: Sendable {
    var executable: String
    var arguments: [String]
    var environment: [String: String] = ProcessInfo.processInfo.environment
}

enum ProcessOutput: Sendable {
    case stdout(String), stderr(String), exited(Int32), failed(String)
}

/// The lock protects cancellation and the group ID; all blocking I/O is off the main actor.
/// POSIX_SPAWN_SETPGROUP establishes the group before the child executes any code.
final class GroupProcess: @unchecked Sendable {
    private let lock = NSLock()
    private var pid: pid_t = 0
    private var cancelAt: Date?
    private var started = false
    let gracePeriod: TimeInterval

    init(gracePeriod: TimeInterval = 10) { self.gracePeriod = gracePeriod }

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        if cancelAt == nil { cancelAt = Date() }
        if pid > 0 { kill(-pid, SIGTERM) }
    }

    func run(_ command: ProcessCommand, secrets: [String] = []) -> AsyncStream<ProcessOutput> {
        AsyncStream { continuation in
            lock.lock()
            let duplicate = started
            started = true
            lock.unlock()
            guard !duplicate else {
                continuation.yield(.failed("程序不可重複啟動")); continuation.finish(); return
            }
            DispatchQueue.global(qos: .userInitiated).async {
                self.execute(command, redactor: SecretRedactor(secrets: secrets), continuation: continuation)
            }
        }
    }

    private func execute(_ command: ProcessCommand, redactor: SecretRedactor,
                         continuation: AsyncStream<ProcessOutput>.Continuation) {
        defer { continuation.finish() }
        var out: [Int32] = [0, 0], err: [Int32] = [0, 0]
        guard pipe(&out) == 0 else { continuation.yield(.failed("無法建立輸出管線")); return }
        guard pipe(&err) == 0 else {
            close(out[0]); close(out[1]); continuation.yield(.failed("無法建立錯誤管線")); return
        }
        var actions: posix_spawn_file_actions_t?
        var attrs: posix_spawnattr_t?
        posix_spawn_file_actions_init(&actions)
        posix_spawnattr_init(&attrs)
        defer { posix_spawn_file_actions_destroy(&actions); posix_spawnattr_destroy(&attrs) }
        posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO)
        posix_spawn_file_actions_adddup2(&actions, err[1], STDERR_FILENO)
        for fd in out + err { posix_spawn_file_actions_addclose(&actions, fd) }
        // XCTest and GUI launchers can block/ignore signals; do not pass those masks to workers.
        var mask = sigset_t()
        var defaults = sigset_t()
        sigemptyset(&mask)
        sigemptyset(&defaults)
        for signal in [SIGTERM, SIGINT, SIGHUP, SIGPIPE, SIGCHLD] { sigaddset(&defaults, signal) }
        posix_spawnattr_setsigmask(&attrs, &mask)
        posix_spawnattr_setsigdefault(&attrs, &defaults)
        posix_spawnattr_setflags(&attrs, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF))
        posix_spawnattr_setpgroup(&attrs, 0)
        let argv = ([command.executable] + command.arguments).map { strdup($0) } + [nil]
        let env = command.environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { argv.forEach { free($0) }; env.forEach { free($0) } }
        var child: pid_t = 0
        lock.lock()
        if cancelAt != nil {
            lock.unlock()
            for fd in out + err { close(fd) }
            continuation.yield(.exited(SIGTERM)); return
        }
        let result = argv.withUnsafeBufferPointer { args in
            env.withUnsafeBufferPointer { vars in
                posix_spawn(&child, command.executable, &actions, &attrs, args.baseAddress!, vars.baseAddress!)
            }
        }
        if result == 0 { pid = child }
        lock.unlock()
        close(out[1]); close(err[1])
        guard result == 0 else {
            close(out[0]); close(err[0])
            continuation.yield(.failed("無法啟動 Python：\(String(cString: strerror(result)))")); return
        }
        let readers = DispatchGroup()
        for (fd, isError) in [(out[0], false), (err[0], true)] {
            readers.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { close(fd); readers.leave() }
                var decoder = LineDecoder()
                var buffer = [UInt8](repeating: 0, count: 8192)
                while true {
                    let count = read(fd, &buffer, buffer.count)
                    if count < 0 && errno == EINTR { continue }
                    let lines = count > 0 ? decoder.feed(Data(buffer.prefix(count))) : decoder.feed(Data(), final: true)
                    for line in lines {
                        let safe = redactor.redact(line)
                        continuation.yield(isError ? .stderr(safe) : .stdout(safe))
                    }
                    if count <= 0 { break }
                }
            }
        }
        var status: Int32 = 0
        var reaped = false
        var cleanupAt: Date?
        while true {
            if !reaped {
                let waited = waitpid(child, &status, WNOHANG)
                reaped = waited == child || (waited == -1 && errno == ECHILD)
            }
            lock.lock()
            let requested = cancelAt
            lock.unlock()
            // Also clean descendants that outlive an otherwise normally exiting parent.
            if reaped && cleanupAt == nil {
                cleanupAt = Date()
                kill(-child, SIGTERM)
            }
            if let deadline = requested ?? cleanupAt,
               Date().timeIntervalSince(deadline) >= gracePeriod {
                kill(-child, SIGKILL)
                if !reaped {
                    while waitpid(child, &status, 0) == -1 && errno == EINTR {}
                    reaped = true
                }
                break
            }
            if reaped && kill(-child, 0) == -1 && errno == ESRCH { break }
            Thread.sleep(forTimeInterval: 0.02)
        }
        readers.wait()
        lock.lock(); pid = 0; lock.unlock()
        let code = (status & 0x7f) == 0 ? (status >> 8) & 0xff : 128 + (status & 0x7f)
        continuation.yield(.exited(code))
    }
}

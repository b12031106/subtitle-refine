import Foundation

enum OverlayPipeline {
    static func command(python: String, input: URL, output: URL, options: WorkerOptions,
                        overlay: OverlayOptions, dialogue: URL?, environment: [String: String]) throws -> ProcessCommand {
        try overlay.validate(provider: options.provider, dialogueEnabled: dialogue != nil || !overlay.useDialogue)
        var args = ["-m", "subtitle_engine.overlay", input.path, "--out-dir", output.path,
                    "--provider", options.provider.rawValue, "--target", overlay.targetLanguage,
                    "--cover", overlay.cover.rawValue, "--context", options.context]
        if let model = options.model?.nilIfBlank { args += ["--model", model] }
        if let start = overlay.start, let duration = overlay.duration {
            args += ["--start", String(start), "--duration", String(duration)]
        }
        if overlay.useDialogue, let dialogue { args += ["--dialogue-srt", dialogue.path] }
        if let languages = LanguageCatalog.ocrLanguages(options.sourceLanguage) { args += ["--ocr-languages", languages] }
        var env = environment
        env["PYTHON_DOTENV_DISABLED"] = "1"
        return ProcessCommand(executable: python, arguments: args, environment: env)
    }

    /// The engine does not emit a manifest. Validate its top-level output after a zero exit,
    /// then add relative artifact references to the app-owned manifest.
    static func artifacts(in directory: URL, relativeTo root: URL) throws -> [JobArtifact] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let required: [(String, (URL) -> Bool)] = [
            ("overlay_video", { $0.pathExtension == "mp4" }),
            ("overlay_review", { $0.lastPathComponent.hasSuffix(".review.html") }),
            ("overlay_ass", { $0.pathExtension == "ass" }),
            ("overlay_events", { $0.pathExtension == "json" })
        ]
        return try required.map { kind, matches in
            let candidates = files.filter(matches)
            guard candidates.count == 1, let file = candidates.first else {
                throw AppError(message: "畫面字成品缺少或不明確：\(kind)。")
            }
            let base = root.standardizedFileURL.path + "/"
            guard file.standardizedFileURL.path.hasPrefix(base) else { throw AppError(message: "畫面字輸出位置不正確。") }
            let artifact = JobArtifact(kind: kind, path: String(file.standardizedFileURL.path.dropFirst(base.count)))
            _ = try ArtifactManifest.resolvedURL(artifact, output: root)
            return artifact
        }
    }

    static func progress(_ line: String) -> (percent: Double, stage: String)? {
        if line.contains("擷取片段") { return (5, "擷取指定片段") }
        if line.contains("抽格") { return (10, "擷取 OCR 畫格") }
        if line.contains("OCR") { return (20, "辨識畫面文字") }
        if line.contains("文字事件") { return (35, "整理文字事件") }
        if line.contains("判讀") {
            if let range = line.range(of: "[0-9]+/[0-9]+", options: .regularExpression) {
                let numbers = line[range].split(separator: "/").compactMap { Double($0) }
                if numbers.count == 2, numbers[1] > 0 { return (40 + 25 * min(1, numbers[0] / numbers[1]), "翻譯與判讀畫面字") }
            }
            return (40, "翻譯與判讀畫面字")
        }
        if line.contains("校正出現") { return (70, "校正文字出現時間") }
        if line.contains("LaMa") { return (82, "LaMa 修補與合成") }
        if line.contains("合成") { return (80, "合成畫面字翻譯") }
        if line.contains("產生驗收頁") { return (96, "產生原畫面／成品驗收頁") }
        if line.contains("✅ 完成") { return (99, "檢查畫面字成品") }
        return nil
    }
}

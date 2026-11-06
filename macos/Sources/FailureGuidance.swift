import Foundation

struct FailureGuidance: Equatable {
    var title: String
    var recovery: String
    static func describe(stage: String, logs: [String]) -> Self {
        let text = ([stage] + logs.suffix(30)).joined(separator: "\n").lowercased()
        if text.contains("無法啟動 python") || text.contains("no module named") || text.contains("modulenotfounderror") {
            return Self(title: "Python 環境尚未就緒", recovery: "到設定確認 Python 路徑並檢查環境，將缺少的套件安裝到同一個 Python，再按「繼續」。")
        }
        if text.contains("api 金鑰") || text.contains("401") || text.contains("unauthorized") || text.contains("authentication") {
            return Self(title: "提供商驗證失敗", recovery: "確認所選提供商的 API 金鑰；local-runtime 請確認本機 Runtime 的登入與設定，再按「繼續」。")
        }
        if text.contains("429") || text.contains("rate limit") || text.contains("quota") {
            return Self(title: "提供商暫時無法接受更多工作", recovery: "確認額度或等待速率限制解除後再按「繼續」，已完成的快取會保留。")
        }
        if text.contains("connection") || text.contains("timed out") || text.contains("timeout") {
            return Self(title: "連線暫時中斷", recovery: "確認網路與本機 Runtime 服務可用後再按「繼續」。")
        }
        if text.contains("無法讀取來源影片") || text.contains("no such file") {
            return Self(title: "找不到來源檔案", recovery: "將影片放回原位置並確認可讀取，再按「繼續」。")
        }
        if text.contains("permission denied") || text.contains("operation not permitted") {
            return Self(title: "無法存取檔案", recovery: "確認來源影片與工作資料夾的讀寫權限，再按「繼續」。")
        }
        if text.contains("no space left") {
            return Self(title: "儲存空間不足", recovery: "釋出工作磁碟的空間後再按「繼續」；不需刪除工作快取。")
        }
        return Self(title: "工作未完成", recovery: "請查看下方原因與日誌，修正環境或來源問題後按「繼續」。工作資料與快取會保留。")
    }
}

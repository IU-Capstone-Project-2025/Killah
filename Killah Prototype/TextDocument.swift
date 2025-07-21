import SwiftUI
import UniformTypeIdentifiers
import AppKit
import Foundation
import SwiftData

struct TextDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText, .rtf] }
    var text: String

    init(text: String = "") {
        let initialText = NSAttributedString(string: text, attributes: [.font: FontManager.shared.defaultEditorFont()])
        self.text = initialText.string
    }
    
    init(fromFileAt url: URL) throws {
        let data = try Data(contentsOf: url)

        if url.pathExtension.lowercased() == "rtf" {
            if let attr = NSAttributedString(rtf: data, documentAttributes: nil) {
                self.text = attr.string
            } else {
                self.text = ""
            }
        } else {
            self.text = String(data: data, encoding: .utf8) ?? ""
        }
    }

    init(configuration: ReadConfiguration) throws {
        if configuration.contentType == .rtf {
            guard let data = configuration.file.regularFileContents,
                  let attr = NSAttributedString(rtf: data, documentAttributes: nil) else {
                text = ""
                return
            }
            text = attr.string
        } else {
            if let data = configuration.file.regularFileContents,
               let str = String(data: data, encoding: .utf8) {
                text = str
            } else {
                text = ""
            }
        }
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        if configuration.contentType == .rtf {
            let attr = NSAttributedString(string: text)
            guard let data = attr.rtf(from: NSRange(location: 0, length: attr.length), documentAttributes: [:]) else {
                throw CocoaError(.fileWriteUnknown)
            }
            return FileWrapper(regularFileWithContents: data)
        } else {
            let data = text.data(using: .utf8) ?? Data()
            return FileWrapper(regularFileWithContents: data)
        }
    }
}


struct DocumentItem: Identifiable {
    let id = UUID()
    let url: URL
    let filename: String
    let contentPreview: String
    let date: Date
    let isPersonalized: Bool

    var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: date)
    }
    
    static func loadFromDirectory(context: ModelContext) -> [DocumentItem] {
        let fileManager = FileManager.default
        var targetURL: URL?

        if let bookmarkData = UserDefaults.standard.data(forKey: "DefaultDocumentsFolderBookmark") {
            do {
                var isStale = false
                targetURL = try URL(resolvingBookmarkData: bookmarkData, options: .withSecurityScope, relativeTo: nil, bookmarkDataIsStale: &isStale)
                if isStale {
                    // Stale bookmark, need to re-authorize
                    targetURL = nil
                }
            } catch {
                print("Error resolving bookmark: \(error)")
                targetURL = nil
            }
        }

        if targetURL == nil {
            let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            targetURL = documentsURL.appendingPathComponent("Killah")
        }

        guard let url = targetURL else { return [] }

        let shouldStopAccessing = url.startAccessingSecurityScopedResource()
        defer {
            if shouldStopAccessing {
                url.stopAccessingSecurityScopedResource()
            }
        }

        if !fileManager.fileExists(atPath: url.path) {
            do {
                try fileManager.createDirectory(at: url, withIntermediateDirectories: true, attributes: nil)
                print("✅ Created directory at \(url.path)")
            } catch {
                print("🫩 Failed to create directory at \(url.path): \(error.localizedDescription)")
                return []
            }
        }

        do {
            let fileURLs = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: [.creationDateKey], options: [])
            
            let documents = fileURLs
                .filter { $0.pathExtension == "txt" || $0.pathExtension == "rtf" }
                .sorted { url1, url2 in
                    let date1 = (try? url1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
                    let date2 = (try? url2.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date.distantPast
                    return date1 > date2
                }
                .map { url in
                    let content = (try? String(contentsOf: url, encoding: .utf8)) ?? "..."
                    let preview = content.components(separatedBy: .whitespacesAndNewlines)
                        .prefix(20)
                        .joined(separator: " ")
                    let date = (try? fileManager.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? Date()
                    
                    // Проверяем, есть ли эмбеддинг в базе данных для этого URL
                    let isPersonalized: Bool
                    do {
                        let descriptor = FetchDescriptor<Embedding>(
                            predicate: #Predicate { $0.documentURL == url && $0.isPersonalized }
                        )
                        let embeddings = try context.fetch(descriptor)
                        isPersonalized = !embeddings.isEmpty
                    } catch {
                        print("🫩 Ошибка при проверке персонализации для \(url.lastPathComponent): \(error)")
                        isPersonalized = false
                    }
                    
                    return DocumentItem(
                        url: url,
                        filename: url.lastPathComponent,
                        contentPreview: preview,
                        date: date,
                        isPersonalized: isPersonalized // Временно случайное значение для демонстрации
                    )
                }
            
            return documents
        } catch {
            return []
        }
    }
}



struct DocumentOpenerView: View {
    var fileURL: URL
    @State private var text: String

    init(fileURL: URL) {
        self.fileURL = fileURL
        if let data = try? Data(contentsOf: fileURL),
           let str = String(data: data, encoding: .utf8) {
            _text = State(initialValue: str)
        } else {
            _text = State(initialValue: "[Ошибка загрузки файла]")
        }
    }

    var body: some View {
        TextEditor(text: $text)
            .padding()
    }
}


import SwiftUI
import AppKit

// Enum to identify the source of text generation
enum GenerationSource {
    case none
    case autocomplete
    case audio
    case prompt
}

class AppStateManager: ObservableObject {
    static let shared = AppStateManager()
    
    @Published var showWelcome: Bool = true
    @Published var openDocuments: [TextDocument] = []
    
    // Глобальные состояния для загрузки
    @Published var isModelDownloading: Bool = false
    @Published var isPythonScriptsStarting: Bool = false
    
    // State for text generation
    @Published var currentGenerationSource: GenerationSource = .none
    
    // Computed property to check if generation is active
    var isGenerating: Bool {
        currentGenerationSource != .none
    }
    
    private init() {}
    
    // Methods to control generation state
    func startGeneration(from source: GenerationSource) {
        DispatchQueue.main.async {
            self.currentGenerationSource = source
        }
    }
    
    func stopGeneration() {
        DispatchQueue.main.async {
            self.currentGenerationSource = .none
        }
    }
    
    func closeModelDownloadSheet() {
        isModelDownloading = false
    }
    
    func createNewDocument() {
        print("📄 Создаем новый документ")
        #if os(macOS)
        // На macOS сначала закрываем Welcome окно, потом создаем документ
        let welcomeWindows = NSApplication.shared.windows.filter { window in
            window.title == "Welcome"
        }
        
        for window in welcomeWindows {
            print("🔒 Закрываем Welcome окно перед созданием документа")
            window.close()
        }
        
        // Создаем документ после закрытия Welcome окна
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NSDocumentController.shared.newDocument(nil)
        }
        #else
        // На других платформах используем нашу логику
        let newDocument = TextDocument()
        openDocuments.append(newDocument)
        showWelcome = false
        #endif
    }
    
    func openDocument(from url: URL) {
        print("📄 Открываем документ: \(url.lastPathComponent)")
        #if os(macOS)
        // На macOS сначала закрываем Welcome окно, потом открываем документ
        let welcomeWindows = NSApplication.shared.windows.filter { window in
            window.title == "Welcome"
        }
        
        for window in welcomeWindows {
            print("🔒 Закрываем Welcome окно перед открытием документа")
            window.close()
        }
        
        // Открываем документ после закрытия Welcome окна
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NSDocumentController.shared.openDocument(withContentsOf: url, display: true) { _, _, _ in
                print("✅ Документ открыт через NSDocumentController")
            }
        }
        #else
        // На других платформах используем нашу логику
        do {
            let document = try TextDocument(fromFileAt: url)
            openDocuments.append(document)
            showWelcome = false
        } catch {
            print("❌ Ошибка открытия документа: \(error)")
        }
        #endif
    }
    
    func closeDocument(_ document: TextDocument) {
        print("📄 Закрываем документ")
        #if os(macOS)
        // На macOS NSDocumentController сам управляет окнами
        return
        #else
        // На других платформах используем нашу логику
        if let index = openDocuments.firstIndex(where: { $0.text == document.text }) {
            openDocuments.remove(at: index)
        }
        
        // Если нет открытых документов, показываем Welcome
        if openDocuments.isEmpty {
            showWelcome = true
        }
        #endif
    }
    
    func closeAllDocuments() {
        print("📄 Закрываем все документы")
        #if os(macOS)
        // На macOS закрываем все окна документов
        let documentWindows = NSApplication.shared.windows.filter { window in
            window.title != "Welcome" && 
            window.title != "Settings" &&
            !window.title.isEmpty
        }
        
        for window in documentWindows {
            window.close()
        }
        #else
        // На других платформах используем нашу логику
        openDocuments.removeAll()
        showWelcome = true
        #endif
    }
} 
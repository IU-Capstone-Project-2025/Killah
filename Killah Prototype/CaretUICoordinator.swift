import SwiftUI
import Combine
import AppKit

class NonResponderHostingView<Content: View>: NSHostingView<Content> {
    override var acceptsFirstResponder: Bool { true }
}

class CaretUICoordinator: ObservableObject {
    // Триггер для caret-эффекта (анимации)
    @Published var triggerBounceRight: Bool = false
    @Published var triggerBounceLeft: Bool = false
    @Published var caretPositionInWindow: CGPoint = .zero
    @Published var caretSize: CGSize = CGSize(width: 2, height: 20)
    // Visibility flag so we can hide caret while scrolling
    @Published var isHidden: Bool = false
    
    // Basic caret state for coordinate calculations
    @Published var isExpanded: Bool = false
    
    // Audio engine state (read-only from coordinator perspective)
    @Published var isRecording: Bool = false
    @Published var isPaused: Bool = false
    @Published var transcribedText: String = ""
    @Published var audioLevel: Float = 0.0
    @Published var isProcessingAudio: Bool = false

    // User input
    @Published var promptText: String = ""
    
    // LLM generation state
    @Published var isGenerating: Bool = false
    
    // Горизонтальное смещение для всей группы UI, чтобы избежать выхода за пределы окна
    @Published var uiGroupOffsetX: CGFloat = 0

    // Текущая фактическая ширина prompt-field (измеряется через GeometryReader)
    @Published var currentPromptFieldWidth: CGFloat = 150

    // Computed property - overlay should show ONLY during recording, not during processing
    var shouldShowOverlay: Bool {
        return isRecording
    }
    var textInsertionHandler: ((String) -> Void)? // Callback to insert text
    weak var textView: CustomInlineNSTextView? // Связь с CustomInlineNSTextView
    private var appStateManager: AppStateManager { AppStateManager.shared }

    private var audioEngine: AudioEngine
    private var llmEngine: LLMEngine
    private let fontManager = FontManager.shared
    private var cancellables = Set<AnyCancellable>()
    
    // Font and size properties from FontManager
    var editorFontSize: CGFloat { fontManager.defaultEditorFontSize }
    var menuItemSize: CGFloat { fontManager.menuItemSize }
    var promptFieldHeight: CGFloat { fontManager.promptFieldHeight }
    var promptFieldFontSize: CGFloat { fontManager.promptFieldFontSize }

    // Basic layout constants
    let basePromptFieldWidth: CGFloat = 150
    let expandedPromptFieldWidth: CGFloat = 300
    let caretButtonPadding: CGFloat = 24
    
    // Dynamic caret offset based on line height
    var caretVerticalOffset: CGFloat {
        // Используем правильную высоту строки вместо высоты глифа
        let additionalOffset = lineHeight * 0.4
        return lineHeight + additionalOffset - 4
    }
    
    // Get proper line height based on current font
    var lineHeight: CGFloat {
        let font = fontManager.defaultEditorFont()
        return font.ascender - font.descender + font.leading
    }
    
    init(llmEngine: LLMEngine, audioEngine: AudioEngine) {
        self.llmEngine = llmEngine
        self.audioEngine = audioEngine

        // Bind AudioEngine properties to CaretUICoordinator properties
        audioEngine.$isRecording
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isRecording in
                self?.isRecording = isRecording
            }
            .store(in: &cancellables)

        audioEngine.$isPaused
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isPaused in
                self?.isPaused = isPaused
            }
            .store(in: &cancellables)

        audioEngine.$transcribedText
            .receive(on: DispatchQueue.main)
            .sink { [weak self] text in
                self?.transcribedText = text
            }
            .store(in: &cancellables)

        audioEngine.$audioLevel
            .receive(on: DispatchQueue.main)
            .sink { [weak self] level in
                self?.audioLevel = level
            }
            .store(in: &cancellables)

        audioEngine.$isProcessingAudio
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isProcessing in
                self?.isProcessingAudio = isProcessing
            }
            .store(in: &cancellables)
        
        // Bind AudioEngine's onTranscriptionComplete to generate suggestions
        audioEngine.onTranscriptionComplete = { [weak self] transcription in
            self?.generateFromAudioTranscription(transcription)
        }
    }

    func updateCaretPosition(for textView: NSTextView, at charIndex: Int? = nil) {
        let currentInsertionPoint: Int
        if let charIndex = charIndex {
            currentInsertionPoint = charIndex
        } else {
            currentInsertionPoint = textView.selectedRange().location
        }

        let textLength = (textView.string as NSString).length
        let insertionPoint = max(0, min(currentInsertionPoint, textLength))

        var finalCaretPos: CGPoint?
        var finalCaretHeight: CGFloat?

        if let layoutManager = textView.layoutManager, let textContainer = textView.textContainer {
            let glyphRange = layoutManager.glyphRange(forCharacterRange: NSRange(location: insertionPoint, length: 0), actualCharacterRange: nil)
            var localRect = layoutManager.boundingRect(forGlyphRange: glyphRange, in: textContainer)

            // Offset to account for text container origin.
            let containerOrigin = textView.textContainerOrigin
            localRect.origin.x += containerOrigin.x
            localRect.origin.y += containerOrigin.y

            // Convert to window coordinates (bottom-left origin).
            if localRect.height > 0, let window = textView.window {
                let rectInWindow = textView.convert(localRect, to: nil)

                let containerHeight = window.contentView?.bounds.height ?? 0

                let centerX = rectInWindow.origin.x + (rectInWindow.width / 2)
                // Flip Y to SwiftUI's top-left origin.
                let centerY = containerHeight - rectInWindow.origin.y - (rectInWindow.height / 2)

                finalCaretPos = CGPoint(x: centerX, y: centerY)
                finalCaretHeight = rectInWindow.height
            }
        }

        if let pos = finalCaretPos, let height = finalCaretHeight,
           !pos.x.isNaN, !pos.x.isInfinite, !pos.y.isNaN, !pos.y.isInfinite, height > 0 {
            DispatchQueue.main.async {
                // Проверяем, изменилась ли позиция, чтобы избежать лишних обновлений
                if self.caretPositionInWindow != pos || self.caretSize.height != height {
                    self.caretPositionInWindow = pos
                    self.caretSize = CGSize(width: 2, height: height)
                    
                    // If the caret moves, collapse the UI.
                    if self.isExpanded {
                        self.collapseUI()
                    }
                }
            }
        }
        // Больше не сбрасываем позицию на дефолтную при ошибке, чтобы избежать застревания
    }

    // Simple state toggle without animation (views handle their own animations)
    func toggleExpanded() {
        let wasExpanded = isExpanded
        isExpanded.toggle()
        
        // Пересчитываем смещение только при открытии
        if !wasExpanded {
            updateUIGroupOffset()
        }
    }
    
    func updateUIGroupOffset() {
        guard let windowContentRect = NSApplication.shared.keyWindow?.contentView?.frame else {
            print("⚠️ updateUIGroupOffset: Could not get window content rect.")
            return
        }

        let windowWidth = windowContentRect.width
        let margin: CGFloat = 10.0

        let recordButtonWidth = menuItemSize + 12
        let promptFieldWidthForCalc = currentPromptFieldWidth
        let padding = caretButtonPadding
        let caretXInWindow = caretPositionInWindow.x

        let recordButtonCenterX = caretXInWindow - padding - 5
        let uiMinX = recordButtonCenterX - (recordButtonWidth / 2)

        let promptFieldCenterX = caretXInWindow + padding + (currentPromptFieldWidth / 2) + 5
        let uiMaxX = promptFieldCenterX + (promptFieldWidthForCalc / 2)

        var newOffset: CGFloat = 0
        if uiMinX < margin {
            newOffset = margin - uiMinX
        } else if uiMaxX > (windowWidth - margin) {
            newOffset = (windowWidth - margin) - uiMaxX
        } else {
            newOffset = 0
        }
        
        // Применяем смещение с анимацией
        if abs(newOffset - uiGroupOffsetX) > 1 {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                self.uiGroupOffsetX = newOffset
            }
        }
    }
    
    func setExpanded(_ expanded: Bool) {
        isExpanded = expanded
        if !expanded {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                uiGroupOffsetX = 0
            }
        }
    }
    
    func collapseUI() {
        setExpanded(false)
    }

    // MARK: - Visibility helpers
    func hideCaret() {
        isHidden = true
    }
    func showCaret() {
        isHidden = false
    }
    
    // Audio control delegation
    func startRecording() {
        audioEngine.startRecording()
    }
    
    func stopRecording() {
        audioEngine.stopRecording()
        isRecording = false
    }
    
    func togglePause() {
        audioEngine.togglePause()
    }
    
    func generateFromTextPrompt(selectedRange: NSRange?) {
        let selectedText = getSelectedText(from: selectedRange)
        guard !promptText.isEmpty || selectedText != nil else { return }

        // 1. Abort any ongoing autocomplete suggestions
        llmEngine.abortSuggestion(for: "autocomplete")
        
        // 2. Set the global generation state to indicate a high-priority task
        appStateManager.startGeneration(from: .prompt)
        isGenerating = true
        
        // 3. Collapse the UI
        collapseUI()
        
        // 4. Construct the prompt
        let finalPrompt: String
        if let selected = selectedText, !selected.isEmpty {
            finalPrompt = promptText.isEmpty ? selected : "\(selected) : \(promptText)"
        } else {
            finalPrompt = promptText
        }
        
        promptText = ""
        
//        // 5. Generate embeddings
//        llmEngine.generateEmbedding(for: finalPrompt) { [weak self] result in
//            guard let self = self else { return }
//            
//            switch result {
//            case .success(let embeddings):
//                // 6. Generate the final text using the embeddings
//                self.generateTextFromEmbeddings(embeddings: embeddings, replacementRange: selectedRange)
//            case .failure(let error):
//                print("Error generating embeddings: \(error)")
//                self.isGenerating = false
//                self.appStateManager.stopGeneration()
//            }
//        }
        
        let taskType = selectedRange != nil ? "rewriting" : "generation"
        
        llmEngine.generateSuggestion(
            for: "autocomplete",
            prompt: finalPrompt,
            isFromCaret: false,
            taskType: taskType,
            tokenStreamCallback: { [weak self] token in
                DispatchQueue.main.async {
                    if selectedRange != nil {
                        // For replacement, we wait for the full text.
                    } else {
                        self?.textInsertionHandler?(token)
                    }
                }
            },
            onComplete: { [weak self] result in
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    self.isGenerating = false
                    self.appStateManager.stopGeneration()
                    
                    switch result {
                    case .success(let fullSuggestion):
                        if let range = selectedRange {
                            self.textView?.textStorage?.replaceCharacters(in: range, with: fullSuggestion)
                        } else if fullSuggestion.isEmpty {
                            self.textView?.clearGhostText()
                        }
                        // If not replacing, the text was already inserted via stream.
                    case .failure(let error):
                        print("❌ Generation failed: \(error)")
                        if case LLMEngine.LLMError.aborted = error {
                            // Nothing to do on abort
                        } else {
                            self.textView?.clearGhostText()
                        }
                    }
                }
            }
        )
    }
    
    func generateFromAudioTranscription(_ transcriptionOrJson: String) {
        guard !transcriptionOrJson.isEmpty else {
            print("⚠️ Empty transcription or JSON, skipping generation")
            return
        }
        
        llmEngine.abortSuggestion(for: "autocomplete")
        appStateManager.startGeneration(from: .audio)
        isGenerating = true
        
        // Collapse the UI as soon as generation starts
        collapseUI()
        
        let selectedRange = self.textView?.selectedRange
        let selectedText = getSelectedText(from: selectedRange)
        
        let taskType = selectedRange != nil ? "rewriting" : "generation"
        var finalPrompt: String
        
        // Check if the input is JSON (for embeddings) or plain text (for transcription)
        do {
            if transcriptionOrJson.starts(with: "{\"type\":"),
               let data = transcriptionOrJson.data(using: .utf8),
               let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
               let type = json["type"] as? String {
                
                if type == "projected_audio_embeds" {
                    // Handle embeddings
                    if let embeddings = json["embeddings"] as? [[Double]] {
                        let embeddingsJson: [String: Any] = [
                            "type": "projected_audio_embeds",
                            "embeddings": embeddings,
                            "prompt": selectedText ?? ""
                        ]
                        guard let embeddingsJsonData = try? JSONSerialization.data(withJSONObject: embeddingsJson),
                              let embeddingsJsonString = String(data: embeddingsJsonData, encoding: .utf8) else {
                            print("❌ Failed to create embeddings JSON string")
                            isGenerating = false
                            appStateManager.stopGeneration()
                            return
                        }
                        finalPrompt = embeddingsJsonString
                    } else {
                        print("❌ Invalid embeddings JSON: missing or invalid 'embeddings' field")
                        isGenerating = false
                        appStateManager.stopGeneration()
                        return
                    }
                } else if type == "transcription" {
                    // Handle transcription
                    if let text = json["text"] as? String {
                        if let selected = selectedText, !selected.isEmpty {
                            finalPrompt = text.isEmpty ? selected : "\(selected) : \(text)"
                        } else {
                            finalPrompt = text
                        }
                    } else {
                        print("❌ Invalid transcription JSON: missing 'text' field")
                        isGenerating = false
                        appStateManager.stopGeneration()
                        return
                    }
                } else {
                    print("❌ Unknown JSON type: \(type)")
                    isGenerating = false
                    appStateManager.stopGeneration()
                    return
                }
            } else {
                // Assume plain text transcription for backward compatibility
                if let selected = selectedText, !selected.isEmpty {
                    finalPrompt = transcriptionOrJson.isEmpty ? selected : "\(selected) : \(transcriptionOrJson)"
                } else {
                    finalPrompt = transcriptionOrJson
                }
            }
            
            // Generate suggestion
            llmEngine.generateSuggestion(
                for: "autocomplete",
                prompt: finalPrompt,
                isFromCaret: finalPrompt.starts(with: "{\"type\":\"projected_audio_embeds\""),
                taskType: taskType,
                tokenStreamCallback: { [weak self] token in
                    DispatchQueue.main.async {
                        if selectedRange != nil {
                            // For replacement, wait for the full text
                        } else {
                            self?.textInsertionHandler?(token)
                        }
                    }
                },
                onComplete: { [weak self] result in
                    DispatchQueue.main.async {
                        guard let self = self else { return }
                        self.isGenerating = false
                        self.appStateManager.stopGeneration()
                        
                        switch result {
                        case .success(let fullSuggestion):
                            if let range = selectedRange {
                                self.textView?.textStorage?.replaceCharacters(in: range, with: fullSuggestion)
                            } else if fullSuggestion.isEmpty {
                                self.textView?.clearGhostText()
                            }
                            // If not replacing, text was already inserted via stream
                        case .failure(let error):
                            print("❌ Generation failed: \(error)")
                            if case LLMEngine.LLMError.aborted = error {
                                // Nothing to do on abort
                            } else {
                                self.textView?.clearGhostText()
                            }
                        }
                    }
                }
            )
        } catch {
            print("❌ Error parsing JSON: \(error)")
            isGenerating = false
            appStateManager.stopGeneration()
        }
    }

    private func generateTextFromEmbeddings(embeddings: [Float], replacementRange: NSRange?) {
        let embeddingsDict: [String: Any] = ["type": "text_embeds", "embeddings": embeddings]
        guard let embeddingsJsonData = try? JSONSerialization.data(withJSONObject: embeddingsDict),
              let embeddingsJsonString = String(data: embeddingsJsonData, encoding: .utf8) else {
            print("❌ Failed to create embeddings JSON string")
            isGenerating = false
            appStateManager.stopGeneration()
            return
        }
        
        let prompt = embeddingsJsonString
        let taskType = replacementRange != nil ? "rewriting" : "generation"
        
        llmEngine.generateSuggestion(
            for: "caret",
            prompt: prompt,
            isFromCaret: true,
            taskType: taskType,
            tokenStreamCallback: { [weak self] token in
                DispatchQueue.main.async {
                    if replacementRange != nil {
                        // For replacement, we wait for the full text.
                    } else {
                        self?.textInsertionHandler?(token)
                    }
                }
            },
            onComplete: { [weak self] result in
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    self.isGenerating = false
                    self.appStateManager.stopGeneration()
                    
                    switch result {
                    case .success(let fullSuggestion):
                        if let range = replacementRange {
                            self.textView?.textStorage?.replaceCharacters(in: range, with: fullSuggestion)
                        } else if fullSuggestion.isEmpty {
                            self.textView?.clearGhostText()
                        }
                        // If not replacing, the text was already inserted via stream.
                    case .failure(let error):
                        print("❌ Generation failed: \(error)")
                        if case LLMEngine.LLMError.aborted = error {
                            // Nothing to do on abort
                        } else {
                            self.textView?.clearGhostText()
                        }
                    }
                }
            }
        )
    }
    
    private func getSelectedText(from range: NSRange?) -> String? {
        guard let range = range,
              let textView = self.textView,
              let textStorage = textView.textStorage,
              range.location != NSNotFound,
              NSMaxRange(range) <= textStorage.length else {
            return nil
        }
        return (textStorage.string as NSString).substring(with: range)
    }
    
    private func processEmbeddings(_ embeddingsJson: String) {
        llmEngine.startEngine(for: "caret")
        let checkInterval: TimeInterval = 0.1
        let maxAttempts = 50
        var attempts = 0
        
        // В цикле ждем, пока Python-скрипт не будет готов к работе.
        // Это ожидание происходит в фоновом потоке.
        while self.llmEngine.getRunnerState(for: "caret") != .running && attempts < maxAttempts {
            Thread.sleep(forTimeInterval: checkInterval) // Пауза в фоновом потоке
            attempts += 1
        }
        
        // Проверяем, запустился ли движок после ожидания
        if self.llmEngine.getRunnerState(for: "caret") == .running {
            let prompt = "\(embeddingsJson)|||Generate text based on this input:"
            llmEngine.generateSuggestion(
                for: "caret",
                prompt: prompt,
                isFromCaret: true,
                taskType: "generation",
                tokenStreamCallback: { [weak self] token in
                    DispatchQueue.main.async {
                        self?.textInsertionHandler?(token)
                    }
                },
                onComplete: { [weak self] result in
                    DispatchQueue.main.async {
                        guard let self = self else { return }
                        self.isGenerating = false
                        self.appStateManager.stopGeneration()
                        
                        switch result {
                        case .success(let fullSuggestion):
                            if fullSuggestion.isEmpty {
                                self.textView?.clearGhostText()
                            }
                        case .failure(let error):
                            print("❌ Generation failed: \(error)")
                            if case LLMEngine.LLMError.aborted = error {
                                // Ничего не делаем при прерывании
                            } else {
                                self.textView?.clearGhostText()
                            }
                        }
                    }
                }
            )
        } else {
            print("❌ caret.py failed to reach running state after \(Double(maxAttempts) * checkInterval) seconds")
            self.isGenerating = false
            self.appStateManager.stopGeneration()
        }
    }
    
    // Helper function for prompt field height calculation
    func calculatePromptFieldHeight() -> CGFloat {
        let font = NSFont.systemFont(ofSize: promptFieldFontSize)
        // Используем expandedPromptFieldWidth для расчёта высоты (или basePromptFieldWidth, если не расширено)
        let width = expandedPromptFieldWidth - 24
        let text = promptText.isEmpty ? " " : promptText
        let nsText = text as NSString
        let boundingRect = nsText.boundingRect(
            with: CGSize(width: width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: font],
            context: nil
        )
        // Корректная высота строки для NSFont
        let lineHeight = font.ascender - font.descender + font.leading
        let numberOfLines = max(1, Int(ceil(boundingRect.height / lineHeight)))
        let minHeight = promptFieldHeight
        let maxHeight = minHeight * 3
        let totalHeight = CGFloat(numberOfLines) * lineHeight + 12 // 12 — паддинг
        return max(minHeight, min(totalHeight, maxHeight))
    }
}

private extension String {
    func height(withConstrainedWidth width: CGFloat, font: NSFont) -> CGFloat {
        let constraintRect = CGSize(width: width, height: .greatestFiniteMagnitude)
        let boundingBox = self.boundingRect(with: constraintRect, options: .usesLineFragmentOrigin, attributes: [.font: font], context: nil)
        return ceil(boundingBox.height)
    }

    func width(withConstrainedHeight height: CGFloat, font: NSFont) -> CGFloat {
        let constraintRect = CGSize(width: .greatestFiniteMagnitude, height: height)
        let boundingBox = self.boundingRect(with: constraintRect, options: .usesLineFragmentOrigin, attributes: [.font: font], context: nil)
        return ceil(boundingBox.width)
    }
}

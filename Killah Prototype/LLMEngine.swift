import Foundation
import Combine
import AppKit
import CryptoKit // Required for CacheManager's SHA-256 extension
import Darwin
import SwiftData

// Структура для декодирования JSON из text_to_embeddings.py
struct EmbeddingResponse: Codable {
    let type: String
    let embeddings: [Float]
}

// Structure for decoding attention weights and indices from attention.py
struct AttentionResponse: Codable {
    let weights: [Double]
    let indices: [Int]
}

class LLMEngine: ObservableObject {
    @Published var suggestion: String = ""
    @Published var engineState: EngineState = .idle
    private var activeTasks: [String: Bool] = [:] // Отслеживание активных задач
    
    enum EngineState: Equatable {
        case idle
        case starting
        case running
        case stopped
        case error(String)
        static func == (lhs: EngineState, rhs: EngineState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.starting, .starting), (.running, .running), (.stopped, .stopped):
                return true
            case let (.error(lhsError), .error(rhsError)):
                return lhsError == rhsError
            default:
                return false
            }
        }
    }

    enum LLMError: Error {
        case engineNotRunning
        case pythonScriptNotReady
        case processLaunchError(String)
        case promptEncodingError
        case stdinWriteError(String)
        case scriptError(String)
        case aborted
    }

    private var runners: [String: PythonScriptRunner] = [:]
    private var modelServer: ModelServerRunner
    private var cancellables = Set<AnyCancellable>()
    private var currentTemperature: Float = 0.8 // Начальное значение температуры
    private let modelContainer: ModelContainer
    private let loraAdapterMap: [String: String]
    
    init(modelManager: ModelManager,modelContainer: ModelContainer) {
        print("LLMEngine init")
        self.modelContainer = modelContainer
        let modelDir = modelManager.getModelsDirectory().path
        
        let loraDir = modelManager.getModelsDirectory().appendingPathComponent("lora").path
        self.loraAdapterMap = [
            "autocomplete": "\(loraDir)/autocomplete_lora_f16.gguf",
            "rewriting": "\(loraDir)/rewriting_lora_f16.gguf",
            "generation": "\(loraDir)/story_lora_f16.gguf"
            // Add other task-to-adapter mappings here
        ]
        
        // Initialize the model server
        modelServer = ModelServerRunner(modelDirectory: modelDir)
        modelServer.start() // Start the server first
        
        // Initialize Python script runners
        runners["audio"] = AudioScriptRunner(modelDirectory: modelDir)
        runners["autocomplete"] = AutocompleteScriptRunner(modelDirectory: modelDir)
        runners["embeddings"] = EmbeddingsRunner(modelDirectory: modelDir)
        runners["caret"] = CaretScriptRunner(modelDirectory: modelDir)
        runners["attention"] = AttentionRunner(modelDirectory: modelDir)

        NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)
            .sink { [weak self] _ in
                print("App is terminating, stopping engine...")
                self?.stopEngine()
            }
            .store(in: &cancellables)
    }
    
    func getRunnerState(for key: String) -> EngineState? {
            return runners[key]?.state
        }

    func startEngine(for script: String) {
        guard let runner = runners[script] else {
            print("❌ Unknown script: \(script)")
            updateEngineState(.error("Unknown script: \(script)"))
            return
        }
        // Invalidate cache when starting engine, as model may change
        CacheManager.shared.invalidateCache()
        runner.start()
        updateEngineState(runner.state)
    }

    func generateSuggestion(
        for script: String,
        prompt: String,
        isFromCaret: Bool = false,  // Флаг, указывающий, что промпт содержит эмбеддинги
        taskType: String? = nil, // New parameter to specify the task type
        tokenStreamCallback: @escaping (String) -> Void,
        onComplete: @escaping (Result<String, LLMEngine.LLMError>) -> Void
    ) {
        if activeTasks[script] == true {
            print("⏳ Waiting for previous task in \(script) to complete")
            runners[script]?.abortSuggestion(notifyPython: true)
        }
        activeTasks[script] = true
        
        let loraAdapterPath = taskType.flatMap { loraAdapterMap[$0] }
        
        Task {
            let personalizedDocs = await getPersonalizedDocuments()
            let docEmbeddings = personalizedDocs.map { $0.embedding }
            let docURLs = personalizedDocs.map { $0.url }
            
            var augmentedPrompt = prompt
            var weightedDocumentEmbedding: [Float]?
            
            if !docEmbeddings.isEmpty {
                let attentionResult = await computeAttentionWithEmbeddings(
                    prompt: prompt,
                    isFromCaret: isFromCaret,
                    docEmbeddings: docEmbeddings,
                    docURLs: docURLs
                )
                
                switch attentionResult {
                case .success(let result):
                    augmentedPrompt = result.prompt
                    weightedDocumentEmbedding = result.weightedEmbedding
                case .failure(let error):
                    print("⚠️ Attention computation failed: \(error). Proceeding without context.")
                }
            }
            
            continueGeneration(
                script: script,
                prompt: augmentedPrompt,
                loraAdapter: loraAdapterPath,
                weightedDocumentEmbedding: weightedDocumentEmbedding,
                tokenStreamCallback: tokenStreamCallback,
                onComplete: onComplete
            )
        }
    }
    
    // Helper function to encapsulate attention logic
    private func computeAttentionWithEmbeddings(
        prompt: String,
        isFromCaret: Bool,
        docEmbeddings: [[Float]],
        docURLs: [URL]
    ) async -> Result<(prompt: String, weightedEmbedding: [Float]?), LLMEngine.LLMError> {
        
        let targetEmbedding: [Float]
        if isFromCaret {
            if let json = extractEmbeddingsJson(from: prompt), let embeddings = parseEmbeddings(from: json) {
                targetEmbedding = embeddings
            } else {
                return .failure(.scriptError("Failed to extract or parse embeddings from prompt"))
            }
        } else {
            do {
                targetEmbedding = try await withCheckedThrowingContinuation { continuation in
                    generateEmbedding(for: prompt) { result in
                        continuation.resume(with: result)
                    }
                }
            } catch {
                return .failure(error as? LLMEngine.LLMError ?? .scriptError("Unknown embedding error"))
            }
        }
        
        let attentionResult: AttentionResponse
        do {
            attentionResult = try await withCheckedThrowingContinuation { continuation in
                computeAttentionWeights(target: targetEmbedding, histories: docEmbeddings) { result in
                    continuation.resume(with: result)
                }
            }
        } catch {
            return .failure(error as? LLMEngine.LLMError ?? .scriptError("Unknown attention weights error"))
        }

        let weights = attentionResult.weights
        let indices = attentionResult.indices
        print("ℹ️ Selected top \(indices.count) documents with weights: \(weights) and indices: \(indices)")
        
        var weightedEmbedding: [Float]?
        if !weights.isEmpty && !indices.isEmpty {
            // Normalize weights to sum to 1
            let weightSum = weights.reduce(0, +)
            let normalizedWeights = weights.map { $0 / weightSum }

            // Compute weighted sum of embeddings
            let embeddingSize = docEmbeddings[indices[0]].count
            var weightedSum = [Float](repeating: 0.0, count: embeddingSize)
            for (index, weight) in zip(indices, normalizedWeights) {
                let embedding = docEmbeddings[index]
                for i in 0..<embeddingSize {
                    weightedSum[i] += embedding[i] * Float(weight)
                }
            }
            weightedEmbedding = weightedSum
        }
        
        return .success((prompt: prompt, weightedEmbedding: weightedEmbedding))
    }

    
    // Вспомогательный метод для продолжения генерации
    private func continueGeneration(
        script: String,
        prompt: String,
        loraAdapter: String?,
        weightedDocumentEmbedding: [Float]?,
        tokenStreamCallback: @escaping (String) -> Void,
        onComplete: @escaping (Result<String, LLMEngine.LLMError>) -> Void
    ) {
        if let cachedSuggestion = CacheManager.shared.getCachedSuggestion(for: prompt, temperature: self.currentTemperature) {
            print("📦 Cache hit for prompt: \"\(prompt)\"")
            let tokens = cachedSuggestion.components(separatedBy: .newlines).filter { !$0.isEmpty }
            var currentIndex = 0
            func sendNextToken() {
                guard currentIndex < tokens.count else {
                    onComplete(.success(cachedSuggestion))
                    return
                }
                let token = tokens[currentIndex]
                print("📦 Sending cached token: \"\(token)\"")
                tokenStreamCallback(token)
                currentIndex += 1
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    sendNextToken()
                }
            }
            sendNextToken()
            return
        }
        print("📄 Generating suggestion for \(script) with prompt: \"\(prompt.prefix(100))\"")

        guard let runner = runners[script] else {
            onComplete(.failure(.scriptError("Unknown script: \(script)")))
            return
        }

        // Prepare JSON payload with weighted document embedding
        var content: [String: Any] = ["prompt": prompt]
        if let embedding = weightedDocumentEmbedding {
            content["weighted_document_embedding"] = embedding
        }
        let payload: [String: Any] = [
            "messages": [
                [
                    "role": "user",
                    "content": content
                ]
            ],
            "max_tokens": 50,
            "temperature": currentTemperature,
            "min_p": 0.1,
            "stream": true,
            "lora_path": loraAdapter ?? ""
        ]

        
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: payload)
            guard let jsonString = String(data: jsonData, encoding: .utf8) else {
                onComplete(.failure(.promptEncodingError))
                return
            }

            runner.sendData(jsonString, loraPath: loraAdapter, tokenStreamCallback: tokenStreamCallback) { result in
                self.activeTasks[script] = false
                switch result {
                case .success(let suggestion):
                    CacheManager.shared.setCachedSuggestion(suggestion, for: prompt, temperature: self.currentTemperature)
                    onComplete(.success(suggestion))
                case .failure(let error):
                    onComplete(.failure(error))
                }
            }
            updateEngineState(runner.state)
        } catch {
            onComplete(.failure(.promptEncodingError))
        }
    }
    
    // Вспомогательные функции для извлечения и парсинга эмбеддингов
    private func extractEmbeddingsJson(from prompt: String) -> String? {
        // Предполагаем, что формат промпта: "embeddingsJson|||prompt_text"
        let parts = prompt.split(separator: "|||", maxSplits: 1)
        if parts.count == 2 {
            return String(parts[0])
        }
        return nil
    }
    
    private func parseEmbeddings(from jsonString: String?) -> [Float]? {
        guard let jsonString = jsonString else { return nil }
        do {
            let data = jsonString.data(using: .utf8)!
            let json = try JSONSerialization.jsonObject(with: data, options: []) as! [String: Any]
            if let embeddings = json["embeddings"] as? [Double] {
                return embeddings.map { Float($0) }
            } else if let embeddings = json["embeddings"] as? [Float] {
                return embeddings
            }
        } catch {
            print("🫩 Ошибка парсинга JSON эмбеддингов: \(error)")
        }
        return nil
    }
    
    func computeAttentionWeights(
        target: [Float],
        histories: [[Float]],
        onComplete: @escaping (Result<AttentionResponse, LLMEngine.LLMError>) -> Void
    ) {
        guard let runner = runners["attention"] else {
            onComplete(.failure(.scriptError("Attention runner not found")))
            return
        }
        
        let inputData: [String: Any] = [
            "target": target,
            "histories": histories
        ]
        
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: inputData)
            guard let jsonString = String(data: jsonData, encoding: .utf8) else {
                onComplete(.failure(.promptEncodingError))
                return
            }
            
            runner.sendData(jsonString, loraPath: nil, tokenStreamCallback: { _ in }, onComplete: { result in
                switch result {
                case .success(let output):
                    do {
                        let response = try JSONDecoder().decode(AttentionResponse.self, from: Data(output.utf8))
                        onComplete(.success(response))
                    } catch {
                        onComplete(.failure(.scriptError("Failed to parse attention weights: \(error)")))
                    }
                case .failure(let error):
                    onComplete(.failure(error))
                }
            })
        } catch {
            onComplete(.failure(.promptEncodingError))
        }
    }
    
    func generateEmbedding(
        for text: String,
        onComplete: @escaping (Result<[Float], LLMEngine.LLMError>) -> Void
    ) {
        guard let runner = runners["embeddings"] else {
            onComplete(.failure(.scriptError("Embeddings runner not found")))
            return
        }
        let input = text
        runner.sendData(input, loraPath: nil, tokenStreamCallback: { _ in }) { result in
            switch result {
            case .success(let output):
                do {
                    let data = try JSONDecoder().decode(EmbeddingResponse.self, from: Data(output.utf8))
                    if data.type == "text_embeds" {
                        onComplete(.success(data.embeddings))
                    } else {
                        onComplete(.failure(.scriptError("Invalid embeddings type: \(data.type)")))
                    }
                } catch {
                    onComplete(.failure(.scriptError("Failed to parse embeddings: \(error)")))
                }
            case .failure(let error):
                onComplete(.failure(error))
            }
        }
    }
    
    func sendCommand(_ command: String, for script: String) {
        guard let runner = runners[script] else {
            print("❌ Unknown script: \(script)")
            return
        }
        
        if command == "INCREASE_TEMPERATURE" {
            currentTemperature = min(currentTemperature + 0.1, 2.0)
            print("🌡️ Temperature increased to \(currentTemperature)")
        } else if command == "DECREASE_TEMPERATURE" {
            currentTemperature = max(currentTemperature - 0.1, 0.1)
            print("🌡️ Temperature decreased to \(currentTemperature)")
        }
        
        runner.sendCommand(command)
    }

    func getPersonalizedDocuments() async -> [(url: URL, content: String, embedding: [Float])] {
        var personalizedDocs: [(url: URL, content: String, embedding: [Float])] = []
            
        personalizedDocs = await MainActor.run {
            var localDocs: [(url: URL, content: String, embedding: [Float])] = []
            let context = self.modelContainer.mainContext
            print("ℹ️ Using model context: \(ObjectIdentifier(context))")
            do {
                let descriptor = FetchDescriptor<Embedding>(predicate: #Predicate { $0.isPersonalized })
                let embeddings = try context.fetch(descriptor)
                print("ℹ️ Найдено эмбеддингов в базе: \(embeddings.count)")
                for embedding in embeddings {
                    let documentURL = embedding.documentURL.standardizedFileURL
                    if let content = try? String(contentsOf: documentURL, encoding: .utf8),
                       let embeddingArray = try? JSONDecoder().decode([Float].self, from: embedding.embeddingData) {
                        localDocs.append((url: documentURL, content: content, embedding: embeddingArray))
                    } else {
                        print("⚠️ Не удалось загрузить содержимое или эмбеддинг для документа: \(documentURL.lastPathComponent)")
                    }
                }
            } catch {
                print("🫩 Ошибка загрузки персонализированных документов: \(error)")
            }
            return localDocs
        }
        
        print("ℹ️ Найдено персонализированных документов: \(personalizedDocs.count)")
        return personalizedDocs
    }
        
    
    func stopEngine(for script: String? = nil) {
        if let script = script, let runner = runners[script] {
            runner.stop()
            updateEngineState(runner.state)
        } else {
            modelServer.stop()
            runners.forEach { $0.value.stop() }
            updateEngineState(.stopped)
        }
    }
    
    func abortSuggestion(for script: String, notifyPython: Bool = true) {
        guard let runner = runners[script] else {
            print("❌ Unknown script: \(script)")
            return
        }
        print("ℹ️ Aborting suggestion for \(script)")
        runner.abortSuggestion(notifyPython: notifyPython)
        updateEngineState(runner.state)
    }
    
    private func updateEngineState(_ newState: EngineState) {
        DispatchQueue.main.async {
            if self.engineState != newState {
                print("⚙️ LLMEngine state changing from \(self.engineState) to \(newState)")
                self.engineState = newState
            }
        }
    }

    deinit {
        print("🗑️ LLMEngine deinit - Stopping engine.")
        print(Thread.callStackSymbols.joined(separator: "\n"))
        stopEngine()
    }
}

class ModelServerRunner {
    private var serverProcess: Process?
    private var _state: LLMEngine.EngineState = .idle
    private let modelDirectory: String

    init(modelDirectory: String) {
        self.modelDirectory = modelDirectory
    }

    var state: LLMEngine.EngineState {
        return _state
    }

    func start() {
        guard state == .idle || state == .stopped else {
            print("ℹ️ Model server already running or starting")
            return
        }
        
        print("🚀 Starting Python server...")
        updateState(.starting)
        
        let process = Process()
        serverProcess = process
        
        guard let resourcesPath = Bundle.main.resourcePath else {
            updateState(.error("Bundle resources path not found"))
            return
        }
        
        let pythonPath = resourcesPath + "/venv/bin/python"
        let scriptPath = resourcesPath + "/run_server.py"
        
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [scriptPath]
        
        var environment = ProcessInfo.processInfo.environment
        environment["MODEL_DIR"] = self.modelDirectory
        process.environment = environment
        
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        
        stderrPipe.fileHandleForReading.readabilityHandler = { pipe in
            let data = pipe.availableData
            if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                print("🐍 Python Server STDERR: \"\(output.trimmingCharacters(in: .whitespacesAndNewlines))\"")
                if output.contains("Uvicorn running on") {
                    self.updateState(.running)
                }
            }
        }
        
        do {
            try process.run()
            print("✅ Python server launched. PID: \(process.processIdentifier)")
        } catch {
            print("🫩 Error launching Python server: \(error)")
            updateState(.error("Launch fail: \(error.localizedDescription)"))
            serverProcess = nil
        }
    }

    func stop() {
        if let process = serverProcess, process.isRunning {
            print("🛑 Stopping Python server with PID: \(process.processIdentifier)")
            process.terminate() // Отправляем SIGTERM
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                if process.isRunning {
                    print("🛑 Python server is still running, forcing termination")
                    kill(process.processIdentifier, SIGKILL) // Отправляем SIGKILL
                } else {
                    print("🛑 Python server stopped successfully")
                }
            }
        } else {
            print("🛑 No running server process found")
        }
        serverProcess = nil
        updateState(.stopped)
    }

    // This function is no longer needed as LoRA adapters are passed per-request.
    // func applyLoraAdapter(adapterPath: String?, scale: Float = 1.0, completion: @escaping (Result<Void, Error>) -> Void) { ... }
    
    private func updateState(_ newState: LLMEngine.EngineState) {
        if _state != newState {
            print("⚙️ ModelServerRunner state changing from \(_state) to \(newState)")
            _state = newState
        }
    }
}

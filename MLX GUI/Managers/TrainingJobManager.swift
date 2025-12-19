//
//  TrainingJobManager.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation
import Combine
import Darwin

@MainActor
class TrainingJobManager: ObservableObject {
    static let shared = TrainingJobManager()
    
    @Published var jobs: [TrainingJob] = []
    
    // Preloaded configuration from YAML file (used when loading a config into the wizard)
    var preloadedConfig: TrainingConfiguration? = nil
    
    // Job currently being edited via the Training Wizard (if any)
    var jobBeingEdited: TrainingJob? = nil
    
    // Track job failures for alert display
    @Published var failedJob: TrainingJob? = nil
    
    // Flag to track if app is shutting down (prevents termination handlers from accessing process info)
    // NOTE: You may see "Unable to obtain a task name port right for pid X: (os/kern) failure (0x5)"
    // in the Xcode console when quitting the app. This is a harmless system-level message from macOS
    // that occurs during app shutdown when the system tries to access process information. It does not
    // affect functionality and can be safely ignored. This is a known limitation of macOS process
    // management during shutdown and cannot be fully eliminated from application code.
    private var isShuttingDown = false
    
    private let jobsDirectory: URL
    private let configsDirectory: URL
    
    private init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let baseDir = appSupport.appendingPathComponent("MLX Training Studio", isDirectory: true)
        self.jobsDirectory = baseDir.appendingPathComponent("jobs", isDirectory: true)
        self.configsDirectory = baseDir.appendingPathComponent("configs", isDirectory: true)
        
        // Create directories if they don't exist
        try? FileManager.default.createDirectory(at: jobsDirectory, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: configsDirectory, withIntermediateDirectories: true)
        
        loadJobs()
    }
    
    // MARK: - Job Management
    
    func createJob(name: String, config: TrainingConfiguration) throws -> TrainingJob {
        // Validate configuration
        let errors = config.validate()
        if !errors.isEmpty {
            throw NSError(domain: "TrainingJobManager", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Configuration validation failed: \(errors.joined(separator: ", "))"
            ])
        }
        
        // Save configuration to YAML
        let configFileName = "\(name.replacingOccurrences(of: " ", with: "_"))_\(UUID().uuidString.prefix(8)).yaml"
        let configPath = configsDirectory.appendingPathComponent(configFileName)
        try YAMLManager.shared.saveConfiguration(config, to: configPath)
        
        // Create job
        let job = TrainingJob(name: name, config: config, configPath: configPath)
        jobs.append(job)
        
        // Save job metadata
        try saveJob(job)
        
        return job
    }
    
    /// Duplicate a job with the same configuration, creating a new job with a modified name
    func duplicateJob(_ job: TrainingJob) throws -> TrainingJob {
        // Generate a new name: "Original Name (Copy)" or "Original Name (Copy 2)", etc.
        let baseName = job.name
        var newName = "\(baseName) (Copy)"
        var copyNumber = 2
        
        // Check if a job with this name already exists, increment the copy number if needed
        while jobs.contains(where: { $0.name == newName }) {
            newName = "\(baseName) (Copy \(copyNumber))"
            copyNumber += 1
        }
        
        // Create a new job with the same configuration
        return try createJob(name: newName, config: job.config)
    }
    
    /// Reset a job's state to allow rerunning it (clears output, errors, metrics, etc.)
    func resetAndRerunJob(_ job: TrainingJob) {
        job.status = .pending
        job.output = ""
        job.error = nil
        job.startedAt = nil
        job.completedAt = nil
        job.metrics = [:]
        job.downloadProgress = 0.0
        job.downloadStatus = nil
        job.isDownloading = false
        
        // Save the reset state
        try? saveJob(job)
    }
    
    /// Update an existing job's configuration and name (only when not running).
    /// This overwrites the existing YAML config on disk and clears prior output/history.
    func updateJob(_ job: TrainingJob, newName: String, newConfig: TrainingConfiguration) throws {
        // Only allow editing when the job is not currently running
        guard job.status != .running else {
            throw NSError(domain: "TrainingJobManager", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Cannot edit a running job."
            ])
        }
        
        // Save updated configuration to the existing YAML file
        try YAMLManager.shared.saveConfiguration(newConfig, to: job.configPath)
        
        // Update in-memory job metadata
        job.name = newName
        job.config = newConfig
        
        // Clear prior run history so the job can be run again with the new config
        job.status = .pending
        job.output = ""
        job.error = nil
        job.startedAt = nil
        job.completedAt = nil
        job.metrics = [:]
        job.downloadProgress = 0.0
        job.downloadStatus = nil
        job.isDownloading = false
        
        // Persist the updated job metadata
        try saveJob(job)
    }
    
    func startJob(_ job: TrainingJob, pythonPath: String, venvPython: URL, hfToken: String? = nil) async {
        guard job.status == .pending || job.status == .paused else { return }
        
        job.status = .running
        job.startedAt = Date()
        job.error = nil
        job.downloadProgress = 0.0
        job.downloadStatus = nil
        job.isDownloading = false
        
        // Pre-download model with progress tracking if needed
        await downloadModelIfNeeded(modelId: job.config.model, hfToken: hfToken, job: job, venvPython: venvPython)
        
        // Note: Dataset downloading is handled by mlx-lm-lora during training
        // The datasets library has a complex cache structure that's best handled natively
        
        // Automatically enable fuse if post-training quantization or GGUF conversion is enabled
        // This is required because quantization and GGUF conversion need the fused model at the adapter path
        var trainingConfig = job.config
        if trainingConfig.enablePostTrainingQuantization || trainingConfig.enableGGUFConversion {
            trainingConfig.fuse = true
        }
        
        let (executable, arguments) = CommandBuilder.buildCommand(from: trainingConfig, pythonPath: venvPython.path, configFilePath: job.configPath.path)
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        
        // Set up environment variables
        // Start with current process environment
        var env = ProcessInfo.processInfo.environment
        
        // Force unbuffered Python output so we see progress in real-time
        env["PYTHONUNBUFFERED"] = "1"
        
        // Load shell environment variables (from .zshrc, .bashrc, etc.)
        let shellEnv = await loadShellEnvironment()
        // Merge shell environment, with shell vars taking precedence for conflicts
        for (key, value) in shellEnv {
            env[key] = value
        }
        
        // Override with GUI settings (HF_TOKEN from settings takes precedence)
        if let token = hfToken, !token.isEmpty {
            env["HF_TOKEN"] = token
        }
        
        process.environment = env
        
        // Log environment variables for debugging (mask token for security)
        let envVarsForLog = env.map { key, value in
            if key == "HF_TOKEN" {
                return "\(key)=\(value.isEmpty ? "(empty)" : "\(String(value.prefix(8)))...")"
            }
            return "\(key)=\(value)"
        }.sorted().joined(separator: "\n")
        
        let envLog = "\n=== Environment Variables ===\n\(envVarsForLog)\n=============================\n\n"
        job.output += envLog
        
        // Create process in a new process group so we can kill the entire tree
        process.qualityOfService = .userInitiated
        
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        // Capture output - use actor-isolated storage for thread-safe access
        actor ErrorDataStore {
            private var data = Data()
            
            func append(_ newData: Data) {
                data.append(newData)
            }
            
            func getData() -> Data {
                return data
            }
        }
        
        let errorDataStore = ErrorDataStore()
        
        outputPipe.fileHandleForReading.readabilityHandler = { [weak job] handle in
            let data = handle.availableData
            if !data.isEmpty {
                if let text = String(data: data, encoding: .utf8), let job = job {
                    Task { @MainActor in
                        job.output += text
                    }
                }
            }
        }
        
        errorPipe.fileHandleForReading.readabilityHandler = { [errorDataStore, weak job] handle in
            let data = handle.availableData
            if !data.isEmpty {
                Task {
                    await errorDataStore.append(data)
                }
                if let text = String(data: data, encoding: .utf8), let job = job {
                    Task { @MainActor in
                        job.output += text
                    }
                }
            }
        }
        
        process.terminationHandler = { [weak self, errorDataStore, weak job] process in
            outputPipe.fileHandleForReading.readabilityHandler = nil
            errorPipe.fileHandleForReading.readabilityHandler = nil
            
            guard let job = job, let self = self else { return }
            
            // Skip processing if app is shutting down to avoid "task name port right" errors
            // We need to check this on MainActor since isShuttingDown is @MainActor
            Task { @MainActor in
                // Check shutdown flag first - if shutting down, just clear and return immediately
                // This prevents accessing process.terminationStatus which causes the error
                if self.isShuttingDown {
                    job.process = nil
                    return
                }
                
                // Only access termination status if not shutting down
                let exitCode = process.terminationStatus
                
                if exitCode == 0 {
                    job.status = .completed
                    
                    // Check if post-training quantization is enabled
                    var quantizationSucceeded = false
                    if job.config.enablePostTrainingQuantization, let quantization = job.config.postTrainingQuantization {
                        // Run quantization after successful training
                        quantizationSucceeded = await self.runPostTrainingQuantization(job: job, quantization: quantization, venvPython: venvPython, hfToken: hfToken)
                    }
                    
                    // Check if GGUF conversion is enabled (after quantization if enabled).
                    // Run GGUF conversion on a detached background task so the main thread
                    // stays responsive (no spinning beach ball during long conversions).
                    if job.config.enableGGUFConversion {
                        Task.detached { [weak self, weak job] in
                            guard let self = self, let job = job else { return }
                            await self.runGGUFConversion(job: job, venvPython: venvPython, hfToken: hfToken, quantizationSucceeded: quantizationSucceeded)
                        }
                    }
                } else {
                    job.status = .failed
                    let errorData = await errorDataStore.getData()
                    if let errorText = String(data: errorData, encoding: .utf8) {
                        // Parse error to provide better user-friendly messages
                        let friendlyError = self.parseTrainingError(errorText, modelId: job.config.model, trainingMode: job.config.trainMode, dataPath: job.config.data)
                        job.error = friendlyError
                        // Trigger alert for this failed job
                        self.failedJob = job
                    }
                }
                job.completedAt = Date()
                job.process = nil
                
                // Only save if not shutting down (saving during shutdown can cause issues)
                if !self.isShuttingDown {
                    try? self.saveJob(job)
                }
            }
        }
        
        job.process = process
        
        do {
            try process.run()
            
            // Set up process group after starting (on macOS, this helps with killing child processes)
            // Note: Process groups are automatically created, but we ensure we can kill the tree
            
            try saveJob(job)
        } catch {
            job.status = .failed
            job.error = error.localizedDescription
            job.process = nil
        }
    }
    
    func stopJob(_ job: TrainingJob) {
        // Cancel all active download tasks first
        // Cancel all download tasks
        for task in job.activeDownloadTasks {
            task.cancel()
        }
        job.activeDownloadTasks.removeAll()
        
        // Invalidate all download sessions
        for session in job.activeDownloadSessions {
            session.invalidateAndCancel()
        }
        job.activeDownloadSessions.removeAll()
        
        // Update download UI state
        job.isDownloading = false
        job.downloadStatus = "Download cancelled"
        
        guard let process = job.process else {
            // If no process, just update status
            if job.status == .running || job.status == .paused {
                job.status = .cancelled
                job.completedAt = Date()
                try? saveJob(job)
            }
            return
        }
        
        // Capture process information BEFORE doing anything else
        let processID = process.processIdentifier
        let wasRunning = process.isRunning
        
        // Clear the process reference immediately to avoid accessing it after termination
        job.process = nil
        
        // Try graceful termination first using the captured PID
        if wasRunning {
            kill(processID, SIGTERM)
            
            // Wait a short time for graceful termination
            // Check if process still exists using kill(pid, 0) which doesn't kill, just checks
            var stillRunning = true
            let deadline = Date().addingTimeInterval(2.0) // 2 second timeout
            var checkCount = 0
            while stillRunning && Date() < deadline && checkCount < 20 {
                // Use kill(pid, 0) to check if process exists without killing it
                let result = kill(processID, 0)
                stillRunning = (result == 0) // 0 means process exists
                if stillRunning {
                    RunLoop.current.run(until: Date().addingTimeInterval(0.1))
                    checkCount += 1
                }
            }
            
            // If still running, force kill using the captured PID
            if stillRunning {
                // Use the PID directly instead of calling forceKillJob to avoid process access
                kill(processID, SIGKILL)
                let pgid = getpgid(processID)
                if pgid > 0 && pgid == processID {
                    kill(-processID, SIGKILL)
                }
            }
        }
        
        // Clean up
        job.status = .cancelled
        job.completedAt = Date()
        try? saveJob(job)
    }
    
    /// Stop all running jobs (called on app termination)
    func stopAllRunningJobs() {
        // Set shutdown flag first to prevent termination handlers from processing
        isShuttingDown = true
        
        for job in jobs {
            if job.status == .running || job.status == .paused {
                // Clear termination handler first to prevent it from being called
                // during shutdown (which causes "task name port right" errors)
                if let process = job.process {
                    process.terminationHandler = nil
                }
                forceKillJob(job)
            }
        }
    }
    
    func forceKillJob(_ job: TrainingJob) {
        guard let process = job.process else {
            // If no process, just update status
            if job.status == .running || job.status == .paused {
                job.status = .cancelled
                job.completedAt = Date()
                try? saveJob(job)
            }
            return
        }
        
        // Capture process information BEFORE doing anything else
        // This avoids accessing process properties after termination which can cause
        // "task name port right" errors
        // Access these properties while the process object is still valid
        let processID = process.processIdentifier
        let wasRunning = process.isRunning
        
        // Clear the process reference immediately to avoid accessing it after termination
        // This prevents "task name port right" errors when the app quits
        job.process = nil
        
        // Force kill the process using the PID we captured (not the process object)
        if wasRunning {
            // First, try to terminate gracefully (in case it responds quickly)
            kill(processID, SIGTERM)
            
            // Wait briefly
            usleep(100000) // 0.1 seconds
            
            // Immediately send SIGKILL to force termination
            // This kills the process and its direct children
            kill(processID, SIGKILL)
            
            // Also try to kill the process group if possible
            // Negative PID means kill the process group
            // Note: This only works if the process is the group leader
            let pgid = getpgid(processID)
            if pgid > 0 && pgid == processID {
                // Process is group leader, kill the whole group
                kill(-processID, SIGKILL)
            }
        }
        
        // Clean up
        job.status = .cancelled
        job.completedAt = Date()
        if let existingError = job.error, !existingError.isEmpty {
            job.error = existingError + "\n[Process was forcefully terminated]"
        } else {
            job.error = "[Process was forcefully terminated]"
        }
        try? saveJob(job)
    }
    
    func pauseJob(_ job: TrainingJob) {
        guard job.status == .running, let process = job.process else { return }
        process.suspend()
        job.status = .paused
        try? saveJob(job)
    }
    
    func resumeJob(_ job: TrainingJob) {
        guard job.status == .paused, let process = job.process else { return }
        process.resume()
        job.status = .running
        try? saveJob(job)
    }
    
    func cancelJob(_ job: TrainingJob) {
        // Cancel a pending job (before it starts)
        guard job.status == .pending else { return }
        job.status = .cancelled
        job.completedAt = Date()
        try? saveJob(job)
    }
    
    func deleteJob(_ job: TrainingJob) {
        stopJob(job)
        jobs.removeAll { $0.id == job.id }
        
        // Delete job file
        let jobFile = jobsDirectory.appendingPathComponent("\(job.id.uuidString).json")
        try? FileManager.default.removeItem(at: jobFile)
        
        // Delete associated config file
        if FileManager.default.fileExists(atPath: job.configPath.path) {
            try? FileManager.default.removeItem(at: job.configPath)
        }
    }
    
    // MARK: - Persistence
    
    private func saveJob(_ job: TrainingJob) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let persisted = job.persistedData
        let data = try encoder.encode(persisted)
        let jobFile = jobsDirectory.appendingPathComponent("\(job.id.uuidString).json")
        try data.write(to: jobFile)
    }
    
    private func loadJobs() {
        guard let files = try? FileManager.default.contentsOfDirectory(at: jobsDirectory, includingPropertiesForKeys: nil) else {
            return
        }
        
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        
        // Track which config files are referenced by jobs
        var referencedConfigPaths = Set<String>()
        
        for file in files where file.pathExtension == "json" {
            if let data = try? Data(contentsOf: file),
               let persisted = try? decoder.decode(TrainingJob.PersistedJob.self, from: data) {
                let job = TrainingJob(from: persisted)
                jobs.append(job)
                // Track the config path
                referencedConfigPaths.insert(job.configPath.path)
            }
        }
        
        // Sort by creation date (newest first)
        jobs.sort { $0.createdAt > $1.createdAt }
        
        // Clean up orphaned config files (configs without corresponding jobs)
        cleanupOrphanedConfigs(referencedConfigPaths: referencedConfigPaths)
        
        // Clean up invalid job files (empty, corrupted, or referencing non-existent configs)
        cleanupInvalidJobs()
    }
    
    /// Remove config files that are not referenced by any existing jobs
    private func cleanupOrphanedConfigs(referencedConfigPaths: Set<String>) {
        guard let configFiles = try? FileManager.default.contentsOfDirectory(at: configsDirectory, includingPropertiesForKeys: nil) else {
            return
        }
        
        for configFile in configFiles where configFile.pathExtension == "yaml" || configFile.pathExtension == "yml" {
            // Check if this config file is referenced by any job
            if !referencedConfigPaths.contains(configFile.path) {
                // Check if file is empty or orphaned
                if let attributes = try? FileManager.default.attributesOfItem(atPath: configFile.path),
                   let fileSize = attributes[.size] as? Int64 {
                    // Remove empty files (always remove these)
                    if fileSize == 0 {
                        try? FileManager.default.removeItem(at: configFile)
                    } else {
                        // Also remove orphaned config files (not referenced by any job)
                        // These are leftover from deleted jobs
                        try? FileManager.default.removeItem(at: configFile)
                    }
                } else {
                    // If we can't get file attributes, try to remove anyway if it's orphaned
                    try? FileManager.default.removeItem(at: configFile)
                }
            }
        }
    }
    
    /// Remove invalid job files (empty, corrupted JSON, or referencing non-existent configs)
    /// Parse training errors and provide user-friendly error messages
    private func parseTrainingError(_ errorText: String, modelId: String, trainingMode: TrainingMode, dataPath: String) -> String {
        // Check for model not found errors
        if errorText.contains("RepositoryNotFoundError") || 
           errorText.contains("404 Client Error") ||
           errorText.contains("Repository Not Found") {
            return """
            ❌ Model Not Found Error
            
            The model "\(modelId)" could not be found on Hugging Face.
            
            Possible reasons:
            • The model identifier is incorrect or misspelled
            • The model doesn't exist (some non-quantized models may not be available)
            • The model is private or gated (requires authentication)
            
            Suggestions:
            • Verify the model name at https://huggingface.co/models
            • Check if you need a quantized version (e.g., add "-4bit" or "-8bit" to the model name)
            • Ensure your HF_TOKEN is set correctly if the model requires authentication
            """
        }
        
        // Check for authentication errors
        if errorText.contains("401") || errorText.contains("Unauthorized") || errorText.contains("authentication") {
            return """
            ❌ Authentication Error
            
            Failed to authenticate with Hugging Face.
            
            Possible reasons:
            • Invalid or missing HF_TOKEN
            • Token doesn't have access to this model
            • Model is private and requires authentication
            
            Suggestions:
            • Check your HF_TOKEN in Settings
            • Verify the token is valid at https://huggingface.co/settings/tokens
            • Ensure the token has read access to the model
            """
        }
        
        // Check for permission errors
        if errorText.contains("403") || errorText.contains("Forbidden") {
            return """
            ❌ Permission Denied Error
            
            You don't have permission to access this model.
            
            Possible reasons:
            • The model is gated and requires acceptance of terms
            • Your HF_TOKEN doesn't have sufficient permissions
            • The model is private and you're not authorized
            
            Suggestions:
            • Visit the model page on Hugging Face and accept any terms
            • Check your HF_TOKEN permissions
            • Verify you have access to the model repository
            """
        }
        
        // Check for data format errors - KeyError (missing required keys)
        if errorText.contains("KeyError") {
            let missingKey = extractMissingKey(from: errorText)
            return formatDataKeyError(missingKey: missingKey, trainingMode: trainingMode, dataPath: dataPath)
        }
        
        // Check for unsupported data format errors
        if errorText.contains("Unsupported data format") || errorText.contains("ValueError") && errorText.contains("data format") {
            return formatUnsupportedDataFormatError(trainingMode: trainingMode, dataPath: dataPath)
        }
        
        // Check for JSON decode errors (malformed JSON, empty lines)
        if errorText.contains("JSONDecodeError") || errorText.contains("json.decoder.JSONDecodeError") || errorText.contains("Expecting value") {
            return formatJSONDecodeError(trainingMode: trainingMode, dataPath: dataPath)
        }
        
        // Check for file not found errors
        if errorText.contains("FileNotFoundError") || errorText.contains("No such file or directory") || errorText.contains("cannot find") {
            return formatFileNotFoundError(dataPath: dataPath)
        }
        
        // Return original error if no specific pattern matched
        return errorText
    }
    
    /// Extract the missing key from a KeyError message
    private func extractMissingKey(from errorText: String) -> String? {
        // Pattern: KeyError: 'answer' or KeyError: "answer"
        let patterns = [
            "KeyError: ['\"]([^'\"]+)['\"]",
            "KeyError: ([a-zA-Z_][a-zA-Z0-9_]*)",
            "Key '([^']+)'",
            "Key \"([^\"]+)\""
        ]
        
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: []),
               let match = regex.firstMatch(in: errorText, options: [], range: NSRange(location: 0, length: errorText.utf16.count)),
               match.numberOfRanges > 1,
               let range = Range(match.range(at: 1), in: errorText) {
                return String(errorText[range])
            }
        }
        
        return nil
    }
    
    /// Format a user-friendly error message for missing data keys
    private func formatDataKeyError(missingKey: String?, trainingMode: TrainingMode, dataPath: String) -> String {
        let keyName = missingKey ?? "unknown key"
        
        // Determine expected format based on training mode
        let expectedFormat: String
        let exampleFormat: String
        let commonMistake: String
        
        switch trainingMode {
        case .sft:
            expectedFormat = """
            SFT supports multiple formats (use one of these):
            • {"prompt": "...", "completion": "..."}
            • {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
            • {"text": "..."}
            """
            exampleFormat = """
            Examples:
            {"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}
            {"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}]}
            """
            commonMistake = "Make sure you're using 'completion', 'messages', or 'text' (not 'answer' or 'chosen')"
            
        case .dpo, .cpo:
            expectedFormat = """
            DPO/CPO requires this format:
            • {"prompt": "...", "chosen": "...", "rejected": "..."}
            • Optional: {"system": "...", "prompt": "...", "chosen": "...", "rejected": "..."}
            """
            exampleFormat = """
            Example:
            {"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "France doesn't have a capital."}
            """
            commonMistake = "Make sure you have both 'chosen' and 'rejected' keys (not 'answer' or 'completion')"
            
        case .orpo:
            expectedFormat = """
            ORPO requires this format:
            • {"prompt": "...", "chosen": "...", "rejected": "..."}
            • Optional: {"system": "...", "prompt": "...", "chosen": "...", "rejected": "..."}
            """
            exampleFormat = """
            Example:
            {"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "France doesn't have a capital."}
            """
            commonMistake = "Make sure you have both 'chosen' and 'rejected' keys (same format as DPO)"
            
        case .grpo, .gspo, .drGrpo, .dapo:
            expectedFormat = """
            GRPO/GSPO/Dr. GRPO/DAPO requires this format:
            • {"prompt": "...", "answer": "..."}
            • Optional: {"system": "...", "prompt": "...", "answer": "..."}
            """
            exampleFormat = """
            Example:
            {"prompt": "What is the capital of France?", "answer": "The capital of France is Paris."}
            """
            commonMistake = "Make sure you're using 'answer' (not 'chosen'/'rejected' or 'completion')"
            
        case .onlineDpo, .xpo:
            expectedFormat = """
            \(trainingMode.displayName) requires this format:
            • {"prompt": "...", "chosen": "...", "rejected": "..."}
            • Or messages format: {"prompt": [{"role": "user", "content": "..."}]}
            
            Note: These modes require a judge model and may use preference data.
            """
            exampleFormat = """
            Example (preference format):
            {"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "France doesn't have a capital."}
            """
            commonMistake = "Verify you're using preference format with 'chosen' and 'rejected', or messages format"
            
        case .rlhfReinforce, .ppo:
            expectedFormat = """
            \(trainingMode.displayName) requires this format:
            • {"prompt": "...", "answer": "..."}
            • Or messages format: {"prompt": [{"role": "user", "content": "..."}]}
            
            Note: These modes require a reward/judge model and generate responses during training.
            """
            exampleFormat = """
            Example:
            {"prompt": "What is the capital of France?", "answer": "The capital of France is Paris."}
            """
            commonMistake = "Verify the data format - may use 'answer' or messages format depending on implementation"
        }
        
        return """
        ❌ Data Format Error - Missing Required Key
        
        The training data is missing the required key: '\(keyName)'
        
        Training Mode: \(trainingMode.displayName)
        Data Path: \(dataPath)
        
        \(expectedFormat)
        
        \(exampleFormat)
        
        Common mistake:
        \(commonMistake)
        
        How to fix:
        1. Check your JSONL files (train.jsonl, valid.jsonl, test.jsonl) in the data directory
        2. Ensure each line is a valid JSON object with the required keys
        3. Make sure there are no empty lines in your JSONL files
        4. Verify the format matches the training mode requirements above
        """
    }
    
    /// Format a user-friendly error message for JSON decode errors
    private func formatJSONDecodeError(trainingMode: TrainingMode, dataPath: String) -> String {
        return """
        ❌ JSON Format Error
        
        The training data contains invalid JSON or formatting issues.
        
        Training Mode: \(trainingMode.displayName)
        Data Path: \(dataPath)
        
        Common causes:
        • Empty lines in JSONL files (each line must be a valid JSON object)
        • Malformed JSON (missing quotes, brackets, commas, etc.)
        • Trailing commas in JSON objects
        • Invalid escape sequences in strings
        
        How to fix:
        1. Check your JSONL files (train.jsonl, valid.jsonl, test.jsonl)
        2. Remove any empty lines
        3. Validate each line is valid JSON (you can use a JSON validator)
        4. Ensure proper escaping of special characters in strings
        
        Example of valid JSONL format:
        {"prompt": "Question?", "completion": "Answer."}
        {"prompt": "Another question?", "completion": "Another answer."}
        
        Note: Each line must be a complete, valid JSON object. No empty lines allowed.
        """
    }
    
    /// Format a user-friendly error message for file not found errors
    private func formatFileNotFoundError(dataPath: String) -> String {
        return """
        ❌ Data File Not Found
        
        The training data files could not be found.
        
        Data Path: \(dataPath)
        
        Required files:
        • train.jsonl (training data)
        • valid.jsonl (validation data)
        • test.jsonl (test data)
        
        How to fix:
        1. Verify the data path is correct
        2. Ensure the directory contains train.jsonl, valid.jsonl, and test.jsonl
        3. Check that file names are exactly as shown (case-sensitive)
        4. Make sure you have read permissions for the directory
        
        Note: The validation file must be named 'valid.jsonl' (not 'validation.jsonl')
        """
    }
    
    /// Format a user-friendly error message for unsupported data format errors
    private func formatUnsupportedDataFormatError(trainingMode: TrainingMode, dataPath: String) -> String {
        let expectedFormat: String
        let exampleFormat: String
        let commonIssues: String
        
        switch trainingMode {
        case .sft:
            expectedFormat = """
            SFT requires one of these formats:
            • {"prompt": "...", "completion": "..."}
            • {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
            • {"text": "full text content"}
            """
            exampleFormat = """
            Valid examples:
            {"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}
            {"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}]}
            {"text": "Question: What is the capital of France? Answer: Paris."}
            """
            commonIssues = """
            Common issues:
            • Using wrong key names (e.g., "answer" instead of "completion" for prompt/completion format)
            • Mixing formats in the same file (all lines must use the same format)
            • Empty or null values in required fields
            • Invalid JSON structure in messages array
            • Missing required keys (prompt/completion, messages, or text)
            """
            
        case .dpo, .cpo:
            expectedFormat = """
            DPO/CPO requires this format:
            • {"prompt": "...", "chosen": "...", "rejected": "..."}
            • Optional: {"system": "...", "prompt": "...", "chosen": "...", "rejected": "..."}
            """
            exampleFormat = """
            Valid example:
            {"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "France doesn't have a capital."}
            """
            commonIssues = """
            Common issues:
            • Missing "chosen" or "rejected" keys
            • Using "completion" or "answer" instead of "chosen"/"rejected"
            • Empty values in chosen or rejected fields
            """
            
        case .grpo, .gspo, .drGRPO, .dapo:
            expectedFormat = """
            \(trainingMode.displayName) requires this format:
            • {"prompt": "...", "answer": "..."}
            """
            exampleFormat = """
            Valid example:
            {"prompt": "What is the capital of France?", "answer": "The capital of France is Paris."}
            """
            commonIssues = """
            Common issues:
            • Using "completion" instead of "answer"
            • Missing "answer" key
            • Empty answer values
            """
            
        default:
            expectedFormat = """
            Please refer to the mlx-lm-lora documentation for the required data format for \(trainingMode.displayName).
            """
            exampleFormat = ""
            commonIssues = """
            Check the data format requirements for \(trainingMode.displayName) in the README or mlx-lm-lora documentation.
            """
        }
        
        return """
        ❌ Unsupported Data Format Error
        
        The training data format is not supported for \(trainingMode.displayName).
        
        Training Mode: \(trainingMode.displayName)
        Data Path: \(dataPath)
        
        \(expectedFormat)
        
        \(exampleFormat)
        
        \(commonIssues)
        
        How to fix:
        1. Open your JSONL files (train.jsonl, valid.jsonl, test.jsonl) in a text editor
        2. Verify each line matches one of the supported formats above
        3. Ensure all lines in a file use the same format (don't mix formats)
        4. Check that all required keys are present and have non-empty values
        5. Validate that each line is valid JSON (no syntax errors)
        6. Remove any empty lines from your JSONL files
        
        Note: The format must be consistent across all files (train.jsonl, valid.jsonl, test.jsonl).
        """
    }
    
    private func cleanupInvalidJobs() {
        guard let jobFiles = try? FileManager.default.contentsOfDirectory(at: jobsDirectory, includingPropertiesForKeys: nil) else {
            return
        }
        
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        
        // Track valid job IDs from loaded jobs
        let validJobIds = Set(jobs.map { $0.id.uuidString })
        
        for jobFile in jobFiles where jobFile.pathExtension == "json" {
            let jobId = jobFile.deletingPathExtension().lastPathComponent
            
            // Check if this job file corresponds to a loaded job
            if !validJobIds.contains(jobId) {
                // This job file wasn't loaded (likely corrupted or invalid)
                try? FileManager.default.removeItem(at: jobFile)
                continue
            }
            
            // Check if file is empty
            if let attributes = try? FileManager.default.attributesOfItem(atPath: jobFile.path),
               let fileSize = attributes[.size] as? Int64,
               fileSize == 0 {
                // Remove empty job files
                try? FileManager.default.removeItem(at: jobFile)
                continue
            }
            
            // Check if job references a non-existent config file
            if let data = try? Data(contentsOf: jobFile),
               let persisted = try? decoder.decode(TrainingJob.PersistedJob.self, from: data) {
                let configPath = URL(fileURLWithPath: persisted.configPath)
                if !FileManager.default.fileExists(atPath: configPath.path) {
                    // Job references a non-existent config file - remove the orphaned job
                    try? FileManager.default.removeItem(at: jobFile)
                    // Also remove from jobs array if it was loaded
                    jobs.removeAll { $0.id.uuidString == jobId }
                }
            }
        }
    }
    
    // MARK: - Shell Environment Loading
    
    private func loadShellEnvironment() async -> [String: String] {
        // Try to get the user's shell
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/zsh"
        let shellName = (shell as NSString).lastPathComponent
        
        // Determine which config file to source based on shell
        let configFile: String
        if shellName == "zsh" {
            configFile = "~/.zshrc"
        } else if shellName == "bash" {
            configFile = "~/.bash_profile"
        } else {
            configFile = "~/.profile"
        }
        
        let expandedConfig = (configFile as NSString).expandingTildeInPath
        
        // Check if config file exists
        guard FileManager.default.fileExists(atPath: expandedConfig) else {
            return [:]
        }
        
        // Use shell to source config in non-interactive mode and print environment
        // Use -l flag for login shell to ensure proper initialization, but run in non-interactive mode
        let command = """
        if [ -f \(expandedConfig) ]; then
            source \(expandedConfig) 2>/dev/null
        fi
        env
        """
        
        return await withCheckedContinuation { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: shell)
            process.arguments = ["-c", command]
            
            let pipe = Pipe()
            let errorPipe = Pipe()
            process.standardOutput = pipe
            process.standardError = errorPipe
            
            // Set a timeout to prevent hanging - use actor for thread-safe access
            actor TimeoutFlag {
                private var value = false
                
                func set() {
                    value = true
                }
                
                func get() -> Bool {
                    return value
                }
            }
            
            let timeoutFlag = TimeoutFlag()
            let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global())
            timer.schedule(deadline: .now() + 2.0) // 2 second timeout
            timer.setEventHandler {
                if !process.isRunning {
                    timer.cancel()
                    return
                }
                Task {
                    await timeoutFlag.set()
                }
                process.terminate()
                timer.cancel()
            }
            timer.resume()
            
            process.terminationHandler = { _ in
                timer.cancel()
                
                Task {
                    let timedOut = await timeoutFlag.get()
                    guard !timedOut else {
                        continuation.resume(returning: [:])
                        return
                    }
                    
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    guard let output = String(data: data, encoding: .utf8) else {
                        continuation.resume(returning: [:])
                        return
                    }
                    
                    // Parse environment variables from output
                    var envVars: [String: String] = [:]
                    let lines = output.components(separatedBy: .newlines)
                    
                    for line in lines {
                        let trimmed = line.trimmingCharacters(in: .whitespaces)
                        guard !trimmed.isEmpty else { continue }
                        
                        // Split on first '=' to separate key and value
                        if let equalsIndex = trimmed.firstIndex(of: "=") {
                            let key = String(trimmed[..<equalsIndex])
                            let value = String(trimmed[trimmed.index(after: equalsIndex)...])
                            envVars[key] = value
                        }
                    }
                    
                    continuation.resume(returning: envVars)
                }
            }
            
            do {
                try process.run()
            } catch {
                timer.cancel()
                continuation.resume(returning: [:])
            }
        }
    }
    
    // MARK: - Model Download
    
    /// Pre-downloads a model from Hugging Face with progress tracking using Swift/URLSession
    /// This provides better UX with native SwiftUI progress indicators
    private func downloadModelIfNeeded(modelId: String, hfToken: String?, job: TrainingJob, venvPython: URL) async {
        // First check if model is already cached using Python (simpler check)
        let isCached = await checkModelCache(modelId: modelId, hfToken: hfToken, venvPython: venvPython)
        
        if isCached {
            await MainActor.run {
                job.output += "\n[INFO] Model already cached, skipping download...\n\n"
            }
            return
        }
        
        // Download using Swift/URLSession for better progress tracking
        await downloadModelWithSwift(modelId: modelId, hfToken: hfToken, job: job)
    }
    
    /// Checks if model is already cached
    private func checkModelCache(modelId: String, hfToken: String?, venvPython: URL) async -> Bool {
        // Native Swift implementation - check HF_HOME/hub/models--org--name/snapshots/
        let shellEnv = await loadShellEnvironment()
        let hfHome = shellEnv["HF_HOME"] ?? (FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface").path)
        
        let cacheDir = URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        let modelCacheDir = cacheDir.appendingPathComponent("models--\(modelId.replacingOccurrences(of: "/", with: "--"))")
        let snapshotsDir = modelCacheDir.appendingPathComponent("snapshots")
        
        // Check if snapshots directory exists and has at least one subdirectory
        guard let contents = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path) else {
            return false
        }
        
        // Check for at least one snapshot with model files
        for item in contents {
            let snapshotPath = snapshotsDir.appendingPathComponent(item)
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: snapshotPath.path, isDirectory: &isDirectory), isDirectory.boolValue {
                // Check if snapshot has any files (config.json, model files, etc.)
                if let snapshotContents = try? FileManager.default.contentsOfDirectory(atPath: snapshotPath.path), !snapshotContents.isEmpty {
                    return true
                }
            }
        }
        
        return false
    }
    
    /// Progress tracker actor for thread-safe progress tracking
    private actor ProgressTracker {
        var downloadedCount: Int = 0
        var downloadedBytes: Int64 = 0
        // Track partial progress for files currently downloading
        var partialProgress: [Int: Int64] = [:] // fileIndex -> partial bytes
        
        func addDownload(success: Bool, bytes: Int64, fileIndex: Int? = nil) {
            if success {
                downloadedCount += 1
                downloadedBytes += bytes
                // Clear partial progress for this file since it's now complete
                if let index = fileIndex {
                    partialProgress.removeValue(forKey: index)
                }
            }
        }
        
        func updatePartialProgress(fileIndex: Int, bytes: Int64) {
            partialProgress[fileIndex] = bytes
        }
        
        func getProgress() -> (count: Int, bytes: Int64) {
            // Calculate total including partial progress for files currently downloading
            let partialTotal = partialProgress.values.reduce(0, +)
            // Total bytes = completed files + partial progress of in-progress files
            // Note: We don't subtract old partial progress because we're tracking current bytes, not incremental
            let totalBytes = downloadedBytes + partialTotal
            return (downloadedCount, totalBytes)
        }
        
        func getCurrentBytes() -> Int64 {
            // Get just the bytes for progress calculation
            let partialTotal = partialProgress.values.reduce(0, +)
            return downloadedBytes + partialTotal
        }
        
        func clearPartialProgress(fileIndex: Int) {
            partialProgress.removeValue(forKey: fileIndex)
        }
    }
    
    /// Download delegate to track progress for large files
    private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        var progressTracker: ProgressTracker?
        let job: TrainingJob
        let fileName: String
        let currentFile: Int
        let totalFiles: Int
        let totalBytes: Int64
        let fileSize: Int64
        let fileIndex: Int
        let destination: URL
        // Store continuation for async/await support
        var continuation: CheckedContinuation<(success: Bool, bytesDownloaded: Int64), Never>?
        // Use a serial queue for thread-safe access to lastUpdate
        private let updateQueue = DispatchQueue(label: "com.mlxtraining.download.progress")
        private var _lastUpdate = Date()
        var lastUpdate: Date {
            get {
                return updateQueue.sync { _lastUpdate }
            }
            set {
                updateQueue.sync { _lastUpdate = newValue }
            }
        }
        let updateInterval: TimeInterval = 0.1 // Update very frequently (every 0.1 seconds) for smooth progress
        
        init(progressTracker: ProgressTracker?, job: TrainingJob, fileName: String, currentFile: Int, totalFiles: Int, totalBytes: Int64, fileSize: Int64, fileIndex: Int, destination: URL) {
            self.progressTracker = progressTracker
            self.job = job
            self.fileName = fileName
            self.currentFile = currentFile
            self.totalFiles = totalFiles
            self.totalBytes = totalBytes
            self.fileSize = fileSize
            self.fileIndex = fileIndex
            self.destination = destination
        }
        
        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
            // Update progress more aggressively - check time but don't skip if it's been a while
            let now = Date()
            let timeSinceLastUpdate = now.timeIntervalSince(lastUpdate)
            guard timeSinceLastUpdate >= updateInterval else { return }
            lastUpdate = now
            
            // Use actual bytes from HTTP response - this is the real file size
            let actualFileSize = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : Int64(fileSize)
            let fileProgress = actualFileSize > 0 ? Double(totalBytesWritten) / Double(actualFileSize) : 0.0
            let filePercent = Int(fileProgress * 100)
            
            // Use actual bytes written directly from the HTTP response
            // This is more accurate than calculating from fileSize
            let partialBytes = totalBytesWritten
            
            Task {
                // Update the progress tracker with partial progress
                if let tracker = progressTracker {
                    // Update partial progress for this file using actual bytes
                    await tracker.updatePartialProgress(fileIndex: fileIndex, bytes: partialBytes)
                    
                    // Get current overall progress
                    let (_, currentBytes) = await tracker.getProgress()
                    let overallProgress = totalBytes > 0 ? Double(currentBytes) / Double(totalBytes) : 0.0
                    let overallPercent = Int(overallProgress * 100)
                    
                    await MainActor.run {
                        // Update progress bar in real-time - this is the key fix!
                        job.downloadProgress = min(overallProgress, 0.99) // Cap at 99% until all files complete
                        job.downloadStatus = "Downloading: \(currentFile)/\(totalFiles) files - \(fileName) (\(filePercent)%) - Overall: \(overallPercent)%"
                    }
                } else {
                    // Fallback if tracker not available - estimate progress from file progress
                    let estimatedOverall = totalBytes > 0 ? Double(partialBytes) / Double(totalBytes) : 0.0
                    await MainActor.run {
                        job.downloadProgress = min(estimatedOverall, 0.99)
                        job.downloadStatus = "Downloading: \(currentFile)/\(totalFiles) files - \(fileName) (\(filePercent)%)"
                    }
                }
            }
        }
        
        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
            // Remove task from active list
            Task { @MainActor in
                if let index = job.activeDownloadTasks.firstIndex(where: { $0 === downloadTask }) {
                    job.activeDownloadTasks.remove(at: index)
                }
            }
            
            // File downloaded successfully - handle completion here
            // Clear partial progress since file is complete
            Task {
                if let tracker = progressTracker {
                    await tracker.clearPartialProgress(fileIndex: fileIndex)
                }
            }
            
            // Check HTTP response status
            if let httpResponse = downloadTask.response as? HTTPURLResponse {
                if httpResponse.statusCode == 401 {
                    Task { @MainActor in
                        job.output += "  ✗ Authentication failed (401) - check HF_TOKEN\n"
                    }
                    continuation?.resume(returning: (false, 0))
                    return
                } else if httpResponse.statusCode == 403 {
                    Task { @MainActor in
                        job.output += "  ✗ Forbidden (403) - model may be gated or token invalid\n"
                    }
                    continuation?.resume(returning: (false, 0))
                    return
                } else if httpResponse.statusCode != 200 {
                    Task { @MainActor in
                        job.output += "  ✗ HTTP error: \(httpResponse.statusCode)\n"
                    }
                    continuation?.resume(returning: (false, 0))
                    return
                }
            }
            
            do {
                // Move downloaded file to final destination
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try FileManager.default.moveItem(at: location, to: destination)
                
                // Get actual file size from disk
                let fileAttributes = try? FileManager.default.attributesOfItem(atPath: destination.path)
                let downloadedSize = (fileAttributes?[.size] as? Int64) ?? 0
                
                // Resume continuation with success
                continuation?.resume(returning: (true, downloadedSize))
            } catch {
                Task { @MainActor in
                    job.output += "  ✗ Error moving file: \(error.localizedDescription)\n"
                }
                continuation?.resume(returning: (false, 0))
            }
        }
        
        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            // Remove task from active list
            Task { @MainActor in
                if let downloadTask = task as? URLSessionDownloadTask {
                    if let index = job.activeDownloadTasks.firstIndex(where: { $0 === downloadTask }) {
                        job.activeDownloadTasks.remove(at: index)
                    }
                }
            }
            
            // Handle errors - only resume if not already resumed by didFinishDownloadingTo
            if let error = error {
                // Don't report cancellation as an error
                let nsError = error as NSError
                if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled {
                    // Task was cancelled - this is expected when user stops the job
                    continuation?.resume(returning: (false, 0))
                    continuation = nil
                    return
                }
                
                Task { @MainActor in
                    job.output += "  ✗ Error: \(error.localizedDescription)\n"
                }
                // Only resume if continuation hasn't been used yet
                if continuation != nil {
                    continuation?.resume(returning: (false, 0))
                    continuation = nil
                }
            }
            // Note: Success is handled in didFinishDownloadingTo
        }
    }
    
    /// Creates an optimized URLSession for downloads with delegate support
    private func createDownloadSession(delegate: URLSessionDownloadDelegate) -> URLSession {
        // Use ephemeral configuration for better performance (no disk caching, in-memory only)
        let configuration = URLSessionConfiguration.ephemeral
        // Optimize for downloads
        configuration.timeoutIntervalForRequest = 300 // 5 minutes
        configuration.timeoutIntervalForResource = 3600 // 1 hour for large files
        configuration.httpMaximumConnectionsPerHost = 15 // Increased for better throughput (was 10)
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData // Always download fresh
        configuration.urlCache = nil // Disable cache for downloads
        configuration.networkServiceType = .responsiveData // Optimize for responsive data transfer
        configuration.waitsForConnectivity = false // Fail fast if no connection
        configuration.allowsCellularAccess = true // Allow cellular if available
        // Note: HTTP/2 and HTTP/3 are used automatically by URLSession, no need for pipelining
        return URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
    }
    
    /// Downloads model using Swift/URLSession with progress tracking
    private func downloadModelWithSwift(modelId: String, hfToken: String?, job: TrainingJob) async {
        await MainActor.run {
            job.isDownloading = true
            job.downloadProgress = 0.0
            job.downloadStatus = "Fetching file list..."
            job.output += "\n=== Model Download ===\n"
            job.output += "[INFO] Downloading model: \(modelId)\n"
            job.output += "[INFO] Fetching file list from Hugging Face...\n"
            if let token = hfToken, !token.isEmpty {
                job.output += "[INFO] Using HF_TOKEN for authentication\n"
            } else {
                job.output += "[INFO] No HF_TOKEN provided (using public access)\n"
            }
        }
        
        // Get cache directory
        let shellEnv = await loadShellEnvironment()
        let hfHome = shellEnv["HF_HOME"] ?? (FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface").path)
        let cacheDir = URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        let modelCacheDir = cacheDir.appendingPathComponent("models--\(modelId.replacingOccurrences(of: "/", with: "--"))")
        let snapshotsDir = modelCacheDir.appendingPathComponent("snapshots")
        
        // Create directories if needed
        try? FileManager.default.createDirectory(at: snapshotsDir, withIntermediateDirectories: true)
        
        // Get model file list from Hugging Face API
        guard let fileListResult = await getModelFileList(modelId: modelId, hfToken: hfToken) else {
            await MainActor.run {
                job.output += "[WARNING] Could not fetch file list. Training will download model automatically...\n\n"
            }
            return
        }
        
        let fileList = fileListResult.files
        let commitHash = fileListResult.commitHash
        
        // If we got 0 files, fall back to Python download
        guard !fileList.isEmpty else {
            await MainActor.run {
                job.isDownloading = false
                job.downloadProgress = 0.0
                job.downloadStatus = nil
                job.output += "[WARNING] Could not fetch file list from API (got 0 files).\n"
                job.output += "[INFO] Training will download model automatically using Python...\n\n"
            }
            return
        }
        
        // Create snapshot directory with commit hash
        let snapshotDir = snapshotsDir.appendingPathComponent(commitHash)
        try? FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)
        
        await MainActor.run {
            job.downloadStatus = "Found \(fileList.count) files to download"
            job.output += "[INFO] Found \(fileList.count) files to download\n"
            job.output += "[INFO] Commit: \(commitHash)\n"
            job.output += "[INFO] Starting download...\n\n"
        }
        
        // Calculate total size first
        let totalBytes: Int64 = fileList.reduce(0) { $0 + Int64($1.size) }
        
        // Download files in parallel with limited concurrency
        // Increased to 15 for better throughput - URLSessionDownloadTask is more efficient
        // Note: Too high may cause server throttling, adjust based on testing
        let maxConcurrentDownloads = 15
        
        // Create progress tracker instance
        let progressTracker = ProgressTracker()
        
        await MainActor.run {
            job.output += "[INFO] Downloading \(fileList.count) files in parallel (max \(maxConcurrentDownloads) concurrent)...\n\n"
        }
        
        // Use TaskGroup for parallel downloads with concurrency limit
        await withTaskGroup(of: (index: Int, success: Bool, bytesDownloaded: Int64, fileName: String).self) { group in
            // Track active downloads to limit concurrency
            var activeTasks = 0
            var fileIndex = 0
            
            // Start initial batch of downloads
            while fileIndex < fileList.count && activeTasks < maxConcurrentDownloads {
                let fileInfo = fileList[fileIndex]
                let currentIndex = fileIndex
                fileIndex += 1
                activeTasks += 1
                
                group.addTask {
                    let filePath = fileInfo.path
                    let fileName = (filePath as NSString).lastPathComponent
                    
                    // Download file
                    let (success, bytesDownloaded) = await self.downloadFile(
                        modelId: modelId,
                        filePath: filePath,
                        destination: snapshotDir.appendingPathComponent(filePath),
                        hfToken: hfToken,
                        job: job,
                        totalFiles: fileList.count,
                        currentFile: currentIndex + 1,
                        totalBytes: totalBytes,
                        downloadedBytesSoFar: 0, // Will be updated atomically
                        fileList: fileList,
                        progressTracker: progressTracker
                    )
                    
                    return (index: currentIndex, success: success, bytesDownloaded: bytesDownloaded, fileName: fileName)
                }
            }
            
            // Process completed downloads and start new ones
            while fileIndex < fileList.count || activeTasks > 0 {
                // Check if job was cancelled
                let isCancelled = await MainActor.run {
                    return job.status == .cancelled
                }
                if isCancelled {
                    // Cancel all remaining tasks in the group
                    group.cancelAll()
                    break
                }
                
                if let result = await group.next() {
                    activeTasks -= 1
                    
                    // Update progress atomically using actor
                    await progressTracker.addDownload(success: result.success, bytes: result.bytesDownloaded, fileIndex: result.index)
                    let (currentCount, currentBytes) = await progressTracker.getProgress()
                    
                    // Update UI - always update progress bar, not just status
                    if result.success {
                        // Recalculate total bytes if we got actual file size (might be different from API)
                        let actualTotalBytes = max(totalBytes, currentBytes) // Use larger of API total or actual downloaded
                        let percent = actualTotalBytes > 0 ? Int((Double(currentBytes) / Double(actualTotalBytes)) * 100) : 0
                        let progressValue = actualTotalBytes > 0 ? Double(currentBytes) / Double(actualTotalBytes) : 0.0
                        await MainActor.run {
                            // Always update progress bar when a file completes
                            job.downloadProgress = min(progressValue, 0.99) // Cap at 99% until all files complete
                            job.downloadStatus = "Downloading: \(currentCount)/\(fileList.count) files (\(formatBytes(currentBytes))/\(formatBytes(actualTotalBytes)))"
                            
                            // Add completion message
                            job.output += "  ✓ [\(result.index + 1)/\(fileList.count)] \(result.fileName) - \(formatBytes(result.bytesDownloaded))\n"
                            
                            // Update progress more frequently for better feedback
                            // Show progress every file, or every 2 files for large batches
                            let updateFrequency = fileList.count > 20 ? 2 : 1
                            if currentCount % updateFrequency == 0 || currentCount == fileList.count {
                                job.output += "[PROGRESS] Overall: \(percent)% (\(currentCount)/\(fileList.count) files, \(formatBytes(currentBytes))/\(formatBytes(actualTotalBytes)))\n"
                            }
                        }
                    } else {
                        await MainActor.run {
                            job.output += "  ✗ [\(result.index + 1)/\(fileList.count)] \(result.fileName) - Failed\n"
                        }
                    }
                    
                    // Start next download if available (and not cancelled)
                    if fileIndex < fileList.count {
                        // Check again if cancelled before starting new download
                        let isCancelled = await MainActor.run {
                            return job.status == .cancelled
                        }
                        if isCancelled {
                            group.cancelAll()
                            break
                        }
                        
                        let fileInfo = fileList[fileIndex]
                        let currentIndex = fileIndex
                        fileIndex += 1
                        activeTasks += 1
                        
                        group.addTask {
                            let filePath = fileInfo.path
                            let fileName = (filePath as NSString).lastPathComponent
                            
                            let (success, bytesDownloaded) = await self.downloadFile(
                                modelId: modelId,
                                filePath: filePath,
                                destination: snapshotDir.appendingPathComponent(filePath),
                                hfToken: hfToken,
                                job: job,
                                totalFiles: fileList.count,
                                currentFile: currentIndex + 1,
                                totalBytes: totalBytes,
                                downloadedBytesSoFar: 0,
                                fileList: fileList,
                                progressTracker: progressTracker
                            )
                            
                            return (index: currentIndex, success: success, bytesDownloaded: bytesDownloaded, fileName: fileName)
                        }
                    }
                }
            }
        }
        
        // Check if job was cancelled before finalizing
        let wasCancelled = await MainActor.run {
            return job.status == .cancelled
        }
        
        if wasCancelled {
            await MainActor.run {
                job.isDownloading = false
                job.downloadStatus = "Download cancelled"
                job.output += "\n[INFO] Download cancelled by user\n"
            }
            return
        }
        
        // Get final progress for summary
        let (finalCount, finalBytes) = await progressTracker.getProgress()
        
        // Create refs/heads/main file pointing to the snapshot
        let refsDir = modelCacheDir.appendingPathComponent("refs/heads")
        try? FileManager.default.createDirectory(at: refsDir, withIntermediateDirectories: true)
        let mainRef = refsDir.appendingPathComponent("main")
        try? commitHash.write(to: mainRef, atomically: true, encoding: String.Encoding.utf8)
        
        await MainActor.run {
            job.isDownloading = false
            job.downloadProgress = 1.0
            job.downloadStatus = "Download complete!"
            job.output += "\n[SUCCESS] Model downloaded successfully!\n"
            job.output += "[INFO] Downloaded \(finalCount)/\(fileList.count) files (\(formatBytes(finalBytes))/\(formatBytes(totalBytes)))\n"
            job.output += "[INFO] Model cached at: \(modelCacheDir.path)\n"
            job.output += "[INFO] Starting training...\n\n"
        }
    }
    
    // Model file information used for Hugging Face model downloads
    private struct ModelFileInfo {
        let path: String
        let size: Int64
    }
    
    /// Gets the list of files for a model from Hugging Face API
    /// Tries siblings field first (most reliable), then falls back to tree endpoint
    private func getModelFileList(modelId: String, hfToken: String?) async -> (files: [ModelFileInfo], commitHash: String)? {
        // Try siblings first (most reliable and complete)
        if let result = await getModelFileListFromSiblings(modelId: modelId, hfToken: hfToken, commitHash: nil) {
            if !result.files.isEmpty {
                return result
            }
        }
        
        // If siblings failed or returned empty, try tree endpoint
        return await getModelFileListFromTree(modelId: modelId, hfToken: hfToken)
    }
    
    /// Gets file list from tree endpoint (recursive)
    private func getModelFileListFromTree(modelId: String, hfToken: String?) async -> (files: [ModelFileInfo], commitHash: String)? {
        // First get model info to get commit hash
        let modelInfoUrlString = "https://huggingface.co/api/models/\(modelId)"
        guard let modelInfoUrl = URL(string: modelInfoUrlString) else { return nil }
        
        var modelInfoRequest = URLRequest(url: modelInfoUrl)
        if let token = hfToken, !token.isEmpty {
            modelInfoRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        var commitHash = "main"
        
        do {
            let (data, _) = try await URLSession.shared.data(for: modelInfoRequest)
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                commitHash = (json["sha"] as? String) ?? (json["default_branch"] as? String) ?? "main"
            }
        } catch {
            // Continue with default "main"
        }
        
        // Recursively fetch all files from tree
        var allFiles: [ModelFileInfo] = []
        
        func fetchTree(path: String = "") async {
            let treeUrlString = "https://huggingface.co/api/models/\(modelId)/tree/\(commitHash)\(path.isEmpty ? "" : "/\(path)")"
            guard let treeUrl = URL(string: treeUrlString) else { return }
            
            var treeRequest = URLRequest(url: treeUrl)
            if let token = hfToken, !token.isEmpty {
                treeRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }
            
            do {
                let (data, response) = try await URLSession.shared.data(for: treeRequest)
                guard let httpResponse = response as? HTTPURLResponse else {
                    return
                }
                
                // Check for authentication errors
                if httpResponse.statusCode == 401 || httpResponse.statusCode == 403 {
                    return // Authentication failed or forbidden
                }
                
                guard httpResponse.statusCode == 200,
                      let treeArray = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                    return
                }
                
                for item in treeArray {
                    guard let itemPath = item["path"] as? String else { continue }
                    let fullPath = path.isEmpty ? itemPath : "\(path)/\(itemPath)"
                    
                    if item["type"] as? String == "directory" {
                        // Recursively fetch directory contents
                        await fetchTree(path: fullPath)
                    } else if let size = item["size"] as? Int64 {
                        // This is a file
                        allFiles.append(ModelFileInfo(path: fullPath, size: size))
                    }
                }
            } catch {
                // Ignore errors and continue
            }
        }
        
        await fetchTree()
        
        return allFiles.isEmpty ? nil : (files: allFiles, commitHash: commitHash)
    }
    
    /// Gets file list from siblings field in model info (most reliable method)
    private func getModelFileListFromSiblings(modelId: String, hfToken: String?, commitHash: String?) async -> (files: [ModelFileInfo], commitHash: String)? {
        let urlString = "https://huggingface.co/api/models/\(modelId)"
        guard let url = URL(string: urlString) else { return nil }
        
        var request = URLRequest(url: url)
        if let token = hfToken, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            
            // Check for authentication errors
            if let httpResponse = response as? HTTPURLResponse {
                if httpResponse.statusCode == 401 {
                    // Authentication failed - token might be invalid
                    return nil
                } else if httpResponse.statusCode == 403 {
                    // Forbidden - might need authentication or model is gated
                    return nil
                }
            }
            
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            
            // Get commit hash
            let resolvedCommitHash = commitHash ?? (json["sha"] as? String) ?? (json["default_branch"] as? String) ?? "main"
            
            // Get siblings (file list)
            guard let siblings = json["siblings"] as? [[String: Any]] else {
                return nil
            }
            
            let files = siblings.compactMap { fileDict -> ModelFileInfo? in
                // Try both "rfilename" and "path" fields
                let path = (fileDict["rfilename"] as? String) ?? (fileDict["path"] as? String)
                guard let filePath = path else { return nil }
                
                // Size might be Int or Int64
                let size: Int64?
                if let sizeInt = fileDict["size"] as? Int {
                    size = Int64(sizeInt)
                } else if let sizeInt64 = fileDict["size"] as? Int64 {
                    size = sizeInt64
                } else {
                    size = nil
                }
                
                guard let fileSize = size else { return nil }
                return ModelFileInfo(path: filePath, size: fileSize)
            }
            
            return (files: files, commitHash: resolvedCommitHash)
        } catch {
            return nil
        }
    }
    
    /// Downloads a single file with progress tracking
    private func downloadFile(
        modelId: String,
        filePath: String,
        destination: URL,
        hfToken: String?,
        job: TrainingJob,
        totalFiles: Int,
        currentFile: Int,
        totalBytes: Int64,
        downloadedBytesSoFar: Int64,
        fileList: [ModelFileInfo],
        progressTracker: ProgressTracker?
    ) async -> (success: Bool, bytesDownloaded: Int64) {
        // Hugging Face CDN URL format
        let fileUrlString = "https://huggingface.co/\(modelId)/resolve/main/\(filePath)"
        guard let fileUrl = URL(string: fileUrlString) else {
            return (false, 0)
        }
        
        var request = URLRequest(url: fileUrl)
        if let token = hfToken, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        // Create destination directory if needed
        let destDir = destination.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
        
        let fileName = (filePath as NSString).lastPathComponent
        let fileSize = fileList.first(where: { $0.path == filePath })?.size ?? 0
        
        // For large files, use delegate-based download for progress tracking
        // For small files, use async/await for simplicity
        if fileSize > 10 * 1024 * 1024, let tracker = progressTracker { // Files > 10MB
            return await downloadFileWithProgress(
                request: request,
                destination: destination,
                job: job,
                fileName: fileName,
                currentFile: currentFile,
                totalFiles: totalFiles,
                totalBytes: totalBytes,
                fileSize: fileSize,
                fileIndex: currentFile - 1,
                progressTracker: tracker
            )
        } else {
            // Small files - use simple async download
            return await downloadFileSimple(request: request, destination: destination, job: job)
        }
    }
    
    /// Downloads a file with progress tracking using delegate (for large files)
    private func downloadFileWithProgress(
        request: URLRequest,
        destination: URL,
        job: TrainingJob,
        fileName: String,
        currentFile: Int,
        totalFiles: Int,
        totalBytes: Int64,
        fileSize: Int64,
        fileIndex: Int,
        progressTracker: ProgressTracker
    ) async -> (success: Bool, bytesDownloaded: Int64) {
        let delegate = DownloadDelegate(
            progressTracker: progressTracker,
            job: job,
            fileName: fileName,
            currentFile: currentFile,
            totalFiles: totalFiles,
            totalBytes: totalBytes,
            fileSize: fileSize,
            fileIndex: fileIndex,
            destination: destination
        )
        
        let session = createDownloadSession(delegate: delegate)
        
        // Store session reference so we can cancel it if job is stopped
        await MainActor.run {
            job.activeDownloadSessions.append(session)
        }
        
        // Clean up session when done
        defer {
            Task { @MainActor in
                if let index = job.activeDownloadSessions.firstIndex(where: { $0 === session }) {
                    job.activeDownloadSessions.remove(at: index)
                }
            }
            session.invalidateAndCancel()
        }
        
        // Start a periodic progress update task to ensure UI updates even if delegate is slow
        let progressUpdateTask = Task {
            var lastReportedBytes: Int64 = 0
            while !Task.isCancelled {
                // Check if job was cancelled
                await MainActor.run {
                    if job.status == .cancelled {
                        return
                    }
                }
                
                try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 seconds
                
                let (_, currentBytes) = await progressTracker.getProgress()
                // Only update if bytes have changed
                if currentBytes != lastReportedBytes {
                    lastReportedBytes = currentBytes
                    let overallProgress = totalBytes > 0 ? Double(currentBytes) / Double(totalBytes) : 0.0
                    let overallPercent = Int(overallProgress * 100)
                    
                    await MainActor.run {
                        // Check again if cancelled
                        if job.status == .cancelled {
                            return
                        }
                        job.downloadProgress = min(overallProgress, 0.99)
                        if let status = job.downloadStatus, !status.contains("Overall:") {
                            job.downloadStatus = "Downloading: \(currentFile)/\(totalFiles) files - Overall: \(overallPercent)%"
                        } else if job.downloadStatus == nil {
                            job.downloadStatus = "Downloading: \(currentFile)/\(totalFiles) files - Overall: \(overallPercent)%"
                        }
                    }
                }
            }
        }
        defer { progressUpdateTask.cancel() }
        
        // Check if job was cancelled before starting download
        let isCancelled = await MainActor.run {
            return job.status == .cancelled
        }
        if isCancelled {
            return (success: false, bytesDownloaded: 0)
        }
        
        return await withCheckedContinuation { continuation in
            // CRITICAL: Don't use completion handler - it prevents delegate methods from being called!
            // Use downloadTask(with:) without completion handler, handle completion in delegate
            delegate.continuation = continuation
            
            let task = session.downloadTask(with: request)
            
            // Store task reference so we can cancel it if job is stopped
            // Use Task to update MainActor-isolated property from synchronous closure
            Task { @MainActor in
                job.activeDownloadTasks.append(task)
            }
            
            // Remove task when it completes (handled in delegate)
            // We'll remove it in the delegate's completion methods
            
            // Check HTTP response before starting (we'll check again in delegate)
            // Note: We can't check response here without completion handler, so we'll handle errors in delegate
            
            // CRITICAL: Must call resume() or the task won't start!
            task.resume()
        }
    }
    
    /// Downloads a file without progress tracking (for small files)
    private func downloadFileSimple(request: URLRequest, destination: URL, job: TrainingJob) async -> (success: Bool, bytesDownloaded: Int64) {
        do {
            // Use URLSessionDownloadTask for better performance
            let (localUrl, response) = try await URLSession.shared.download(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                return (false, 0)
            }
            
            // Check for authentication errors
            if httpResponse.statusCode == 401 {
                await MainActor.run {
                    job.output += "  ✗ Authentication failed (401) - check HF_TOKEN\n"
                }
                return (false, 0)
            } else if httpResponse.statusCode == 403 {
                await MainActor.run {
                    job.output += "  ✗ Forbidden (403) - model may be gated or token invalid\n"
                }
                return (false, 0)
            } else if httpResponse.statusCode != 200 {
                await MainActor.run {
                    job.output += "  ✗ HTTP error: \(httpResponse.statusCode)\n"
                }
                return (false, 0)
            }
            
            // Move downloaded file to final destination
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: localUrl, to: destination)
            
            // Get actual file size from disk
            let fileAttributes = try? FileManager.default.attributesOfItem(atPath: destination.path)
            let fileSize = (fileAttributes?[.size] as? Int64) ?? 0
            
            return (true, fileSize)
        } catch {
            await MainActor.run {
                job.output += "  ✗ Error: \(error.localizedDescription)\n"
            }
            return (false, 0)
        }
    }
    
    /// Formats bytes to human-readable string
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        // Use all units for better precision (KB, MB, GB)
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        formatter.allowsNonnumericFormatting = false
        return formatter.string(fromByteCount: bytes)
    }
    
    // MARK: - Post-Training Quantization
    
    /// Runs post-training quantization on the trained model
    private func runPostTrainingQuantization(job: TrainingJob, quantization: QuantizationType, venvPython: URL, hfToken: String?) async -> Bool {
        // Determine the model path (base model + adapter if applicable)
        let baseModelPath = job.config.model
        let adapterPath = job.config.adapterPath
        
        // Determine output path for quantized model
        let quantizedModelPath: String
        if let customPath = job.config.quantizedModelPath, !customPath.isEmpty {
            quantizedModelPath = customPath
        } else if let adapterPath = adapterPath {
            // Default to adapterPath/quantized
            quantizedModelPath = URL(fileURLWithPath: adapterPath).appendingPathComponent("quantized").path
        } else {
            // Fallback: use a default location based on job name
            let jobName = job.name.replacingOccurrences(of: " ", with: "_")
            let defaultPath = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                .appendingPathComponent("MLX Training Studio")
                .appendingPathComponent("adapters")
                .appendingPathComponent(jobName)
                .appendingPathComponent("quantized")
            quantizedModelPath = defaultPath.path
        }
        
        let quantizationBits = quantization == .bits4 ? "4" : "8"
        
        // Determine the source model path (fused adapter or base model)
        let sourceModelPath: String
        if let adapterPath = adapterPath {
            sourceModelPath = adapterPath  // Fused model is in adapter path
        } else {
            sourceModelPath = baseModelPath
        }
        
        // Check if model is already quantized - if so, skip quantization entirely
        // Check 1: Base model name contains quantization indicator
        let baseModelLower = baseModelPath.lowercased()
        let isBaseNameQuantized = baseModelLower.contains("4bit") || baseModelLower.contains("8bit") || baseModelLower.contains("6bit")
        
        // Check 2: Source model config.json contains quantization info
        let configPath = URL(fileURLWithPath: sourceModelPath).appendingPathComponent("config.json")
        var isConfigQuantized = false
        if let configData = try? Data(contentsOf: configPath),
           let configJson = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] {
            // MLX quantized models have "quantization" key in config.json
            if configJson["quantization"] != nil {
                isConfigQuantized = true
            }
        }
        
        if isBaseNameQuantized || isConfigQuantized {
            await MainActor.run {
                job.output += "\n=== Post-Training Quantization ===\n"
                job.output += "[INFO] Model is already quantized.\n"
                job.output += "[INFO] Skipping quantization - re-quantizing would cause errors.\n"
                job.output += "[INFO] Fused model location: \(sourceModelPath)\n\n"
            }
            return true
        }
        
        await MainActor.run {
            job.output += "\n=== Post-Training Quantization ===\n"
            job.output += "[INFO] Quantizing model to \(quantization.displayName)...\n"
            job.output += "[INFO] Source model: \(sourceModelPath)\n"
            job.output += "[INFO] Output path: \(quantizedModelPath)\n"
            job.isDownloading = true
            job.downloadProgress = 0.0
            job.downloadStatus = "Quantizing model..."
        }
        
        // Clean up output directory if it exists (mlx_lm.convert doesn't allow overwriting)
        let outputURL = URL(fileURLWithPath: quantizedModelPath)
        if FileManager.default.fileExists(atPath: quantizedModelPath) {
            await MainActor.run {
                job.output += "[INFO] Output directory exists, removing it...\n"
            }
            try? FileManager.default.removeItem(at: outputURL)
        }
        
        // Create parent directory
        try? FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        
        // Build arguments for mlx_lm convert command
        let args = ["-m", "mlx_lm", "convert",
                    "--hf-path", sourceModelPath,
                    "--mlx-path", quantizedModelPath,
                    "--q-bits", quantizationBits]
        
        await MainActor.run {
            job.output += "[INFO] Running: python \(args.joined(separator: " "))\n"
            job.output += "[INFO] This may take several minutes...\n\n"
        }
        
        let shellEnv = await loadShellEnvironment()
        
        let quantizeProcess = Process()
        quantizeProcess.executableURL = venvPython
        quantizeProcess.arguments = args
        
        var quantizeEnv = ProcessInfo.processInfo.environment
        quantizeEnv["PYTHONUNBUFFERED"] = "1"
        if let token = hfToken, !token.isEmpty {
            quantizeEnv["HF_TOKEN"] = token
        }
        for (key, value) in shellEnv {
            if key == "HF_HOME" {
                quantizeEnv[key] = value
            }
        }
        quantizeProcess.environment = quantizeEnv
        
        let quantizeOutputPipe = Pipe()
        let quantizeErrorPipe = Pipe()
        quantizeProcess.standardOutput = quantizeOutputPipe
        quantizeProcess.standardError = quantizeErrorPipe
        
        quantizeOutputPipe.fileHandleForReading.readabilityHandler = { [weak job] handle in
            let data = handle.availableData
            if !data.isEmpty {
                if let text = String(data: data, encoding: .utf8), let job = job {
                    Task { @MainActor in
                        job.output += text
                    }
                }
            }
        }
        
        quantizeErrorPipe.fileHandleForReading.readabilityHandler = { [weak job] handle in
            let data = handle.availableData
            if !data.isEmpty {
                if let text = String(data: data, encoding: .utf8), let job = job {
                    Task { @MainActor in
                        job.output += text
                    }
                }
            }
        }
        
        defer {
            quantizeOutputPipe.fileHandleForReading.readabilityHandler = nil
            quantizeErrorPipe.fileHandleForReading.readabilityHandler = nil
        }
        
        do {
            try quantizeProcess.run()
        } catch {
            await MainActor.run {
                job.isDownloading = false
                job.downloadProgress = 0.0
                job.downloadStatus = nil
                job.output += "\n[WARNING] Could not start quantization process: \(error.localizedDescription)\n"
                job.output += "[INFO] Training completed successfully, but quantization failed.\n"
                job.output += "[INFO] You can manually quantize the model later if needed.\n\n"
            }
            return false
        }

        let exitCode: Int32 = await withCheckedContinuation { continuation in
            quantizeProcess.terminationHandler = { process in
                continuation.resume(returning: process.terminationStatus)
            }
        }

        // Read any remaining output after process completes
        let remainingOutput = quantizeOutputPipe.fileHandleForReading.readDataToEndOfFile()
        let remainingError = quantizeErrorPipe.fileHandleForReading.readDataToEndOfFile()
        
        if !remainingOutput.isEmpty {
            if let text = String(data: remainingOutput, encoding: .utf8) {
                Task { @MainActor in
                    job.output += text
                }
            }
        }
        
        if exitCode != 0 {
            Task { @MainActor in
                job.isDownloading = false
                job.downloadProgress = 0.0
                job.downloadStatus = nil
                if let errorText = String(data: remainingError, encoding: .utf8) {
                    job.output += "\n[WARNING] Quantization had issues:\n"
                    job.output += errorText
                }
                job.output += "\n[INFO] Training completed successfully, but quantization failed.\n"
                job.output += "[INFO] You can manually quantize the model later if needed.\n\n"
            }
            return false
        } else {
            Task { @MainActor in
                job.isDownloading = false
                job.downloadProgress = 0.0
                job.downloadStatus = nil
                job.output += "\n[INFO] Post-training quantization completed successfully!\n\n"
            }
            return true
        }
    }
    
    private func runGGUFConversion(job: TrainingJob, venvPython: URL, hfToken: String?, quantizationSucceeded: Bool) async {
        // Get llama.cpp path from UserDefaults
        let llamaCppPath = UserDefaults.standard.string(forKey: "llamaCppPath") ?? ""
        
        guard !llamaCppPath.isEmpty else {
            Task { @MainActor in
                job.output += "\n[WARNING] GGUF conversion skipped: llama.cpp repository path not set in Settings.\n\n"
            }
            return
        }
        
        // Verify conversion script exists
        let expandedPath = NSString(string: llamaCppPath).expandingTildeInPath
        let convertScriptPath = URL(fileURLWithPath: expandedPath).appendingPathComponent("convert_hf_to_gguf.py")
        
        guard FileManager.default.fileExists(atPath: convertScriptPath.path) else {
            Task { @MainActor in
                job.output += "\n[WARNING] GGUF conversion skipped: convert_hf_to_gguf.py not found at \(convertScriptPath.path).\n"
                job.output += "[INFO] Please verify the llama.cpp repository path in Settings.\n\n"
            }
            return
        }
        
        // Guard: GGUF conversion is only supported for non-quantized base models.
        // Many community 4bit/8bit models include additional quantization tensors that
        // the convert_hf_to_gguf.py script cannot handle (e.g., *.biases, *.scales).
        let baseModelIdLowercased = job.config.model.lowercased()
        let quantizedMarkers = ["4bit", "8bit", "6bit", "3bit", "awq", "gptq", "bnb", "kbit"]
        if quantizedMarkers.contains(where: { baseModelIdLowercased.contains($0) }) {
            Task { @MainActor in
                job.output += "\n[WARNING] GGUF conversion skipped: base model appears to be quantized (\(job.config.model)).\n"
                job.output += "[INFO] GGUF conversion currently supports only full-precision (e.g., BF16/FP16) base models.\n"
                job.output += "[INFO] Please train using a non-quantized base model and disable post-training quantization for GGUF export.\n\n"
            }
            return
        }
        
        // Determine input model path
        // IMPORTANT: We always use the fused Hugging Face-style model directory
        // (base model + adapters merged together). This is written to adapterPath
        // when fuse=true. We do NOT use any MLX-quantized output directory here.
        guard let adapterPath = job.config.adapterPath else {
            Task { @MainActor in
                job.output += "\n[WARNING] GGUF conversion skipped: No adapter path set.\n\n"
            }
            return
        }
        let inputModelPath = adapterPath
        
        // Verify that the input path contains the trained model (should have config.json and model files)
        guard FileManager.default.fileExists(atPath: URL(fileURLWithPath: inputModelPath).appendingPathComponent("config.json").path) else {
            Task { @MainActor in
                job.output += "\n[WARNING] GGUF conversion skipped: config.json not found at \(inputModelPath).\n"
                job.output += "[INFO] The trained model may not be available at this path.\n\n"
            }
            return
        }
        
        // Build a descriptive default file name based on the base model id
        let baseModelId = job.config.model
        let rawModelName = baseModelId.components(separatedBy: "/").last ?? baseModelId
        let cleanedModelName = rawModelName
            .replacingOccurrences(of: "-4bit", with: "")
            .replacingOccurrences(of: "-8bit", with: "")
            .replacingOccurrences(of: "-6bit", with: "")
            .replacingOccurrences(of: "-3bit", with: "")
        let defaultStemBase = cleanedModelName.isEmpty ? "model-trained" : "\(cleanedModelName)-mlx-trained"
        let safeDefaultStem = defaultStemBase
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "/", with: "_")
        
        // Determine output path (must be a file path ending in .gguf, not a directory)
        let outputPath: String
        if let ggufPath = job.config.ggufOutputPath {
            // If user provided a path, use it (add .gguf extension if not present)
            if ggufPath.hasSuffix(".gguf") {
                outputPath = ggufPath
            } else if FileManager.default.fileExists(atPath: ggufPath) && (try? FileManager.default.attributesOfItem(atPath: ggufPath)[.type] as? FileAttributeType) == .typeDirectory {
                // If it's a directory, create a descriptive filename inside it
                outputPath = URL(fileURLWithPath: ggufPath).appendingPathComponent("\(safeDefaultStem).gguf").path
            } else {
                // Assume it's meant to be a file path, add .gguf if missing
                outputPath = ggufPath.hasSuffix(".gguf") ? ggufPath : "\(ggufPath).gguf"
            }
        } else if let adapterPath = job.config.adapterPath {
            // Default: write alongside the fused adapters directory with a descriptive name
            outputPath = URL(fileURLWithPath: adapterPath).appendingPathComponent("\(safeDefaultStem).gguf").path
        } else {
            Task { @MainActor in
                job.output += "\n[WARNING] GGUF conversion skipped: Could not determine output path.\n\n"
            }
            return
        }
        
        // Ensure output directory exists (create parent directory for the .gguf file)
        do {
            let outputDir = URL(fileURLWithPath: outputPath).deletingLastPathComponent()
            try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            Task { @MainActor in
                job.output += "\n[WARNING] Could not create output directory: \(error.localizedDescription)\n\n"
            }
            return
        }
        
        // Try to determine the model type from config.json in the input directory
        // This helps the script identify the model architecture
        let configPath = URL(fileURLWithPath: inputModelPath).appendingPathComponent("config.json")
        var modelName: String? = nil
        
        // Fix config.json if it's missing the 'architectures' field (required by convert_hf_to_gguf.py)
        // The fused model might not have this field, so we need to add it based on model_type
        if let configData = try? Data(contentsOf: configPath),
           var config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] {
            if config["architectures"] == nil, let modelType = config["model_type"] as? String {
                // Map model_type to architecture class name
                let architectureMap: [String: String] = [
                    "qwen2": "Qwen2ForCausalLM",
                    "qwen2_5": "Qwen2ForCausalLM",
                    "phi3": "Phi3ForCausalLM",
                    "llama": "LlamaForCausalLM",
                    "mistral": "MistralForCausalLM",
                    "mixtral": "MixtralForCausalLM"
                ]
                
                if let arch = architectureMap[modelType.lowercased()] {
                    config["architectures"] = [arch]
                    // Write the updated config back
                    if let updatedData = try? JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys]) {
                        try? updatedData.write(to: configPath)
                        Task { @MainActor in
                            job.output += "[INFO] Added missing 'architectures' field to config.json for GGUF conversion.\n"
                        }
                    }
                }
            }
            
            // First try to get model_type (e.g., "qwen2", "phi3", "llama", etc.)
            if let modelType = config["model_type"] as? String {
                modelName = modelType
            }
            // Fallback to architectures if model_type not found
            else if let arch = config["architectures"] as? [String],
                    let firstArch = arch.first {
                // Extract model name from architecture (e.g., "Phi3ForCausalLM" -> "phi3")
                modelName = firstArch.lowercased()
                    .replacingOccurrences(of: "forcausallm", with: "")
                    .replacingOccurrences(of: "model", with: "")
            }
        }
        
        // Fallback: try to get model name from the original model identifier
        if modelName == nil || modelName?.isEmpty == true {
            let modelId = job.config.model
            // Extract model name from identifier (e.g., "mlx-community/Phi-3.5-mini-instruct" -> "phi-3.5-mini-instruct")
            if let lastComponent = modelId.components(separatedBy: "/").last {
                // Try to extract model type from name (e.g., "Qwen2.5-0.5B-Instruct" -> "qwen2")
                let lowercased = lastComponent.lowercased()
                if lowercased.contains("qwen") {
                    modelName = "qwen2"  // Qwen2.5 uses qwen2 model type
                } else if lowercased.contains("phi") {
                    modelName = "phi3"
                } else if lowercased.contains("llama") {
                    modelName = "llama"
                } else {
                    modelName = lastComponent.lowercased()
                }
            }
        }
        
        Task { @MainActor in
            job.isDownloading = true
            job.downloadProgress = 0.0
            job.downloadStatus = "Converting to GGUF format..."
            job.output += "\n=== GGUF Conversion ===\n"
            job.output += "[INFO] Input model: \(inputModelPath)\n"
            if let name = modelName {
                job.output += "[INFO] Model name: \(name)\n"
            }
            job.output += "[INFO] Output path: \(outputPath)\n"
            job.output += "[INFO] This may take several minutes...\n\n"
        }
        
        // Load shell environment for HF_HOME
        let shellEnv = await loadShellEnvironment()
        
        // Run conversion using subprocess
        // convert_hf_to_gguf.py takes the model path as a positional argument
        // and --outfile for the output file path (must be a .gguf file, not a directory)
        // Optionally use --model-name and --outtype if we can determine them
        var convertArgs = [convertScriptPath.path, inputModelPath, "--outfile", outputPath]
        if let name = modelName, !name.isEmpty {
            convertArgs.append("--model-name")
            convertArgs.append(name)
        }
        // Always pass the desired GGUF outtype (default is .auto)
        convertArgs.append("--outtype")
        convertArgs.append(job.config.ggufOutType.rawValue)
        
        let convertProcess = Process()
        convertProcess.executableURL = venvPython
        convertProcess.arguments = convertArgs
        
        var convertEnv = ProcessInfo.processInfo.environment
        convertEnv["PYTHONUNBUFFERED"] = "1"
        if let token = hfToken, !token.isEmpty {
            convertEnv["HF_TOKEN"] = token
        }
        // Load HF_HOME from shell environment
        for (key, value) in shellEnv {
            if key == "HF_HOME" {
                convertEnv[key] = value
            }
        }
        convertProcess.environment = convertEnv
        
        let convertOutputPipe = Pipe()
        let convertErrorPipe = Pipe()
        convertProcess.standardOutput = convertOutputPipe
        convertProcess.standardError = convertErrorPipe
        
        // Set up handlers BEFORE starting the process
        convertOutputPipe.fileHandleForReading.readabilityHandler = { [weak job] handle in
            let data = handle.availableData
            if !data.isEmpty {
                if let text = String(data: data, encoding: .utf8), let job = job {
                    Task { @MainActor in
                        job.output += text
                    }
                }
            }
        }
        
        convertErrorPipe.fileHandleForReading.readabilityHandler = { [weak job] handle in
            let data = handle.availableData
            if !data.isEmpty {
                if let text = String(data: data, encoding: .utf8), let job = job {
                    Task { @MainActor in
                        job.output += text
                    }
                }
            }
        }
        
        do {
            // Start the process - handlers are already set up
            try convertProcess.run()
        } catch {
            Task { @MainActor in
                job.isDownloading = false
                job.downloadProgress = 0.0
                job.downloadStatus = nil
                job.output += "\n[WARNING] Could not start GGUF conversion process: \(error.localizedDescription)\n"
                job.output += "[INFO] GGUF conversion failed. You can manually convert the model later if needed.\n\n"
            }
            return
        }

        // Wait asynchronously for the process to exit without blocking the main thread
        let convertExitCode: Int32 = await withCheckedContinuation { continuation in
            convertProcess.terminationHandler = { process in
                continuation.resume(returning: process.terminationStatus)
            }
        }

        // Read any remaining output after process completes
        let remainingOutput = convertOutputPipe.fileHandleForReading.readDataToEndOfFile()
        let remainingError = convertErrorPipe.fileHandleForReading.readDataToEndOfFile()
        
        if !remainingOutput.isEmpty {
            if let text = String(data: remainingOutput, encoding: .utf8) {
                Task { @MainActor in
                    job.output += text
                }
            }
        }
        
        if convertExitCode != 0 {
            Task { @MainActor in
                job.isDownloading = false
                job.downloadProgress = 0.0
                job.downloadStatus = nil
                if let errorText = String(data: remainingError, encoding: .utf8) {
                    job.output += "\n[WARNING] GGUF conversion had issues:\n"
                    job.output += errorText
                }
                job.output += "\n[INFO] GGUF conversion failed. You can manually convert the model later if needed.\n\n"
            }
        } else {
            Task { @MainActor in
                job.isDownloading = false
                job.downloadProgress = 0.0
                job.downloadStatus = nil
                job.output += "\n[SUCCESS] GGUF conversion completed successfully!\n"
                job.output += "[INFO] GGUF model saved at: \(outputPath)\n\n"
            }
        }
        
        // Clean up handlers
        convertOutputPipe.fileHandleForReading.readabilityHandler = nil
        convertErrorPipe.fileHandleForReading.readabilityHandler = nil
    }
}



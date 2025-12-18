//
//  ContentView.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import SwiftUI
import Combine
import UniformTypeIdentifiers
import AppKit

// MARK: - Root / Launch Gate

struct ContentView: View {
    @EnvironmentObject var env: EnvironmentManager

    var body: some View {
        Group {
            switch env.healthStatus {
            case .unknown:
                VStack(spacing: 10) {
                    ProgressView()
                        .frame(width: 20, height: 20)
                    Text("Checking environment…")
                        .foregroundStyle(.secondary)
                        .font(.footnote)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            case .needsSetup(let message):
                SetupRequiredView(message: message) {
                    openSettingsWindow()
                } retryAction: {
                    Task { await env.quickHealthCheck() }
                }

            case .ready:
                JobsListView()
                    .environmentObject(TrainingJobManager.shared)
            }
        }
        .task {
            // Run once at launch
            await env.quickHealthCheck()
        }
    }
    
    private func openSettingsWindow() {
        // Open the native macOS Settings window
        // SwiftUI Settings scenes are automatically connected to the App menu → Settings...
        // Find and trigger the Settings menu item
        guard let mainMenu = NSApp.mainMenu else {
            fallbackOpenSettings()
            return
        }
        
        // Get app name from bundle
        let appName = Bundle.main.localizedInfoDictionary?["CFBundleName"] as? String
            ?? Bundle.main.infoDictionary?["CFBundleName"] as? String
            ?? "App"
        
        // Find the app menu (usually first item or one matching app name)
        var appMenu: NSMenuItem?
        for menuItem in mainMenu.items {
            let title = menuItem.title
            if title == appName || title == "App" || menuItem == mainMenu.items.first {
                appMenu = menuItem
                break
            }
        }
        
        guard let foundAppMenu = appMenu,
              let submenu = foundAppMenu.submenu else {
            fallbackOpenSettings()
            return
        }
        
        // Find Settings/Preferences menu item
        var settingsItem: NSMenuItem?
        for menuItem in submenu.items {
            let title = menuItem.title
            if title.contains("Settings") || title.contains("Preferences") {
                settingsItem = menuItem
                break
            }
        }
        
        if let foundSettingsItem = settingsItem,
           let action = foundSettingsItem.action {
            NSApp.sendAction(action, to: foundSettingsItem.target, from: nil)
            return
        }
        
        fallbackOpenSettings()
    }
    
    private func fallbackOpenSettings() {
        // Fallback: activate app and try standard action
        NSApp.activate(ignoringOtherApps: true)
        if #available(macOS 13, *) {
            NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        } else {
            NSApp.sendAction(Selector(("showPreferencesWindow:")), to: nil, from: nil)
        }
    }
}

// MARK: - Setup Required

private struct SetupRequiredView: View {
    let message: String
    let openSettingsAction: () -> Void
    let retryAction: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Setup Required")
                .font(.title2)
                .bold()

            Text(message)
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                Button("Open Environment Settings…") { openSettingsAction() }
                Button("Check Again") { retryAction() }
                Spacer()
            }

            Spacer()
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

#Preview {
    ContentView()
        .environmentObject(EnvironmentManager())
}

// MARK: - Environment Manager

@MainActor
final class EnvironmentManager: ObservableObject, @unchecked Sendable {
    // Persisted user settings
    @Published var pythonPath: String {
        didSet { UserDefaults.standard.set(pythonPath, forKey: Self.pythonPathKey) }
    }
    
    @Published var hfToken: String {
        didSet { UserDefaults.standard.set(hfToken, forKey: Self.hfTokenKey) }
    }
    
    @Published var llamaCppPath: String {
        didSet { UserDefaults.standard.set(llamaCppPath, forKey: Self.llamaCppPathKey) }
    }

    // Status
    @Published var pythonStatus: String = ""
    @Published var pythonStatusIsError: Bool = false

    @Published var envStatus: String = ""
    @Published var envStatusIsError: Bool = false
    
    @Published var llamaCppStatus: String = ""
    @Published var llamaCppStatusIsError: Bool = false

    @Published var log: String = ""
    @Published var isBusy: Bool = false

    @Published var healthStatus: HealthStatus = .unknown

    // Paths
    let appSupportDir: URL
    let venvDir: URL

    var venvExists: Bool {
        FileManager.default.fileExists(atPath: venvPython.path)
    }

    var venvPython: URL {
        venvDir.appendingPathComponent("bin/python")
    }

    var venvPipArgs: [String] {
        ["-m", "pip"]
    }

    private static let pythonPathKey = "pythonPath"
    private static let hfTokenKey = "hfToken"
    private static let llamaCppPathKey = "llamaCppPath"

    init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        self.appSupportDir = base.appendingPathComponent("MLX Training Studio", isDirectory: true)
        self.venvDir = appSupportDir.appendingPathComponent("venv", isDirectory: true)

        self.pythonPath = UserDefaults.standard.string(forKey: Self.pythonPathKey) ?? ""
        self.hfToken = UserDefaults.standard.string(forKey: Self.hfTokenKey) ?? ""
        self.llamaCppPath = UserDefaults.standard.string(forKey: Self.llamaCppPathKey) ?? ""

        do {
            try FileManager.default.createDirectory(at: appSupportDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            // surfaced later via logs
        }
    }

    // MARK: - Launch / Health

    func quickHealthCheck() async {
        // Avoid overlapping checks
        if isBusy { return }

        // Minimal, fast check; does NOT install anything.
        if pythonPath.isEmpty {
            // Try auto-detection to help first-time users
            await detectPython()
        }

        guard !pythonPath.isEmpty else {
            healthStatus = .needsSetup("Select a Python executable (Homebrew/pyenv/conda). Then create a managed venv and install mlx-lm-lora.")
            return
        }

        // Validate python runs
        let res = await runCapture(executable: pythonPath, args: ["-c", "import sys; print(sys.executable)"])
        guard res.exitCode == 0 else {
            healthStatus = .needsSetup("Python is set but failed to run. Open Environment Settings and choose a working Python.")
            return
        }

        // Validate venv + imports (if venv exists)
        if venvExists {
            let imp = await runCapture(executable: venvPython.path, args: ["-c", "import mlx; import mlx_lm_lora; print('ok')"])
            if imp.exitCode == 0 {
                healthStatus = .ready
            } else {
                healthStatus = .needsSetup("Managed venv exists, but mlx-lm-lora (or its deps) are not importable. Open Environment Settings and run Install/Update.")
            }
        } else {
            healthStatus = .needsSetup("Managed venv is missing. Open Environment Settings and click Create venv.")
        }
    }

    // MARK: - Actions

    func detectPython() async {
        isBusy = true
        defer { isBusy = false }

        appendLog("\n== Detecting python on PATH ==\n")

        // Prefer real Python installs over Apple's /usr/bin/python3 shim (xcrun)
        let preferredCandidates = [
            "/opt/homebrew/bin/python3",                // Homebrew (Apple Silicon default)
            "/usr/local/bin/python3",                  // Homebrew (Intel default)
            "~/.pyenv/shims/python3",                  // pyenv (python3)
            "~/.pyenv/shims/python",                   // pyenv (python)
            "~/miniconda3/bin/python3",                // conda
            "~/anaconda3/bin/python3"                  // conda
        ].map { NSString(string: $0).expandingTildeInPath }

        // 1) Try preferred known locations
        if let first = preferredCandidates.first(where: { FileManager.default.fileExists(atPath: $0) }) {
            pythonPath = first
        }

        // 2) Try `which python3` then `which python`
        if pythonPath.isEmpty {
            for name in ["python3", "python"] {
                let which = await runCapture(executable: "/usr/bin/which", args: [name], env: nil)
                if which.exitCode == 0 {
                    let path = which.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !path.isEmpty {
                        pythonPath = path
                        break
                    }
                }
            }
        }

        // 3) Last resort: /usr/bin/python3 (often an xcrun shim)
        if pythonPath.isEmpty, FileManager.default.fileExists(atPath: "/usr/bin/python3") {
            pythonPath = "/usr/bin/python3"
        }

        if pythonPath.isEmpty {
            pythonStatus = "Could not auto-detect python. Use Browse… to select it."
            pythonStatusIsError = true
            appendLog("[detect] Failed to locate python.\n")
            return
        }

        await validateCurrentPython()
    }

    func setPythonFromPickedURL(_ url: URL) async {
        isBusy = true
        defer { isBusy = false }

        appendLog("\n== Selected python via picker ==\n")
        appendLog("[picker] \(url.path)\n")

        pythonPath = url.path
        await validateCurrentPython()
    }

    private func validateCurrentPython() async {
        guard !pythonPath.isEmpty else { return }

        appendLog("[detect] Using python: \(pythonPath)\n")
        let check = await runCapture(
            executable: pythonPath,
            args: ["-c", "import sys; print(sys.executable); print(sys.version)"]
        )

        if check.exitCode == 0 {
            pythonStatus = "Python OK: \(check.stdout.split(separator: "\n").first.map(String.init) ?? pythonPath)"
            pythonStatusIsError = false
            appendLog(check.stdout)
        } else {
            let errText = check.stderr.isEmpty ? check.stdout : check.stderr
            if errText.contains("xcrun: error") {
                pythonStatus = "The selected python appears to be an Apple xcrun shim (/usr/bin/python3). Use Homebrew/pyenv/conda Python instead."
            } else {
                pythonStatus = "Python found, but failed to run. See log output."
            }
            pythonStatusIsError = true
            appendLog(errText)

            if pythonPath == "/usr/bin/python3" {
                pythonPath = ""
            }
        }

        // Keep launch gate status up to date
        await quickHealthCheck()
    }

    func createVenv() async {
        isBusy = true
        defer { isBusy = false }

        guard !pythonPath.isEmpty else {
            envStatus = "Set a Python path first."
            envStatusIsError = true
            return
        }

        if pythonPath == "/usr/bin/python3" {
            envStatus = "Cannot use /usr/bin/python3 (xcrun shim). Choose Homebrew/pyenv/conda Python."
            envStatusIsError = true
            appendLog("[venv] Refusing /usr/bin/python3 (xcrun shim).\n")
            return
        }

        appendLog("\n== Creating venv ==\n")
        appendLog("[venv] Target: \(venvDir.path)\n")

        do {
            try FileManager.default.createDirectory(at: appSupportDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            envStatus = "Failed to create Application Support directory."
            envStatusIsError = true
            appendLog("[venv] Directory error: \(error)\n")
            return
        }

        if venvExists {
            envStatus = "Venv already exists."
            envStatusIsError = false
            appendLog("[venv] Already exists at \(venvPython.path)\n")
            await quickHealthCheck()
            return
        }

        let res = await runStreaming(
            executable: pythonPath,
            args: ["-m", "venv", venvDir.path]
        )

        if res == 0 {
            envStatus = "Venv created."
            envStatusIsError = false
            appendLog("[venv] OK\n")

            appendLog("\n== Upgrading pip ==\n")
            _ = await runStreaming(
                executable: venvPython.path,
                args: venvPipArgs + ["install", "-U", "pip", "setuptools", "wheel"]
            )
        } else {
            envStatus = "Venv creation failed. See log."
            envStatusIsError = true
        }

        await quickHealthCheck()
    }

    func installMlxLmLora() async {
        isBusy = true
        defer { isBusy = false }

        guard venvExists else {
            envStatus = "Create the venv first."
            envStatusIsError = true
            return
        }

        appendLog("\n== Installing mlx-lm-lora ==\n")
        appendLog("[pip] Using: \(venvPython.path)\n")

        let code = await runStreaming(
            executable: venvPython.path,
            args: venvPipArgs + ["install", "-U", "mlx-lm-lora"]
        )

        if code == 0 {
            envStatus = "Installed mlx-lm-lora."
            envStatusIsError = false
        } else {
            envStatus = "Install failed. See log output."
            envStatusIsError = true
        }

        await quickHealthCheck()
    }

    func runTest() async {
        isBusy = true
        defer { isBusy = false }

        guard venvExists else {
            envStatus = "Create the venv first."
            envStatusIsError = true
            return
        }

        appendLog("\n== Running smoke test ==\n")

        let importCode = await runStreaming(
            executable: venvPython.path,
            args: ["-c", "import mlx; import mlx_lm_lora; print('OK: mlx', getattr(mlx, '__version__', 'unknown')); print('OK: mlx_lm_lora', getattr(mlx_lm_lora, '__version__', 'unknown'))"]
        )

        guard importCode == 0 else {
            envStatus = "Import test failed. See log."
            envStatusIsError = true
            await quickHealthCheck()
            return
        }

        appendLog("\n== Running: mlx_lm_lora.train --help ==\n")
        let helpCode = await runStreaming(
            executable: venvPython.path,
            args: ["-m", "mlx_lm_lora.train", "--help"]
        )

        if helpCode == 0 {
            envStatus = "Smoke test passed."
            envStatusIsError = false
        } else {
            envStatus = "CLI test failed. See log."
            envStatusIsError = true
        }

        await quickHealthCheck()
    }
    
    // MARK: - llama.cpp Support
    
    func validateLlamaCppPath() async {
        guard !llamaCppPath.isEmpty else {
            llamaCppStatus = ""
            llamaCppStatusIsError = false
            return
        }
        
        let expandedPath = NSString(string: llamaCppPath).expandingTildeInPath
        let convertScriptPath = URL(fileURLWithPath: expandedPath).appendingPathComponent("convert_hf_to_gguf.py")
        
        if !FileManager.default.fileExists(atPath: expandedPath) {
            llamaCppStatus = "Path does not exist"
            llamaCppStatusIsError = true
            return
        }
        
        if !FileManager.default.fileExists(atPath: convertScriptPath.path) {
            llamaCppStatus = "convert_hf_to_gguf.py not found in this directory"
            llamaCppStatusIsError = true
            return
        }
        
        // Path is valid, install dependencies if venv exists
        if venvExists {
            llamaCppStatus = "Valid - installing dependencies..."
            llamaCppStatusIsError = false
            await installGGUFDependencies()
        } else {
            llamaCppStatus = "Valid (install dependencies after creating venv)"
            llamaCppStatusIsError = false
        }
    }
    
    func setLlamaCppFromPickedURL(_ url: URL) async {
        isBusy = true
        defer { isBusy = false }
        
        appendLog("\n== Selected llama.cpp path via picker ==\n")
        appendLog("[picker] \(url.path)\n")
        
        llamaCppPath = url.path
        await validateLlamaCppPath()
    }
    
    func installGGUFDependencies() async {
        guard venvExists else {
            llamaCppStatus = "Create venv first"
            llamaCppStatusIsError = true
            return
        }
        
        guard !llamaCppPath.isEmpty else {
            return
        }
        
        isBusy = true
        defer { isBusy = false }
        
        appendLog("\n== Installing GGUF conversion dependencies ==\n")
        appendLog("[pip] Using: \(venvPython.path)\n")
        
        // Dependencies typically needed for convert_hf_to_gguf.py
        // These are usually already installed with mlx-lm-lora, but we'll ensure they're there
        // Note: PyTorch is required for convert_hf_to_gguf.py to load and convert models
        let dependencies = [
            "numpy",
            "sentencepiece",
            "protobuf",
            "safetensors",
            "transformers",
            "torch"
        ]
        
        let code = await runStreaming(
            executable: venvPython.path,
            args: venvPipArgs + ["install", "-U"] + dependencies
        )
        
        if code == 0 {
            llamaCppStatus = "Valid and dependencies installed"
            llamaCppStatusIsError = false
        } else {
            llamaCppStatus = "Dependencies installation failed. See log."
            llamaCppStatusIsError = true
        }
    }
    
    var convertHfToGgufScriptPath: URL? {
        guard !llamaCppPath.isEmpty else { return nil }
        let expandedPath = NSString(string: llamaCppPath).expandingTildeInPath
        let scriptPath = URL(fileURLWithPath: expandedPath).appendingPathComponent("convert_hf_to_gguf.py")
        return FileManager.default.fileExists(atPath: scriptPath.path) ? scriptPath : nil
    }

    // MARK: - Helpers

    private func appendLog(_ text: String) {
        let maxChars = 250_000
        if log.count + text.count > maxChars {
            let overflow = (log.count + text.count) - maxChars
            if overflow < log.count {
                log.removeFirst(overflow)
            } else {
                log = ""
            }
        }
        log += text
    }

    private func runCapture(executable: String, args: [String], env: [String: String]? = nil) async -> ProcessResult {
        await withCheckedContinuation { continuation in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: executable)
            proc.arguments = args

            if let env { proc.environment = env }

            let out = Pipe()
            let err = Pipe()
            proc.standardOutput = out
            proc.standardError = err

            proc.terminationHandler = { p in
                let stdoutData = out.fileHandleForReading.readDataToEndOfFile()
                let stderrData = err.fileHandleForReading.readDataToEndOfFile()

                let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
                let stderr = String(data: stderrData, encoding: .utf8) ?? ""

                continuation.resume(returning: ProcessResult(exitCode: p.terminationStatus, stdout: stdout, stderr: stderr))
            }

            do {
                try proc.run()
            } catch {
                continuation.resume(returning: ProcessResult(exitCode: 127, stdout: "", stderr: "Failed to run process: \(error)\n"))
            }
        }
    }

    private func runStreaming(executable: String, args: [String], env: [String: String]? = nil) async -> Int32 {
        await withCheckedContinuation { continuation in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: executable)
            proc.arguments = args

            if let env { proc.environment = env }

            let out = Pipe()
            let err = Pipe()
            proc.standardOutput = out
            proc.standardError = err

            // Create a sendable closure wrapper to avoid concurrency warnings
            final class LogAppender: @unchecked Sendable {
                weak var manager: EnvironmentManager?
                
                init(manager: EnvironmentManager) {
                    self.manager = manager
                }
                
                func append(_ text: String) {
                    Task { @MainActor in
                        self.manager?.appendLog(text)
                    }
                }
            }
            
            let logAppender = LogAppender(manager: self)

            // Capture logAppender in each closure to avoid concurrency warnings
            out.fileHandleForReading.readabilityHandler = { h in
                let data = h.availableData
                guard !data.isEmpty else { return }
                guard let text = String(data: data, encoding: .utf8) else { return }
                logAppender.append(text)
            }
            
            err.fileHandleForReading.readabilityHandler = { h in
                let data = h.availableData
                guard !data.isEmpty else { return }
                guard let text = String(data: data, encoding: .utf8) else { return }
                logAppender.append(text)
            }

            proc.terminationHandler = { p in
                out.fileHandleForReading.readabilityHandler = nil
                err.fileHandleForReading.readabilityHandler = nil

                logAppender.append("\n[exit] code \(p.terminationStatus)\n")
                continuation.resume(returning: p.terminationStatus)
            }

            do {
                logAppender.append("$ \(executable) \(args.joined(separator: " "))\n")
                try proc.run()
            } catch {
                logAppender.append("Failed to run process: \(error)\n")
                continuation.resume(returning: 127)
            }
        }
    }
}

enum HealthStatus: Equatable {
    case unknown
    case needsSetup(String)
    case ready
}

struct ProcessResult {
    let exitCode: Int32
    let stdout: String
    let stderr: String
}

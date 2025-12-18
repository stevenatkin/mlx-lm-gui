//
//  EnvironmentSettingsView.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import SwiftUI
import UniformTypeIdentifiers
import AppKit

struct EnvironmentSettingsView: View {
    @EnvironmentObject var env: EnvironmentManager
    @State private var showPythonPicker = false

    var body: some View {
        Form {
            Section("Python Interpreter") {
                HStack(alignment: .center, spacing: 8) {
                    Text("Python Path:")
                    TextField("", text: $env.pythonPath)
                        .textFieldStyle(.roundedBorder)
                    Button("Detect") {
                        Task { await env.detectPython() }
                    }
                    .buttonStyle(.bordered)
                    .disabled(env.isBusy)
                    Button("Browse…") {
                        let panel = NSOpenPanel()
                        panel.allowsMultipleSelection = false
                        panel.canChooseDirectories = false
                        panel.canChooseFiles = true
                        panel.allowedContentTypes = [.executable, .item]
                        
                        if panel.runModal() == .OK {
                            if let url = panel.url {
                                Task { await env.setPythonFromPickedURL(url) }
                            }
                        }
                    }
                    .buttonStyle(.bordered)
                    .disabled(env.isBusy)
                }
                
                if !env.pythonStatus.isEmpty {
                    HStack {
                        Image(systemName: env.pythonStatusIsError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                            .foregroundStyle(env.pythonStatusIsError ? .red : .green)
                        Text(env.pythonStatus)
                            .font(.caption)
                            .foregroundStyle(env.pythonStatusIsError ? .red : .secondary)
                    }
                } else {
                    Text("e.g., /opt/homebrew/bin/python3 or ~/.pyenv/shims/python3")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            
            Section("Hugging Face") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .center, spacing: 8) {
                        Text("HF Token:")
                        TextField("", text: $env.hfToken)
                            .textFieldStyle(.roundedBorder)
                    }
                    Text("Hugging Face access token for downloading models. Get your token at https://huggingface.co/settings/tokens")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    if !env.hfToken.isEmpty {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                            Text("Token is set")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            
            Section("GGUF Conversion (Optional)") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .center, spacing: 8) {
                        Text("llama.cpp Path:")
                        TextField("", text: $env.llamaCppPath)
                            .textFieldStyle(.roundedBorder)
                            .onChange(of: env.llamaCppPath) { _, _ in
                                Task { await env.validateLlamaCppPath() }
                            }
                        Button("Browse…") {
                            let panel = NSOpenPanel()
                            panel.allowsMultipleSelection = false
                            panel.canChooseDirectories = true
                            panel.canChooseFiles = false
                            
                            if panel.runModal() == .OK {
                                if let url = panel.url {
                                    Task { await env.setLlamaCppFromPickedURL(url) }
                                }
                            }
                        }
                        .buttonStyle(.bordered)
                        .disabled(env.isBusy)
                    }
                    
                    Text("Path to the llama.cpp repository (optional). If set, you can convert trained models to GGUF format. Clone from: https://github.com/ggerganov/llama.cpp")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    if !env.llamaCppStatus.isEmpty {
                        HStack {
                            Image(systemName: env.llamaCppStatusIsError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                                .foregroundStyle(env.llamaCppStatusIsError ? .red : .green)
                            Text(env.llamaCppStatus)
                                .font(.caption)
                                .foregroundStyle(env.llamaCppStatusIsError ? .red : .secondary)
                        }
                    }
                }
            }
            
            Section("Virtual Environment") {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Location:")
                        Text(env.venvDir.path)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    
                    HStack(spacing: 8) {
                        Button("Create venv") {
                            Task { await env.createVenv() }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(env.isBusy || env.pythonPath.isEmpty)
                        
                        Button("Install/Update mlx-lm-lora") {
                            Task { await env.installMlxLmLora() }
                        }
                        .buttonStyle(.bordered)
                        .disabled(env.isBusy || !env.venvExists)
                        
                        Button("Run Smoke Test") {
                            Task { await env.runTest() }
                        }
                        .buttonStyle(.bordered)
                        .disabled(env.isBusy || !env.venvExists)
                    }
                    
                    if !env.envStatus.isEmpty {
                        HStack {
                            Image(systemName: env.envStatusIsError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                                .foregroundStyle(env.envStatusIsError ? .red : .green)
                            Text(env.envStatus)
                                .font(.caption)
                                .foregroundStyle(env.envStatusIsError ? .red : .secondary)
                        }
                    }
                }
            }
            
            Section("Output Log") {
                VStack(alignment: .leading, spacing: 8) {
                    ScrollView {
                        Text(env.log.isEmpty ? "No output yet..." : env.log)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                            .padding(8)
                    }
                    .frame(height: 120)
                    .background(Color(NSColor.textBackgroundColor))
                    .cornerRadius(6)
                    
                    HStack {
                        HStack(spacing: 4) {
                            Circle()
                                .fill(env.isBusy ? Color.orange : Color.green)
                                .frame(width: 8, height: 8)
                            Text(env.isBusy ? "Running…" : "Idle")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        
                        Spacer()
                        
                        Button("Clear Log") {
                            env.log = ""
                        }
                        .buttonStyle(.borderless)
                        .disabled(env.isBusy)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding(20)
        .frame(minWidth: 700, minHeight: 600)
    }
}

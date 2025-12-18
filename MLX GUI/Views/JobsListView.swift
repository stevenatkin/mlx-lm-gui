//
//  JobsListView.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct JobsListView: View {
    @EnvironmentObject var jobManager: TrainingJobManager
    @EnvironmentObject var envManager: EnvironmentManager
    @Environment(\.openWindow) private var openWindow
    @State private var showNewTrainingWizard = false
    @State private var showLoadError = false
    @State private var loadErrorMessage = ""
    @State private var selectedJobId: UUID? = nil
    @State private var showFailureAlert = false
    @State private var failureAlertMessage = ""
    @State private var failureJobName = ""
    @State private var failedJobId: UUID? = nil
    
    var body: some View {
        mainContent
            .modifier(JobsListViewModifiers(
                showNewTrainingWizard: $showNewTrainingWizard,
                jobManager: jobManager,
                failureJobName: $failureJobName,
                failureAlertMessage: $failureAlertMessage,
                failedJobId: $failedJobId,
                showFailureAlert: $showFailureAlert,
                selectedJobId: $selectedJobId,
                showLoadError: $showLoadError,
                loadErrorMessage: loadErrorMessage,
                openWindow: openWindow
            ))
    }
    
    private var mainContent: some View {
        NavigationSplitView {
            sidebarView
        } detail: {
            detailView
        }
    }
    
    private var sidebarView: some View {
        List(selection: $selectedJobId) {
            Section("Training Jobs") {
                ForEach(jobManager.jobs) { job in
                    NavigationLink(value: job.id) {
                        JobRowView(job: job)
                    }
                    .tag(job.id)
                }
            }
        }
        .navigationTitle("MLX Training Studio")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button {
                        TrainingJobManager.shared.preloadedConfig = nil
                        showNewTrainingWizard = true
                    } label: {
                        Label("New Training", systemImage: "plus")
                    }
                    
                    Button {
                        loadConfigurationFromFile()
                    } label: {
                        Label("Load Configuration", systemImage: "folder")
                    }
                } label: {
                    Label("New Training", systemImage: "plus")
                }
            }
            
        }
    }
    
    private var detailView: some View {
        Group {
            if let selectedId = selectedJobId,
               let selectedJob = jobManager.jobs.first(where: { $0.id == selectedId }) {
                JobDetailView(job: selectedJob, selectedJobId: $selectedJobId)
            } else {
                emptyStateView
            }
        }
    }
    
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            if jobManager.jobs.isEmpty {
                Image(systemName: "tray")
                    .font(.system(size: 48))
                    .foregroundStyle(.secondary)
                Text("No Training Jobs")
                    .font(.title2)
                    .bold()
                Text("Create your first training job to get started")
                    .foregroundStyle(.secondary)
            } else {
                Text("Select a job from the sidebar")
                    .foregroundStyle(.secondary)
            }
            
            HStack(spacing: 12) {
                Button {
                    TrainingJobManager.shared.preloadedConfig = nil
                    showNewTrainingWizard = true
                } label: {
                    Label("New Training", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
                
                Button {
                    loadConfigurationFromFile()
                } label: {
                    Label("Load Configuration", systemImage: "folder")
                }
                .buttonStyle(.bordered)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
}

// MARK: - View Modifiers
struct JobsListViewModifiers: ViewModifier {
    @Binding var showNewTrainingWizard: Bool
    @ObservedObject var jobManager: TrainingJobManager
    @Binding var failureJobName: String
    @Binding var failureAlertMessage: String
    @Binding var failedJobId: UUID?
    @Binding var showFailureAlert: Bool
    @Binding var selectedJobId: UUID?
    @Binding var showLoadError: Bool
    var loadErrorMessage: String
    var openWindow: OpenWindowAction
    
    func body(content: Content) -> some View {
        content
            .onChange(of: showNewTrainingWizard) { _, newValue in
                if newValue {
                    openWindow(id: "training-wizard")
                    showNewTrainingWizard = false
                }
            }
            .onChange(of: jobManager.failedJob) { _, failedJob in
                if let failedJob = failedJob, let error = failedJob.error {
                    failureJobName = failedJob.name
                    failureAlertMessage = error
                    failedJobId = failedJob.id
                    showFailureAlert = true
                    // Clear the failed job reference after showing alert
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        jobManager.failedJob = nil
                    }
                }
            }
            .sheet(isPresented: $showFailureAlert) {
                FailureAlertSheet(
                    jobName: failureJobName,
                    errorMessage: failureAlertMessage,
                    onViewDetails: {
                        if let jobId = failedJobId {
                            selectedJobId = jobId
                        }
                        showFailureAlert = false
                        failedJobId = nil
                    },
                    onDismiss: {
                        showFailureAlert = false
                        failedJobId = nil
                    }
                )
            }
            .onChange(of: jobManager.jobs.count) { _, _ in
                // Clear selection if the selected job no longer exists
                if let selectedId = selectedJobId,
                   !jobManager.jobs.contains(where: { $0.id == selectedId }) {
                    selectedJobId = nil
                }
            }
            .alert("Load Configuration Error", isPresented: $showLoadError) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(loadErrorMessage)
            }
    }
}

// MARK: - Helper Methods
extension JobsListView {
    private func loadConfigurationFromFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.yaml, .text]
        panel.title = "Select Training Configuration YAML"
        
        if panel.runModal() == .OK {
            guard let url = panel.url else { return }
            
            do {
                let yamlContent = try String(contentsOf: url, encoding: .utf8)
                let config = try YAMLManager.shared.yamlToConfiguration(yamlContent)
                
                // Store the preloaded config
                TrainingJobManager.shared.preloadedConfig = config
                
                // Open the wizard
                showNewTrainingWizard = true
            } catch {
                loadErrorMessage = "Failed to load configuration: \(error.localizedDescription)"
                showLoadError = true
            }
        }
    }
}

struct JobRowView: View {
    @ObservedObject var job: TrainingJob
    
    var body: some View {
        HStack {
            Text(job.status.icon)
                .font(.title2)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(job.name)
                    .font(.headline)
                Text("\(job.config.trainMode.displayName) • \(job.status.displayName)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
            
            if job.status == .running {
                ProgressView()
                    .scaleEffect(0.7)
                    .frame(width: 16, height: 16)
            }
        }
        .padding(.vertical, 4)
    }
}

struct JobDetailView: View {
    @ObservedObject var job: TrainingJob
    @EnvironmentObject var jobManager: TrainingJobManager
    @EnvironmentObject var envManager: EnvironmentManager
    @Binding var selectedJobId: UUID?
    @Environment(\.openWindow) private var openWindow
    
    // Check if this job still exists in the manager
    private var jobExists: Bool {
        jobManager.jobs.contains(where: { $0.id == job.id })
    }
    
    var body: some View {
        // If job was deleted, show a message
        if !jobExists {
            VStack(spacing: 16) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 48))
                    .foregroundStyle(.secondary)
                Text("Job Not Found")
                    .font(.title2)
                    .bold()
                Text("This job has been deleted")
                    .foregroundStyle(.secondary)
                Text("Select a job from the sidebar")
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Header
                VStack(alignment: .leading, spacing: 8) {
                    Text(job.name)
                        .font(.title)
                        .bold()
                    
                    HStack {
                        Label(job.status.displayName, systemImage: "circle.fill")
                            .foregroundStyle(statusColor)
                        Text("•")
                            .foregroundStyle(.secondary)
                        Text("Created: \(job.createdAt.formatted(date: .abbreviated, time: .shortened))")
                            .foregroundStyle(.secondary)
                        if job.startedAt != nil {
                            Text("•")
                                .foregroundStyle(.secondary)
                            Text("Duration: \(job.formattedDuration)")
                                .foregroundStyle(.secondary)
                        }
                    }
                    .font(.subheadline)
                }
                
                Divider()
                
                // Status and Actions
                HStack {
                    if job.status == .running {
                        Button {
                            jobManager.pauseJob(job)
                        } label: {
                            Label("Pause", systemImage: "pause.fill")
                        }
                        Button {
                            jobManager.stopJob(job)
                        } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        Menu {
                            Button(role: .destructive) {
                                jobManager.forceKillJob(job)
                            } label: {
                                Label("Force Kill", systemImage: "exclamationmark.triangle.fill")
                            }
                        } label: {
                            Label("Force Kill", systemImage: "exclamationmark.triangle")
                        }
                    } else if job.status == .paused {
                        Button {
                            jobManager.resumeJob(job)
                        } label: {
                            Label("Resume", systemImage: "play.fill")
                        }
                        Button {
                            jobManager.stopJob(job)
                        } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        Menu {
                            Button(role: .destructive) {
                                jobManager.forceKillJob(job)
                            } label: {
                                Label("Force Kill", systemImage: "exclamationmark.triangle.fill")
                            }
                        } label: {
                            Label("Force Kill", systemImage: "exclamationmark.triangle")
                        }
                    } else if job.status == .pending {
                        Button {
                            Task {
                                await jobManager.startJob(job, pythonPath: envManager.pythonPath, venvPython: envManager.venvPython, hfToken: envManager.hfToken)
                            }
                        } label: {
                            Label("Start", systemImage: "play.fill")
                        }
                        Button {
                            jobManager.cancelJob(job)
                        } label: {
                            Label("Cancel", systemImage: "xmark.circle.fill")
                        }
                    } else if job.status == .completed || job.status == .failed || job.status == .cancelled {
                        // Show "Run Again" button for completed, failed, or cancelled jobs
                        Button {
                            // Reset the job state and rerun it
                            jobManager.resetAndRerunJob(job)
                            // Automatically start the job
                            Task {
                                await jobManager.startJob(job, pythonPath: envManager.pythonPath, venvPython: envManager.venvPython, hfToken: envManager.hfToken)
                            }
                        } label: {
                            Label("Run Again", systemImage: "arrow.clockwise")
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    
                    // Allow editing any non-running job in-place via the Training Wizard.
                    if job.status != .running {
                        Button {
                            // Prepare the wizard to edit this job in-place.
                            jobManager.preloadedConfig = job.config
                            jobManager.jobBeingEdited = job
                            openWindow(id: "training-wizard")
                        } label: {
                            Label("Edit…", systemImage: "pencil")
                        }
                        .buttonStyle(.bordered)
                    }
                    
                    Spacer()
                    
                    Button(role: .destructive) {
                        jobManager.deleteJob(job)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
                
                Divider()
                
                // Download Progress (if downloading)
                if job.isDownloading {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Downloading Model")
                            .font(.headline)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            if let status = job.downloadStatus {
                                Text(status)
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                            
                            ProgressView(value: job.downloadProgress)
                                .progressViewStyle(.linear)
                        }
                        .padding(12)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(8)
                    }
                    
                    Divider()
                }
                
                // Output
                VStack(alignment: .leading, spacing: 8) {
                    Text("Output")
                        .font(.headline)
                    
                    ScrollViewReader { proxy in
                        ScrollView {
                            Text(job.output.isEmpty ? "No output yet..." : job.output)
                                .font(.system(.caption, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                                .padding(8)
                                .background(Color(NSColor.controlBackgroundColor))
                                .id("output-end")
                        }
                        .frame(minHeight: 300)
                        .onChange(of: job.output) { _, _ in
                            // Auto-scroll to bottom when new output arrives
                            withAnimation {
                                proxy.scrollTo("output-end", anchor: .bottom)
                            }
                        }
                    }
                }
                
                if let error = job.error {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Error")
                            .font(.headline)
                            .foregroundStyle(.red)
                        Text(error)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.red)
                            .textSelection(.enabled)
                            .padding(8)
                            .background(Color.red.opacity(0.1))
                    }
                }
            }
            .padding()
        }
        .frame(minWidth: 600)
        }
    }
    
    private var statusColor: Color {
        switch job.status {
        case .pending: return .orange
        case .running: return .green
        case .paused: return .yellow
        case .completed: return .blue
        case .failed: return .red
        case .cancelled: return .gray
        }
    }
}

// MARK: - Failure Alert Sheet
struct FailureAlertSheet: View {
    let jobName: String
    let errorMessage: String
    let onViewDetails: () -> Void
    let onDismiss: () -> Void
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Training Job Failed")
                    .font(.headline)
                Spacer()
                Button(action: onDismiss) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Scrollable error message
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text(jobName)
                        .font(.title3)
                        .bold()
                        .padding(.bottom, 4)
                    
                    Text(errorMessage)
                        .font(.body)
                        .textSelection(.enabled)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
            }
            
            Divider()
            
            // Buttons
            HStack {
                Button("View Details") {
                    onViewDetails()
                }
                .buttonStyle(.borderedProminent)
                
                Spacer()
                
                Button("Dismiss") {
                    onDismiss()
                }
                .buttonStyle(.bordered)
            }
            .padding()
        }
        .frame(width: 600, height: 500)
    }
}

#Preview {
    JobsListView()
        .environmentObject(TrainingJobManager.shared)
}


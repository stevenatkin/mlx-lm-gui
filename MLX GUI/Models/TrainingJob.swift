//
//  TrainingJob.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation
import Combine

enum JobStatus: String, Codable {
    case pending = "pending"
    case running = "running"
    case paused = "paused"
    case completed = "completed"
    case failed = "failed"
    case cancelled = "cancelled"
    
    var displayName: String {
        switch self {
        case .pending: return "Pending"
        case .running: return "Running"
        case .paused: return "Paused"
        case .completed: return "Completed"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        }
    }
    
    var icon: String {
        switch self {
        case .pending: return "⏳"
        case .running: return "🟢"
        case .paused: return "⏸️"
        case .completed: return "✅"
        case .failed: return "🔴"
        case .cancelled: return "⚪"
        }
    }
}

@MainActor
class TrainingJob: ObservableObject, Identifiable, Equatable {
    let id: UUID
    @Published var name: String
    @Published var config: TrainingConfiguration
    let configPath: URL
    let createdAt: Date
    @Published var status: JobStatus
    @Published var output: String
    @Published var error: String?
    @Published var startedAt: Date?
    @Published var completedAt: Date?
    @Published var metrics: [String: Double]
    
    // Download progress (not persisted)
    @Published var downloadProgress: Double = 0.0
    @Published var downloadStatus: String? = nil
    @Published var isDownloading: Bool = false
    
    // Process management (not codable)
    var process: Process?
    
    // Download task management (not codable)
    // Store active download tasks and their sessions so we can cancel them
    // These are accessed from background threads (delegate methods) so we use MainActor.run
    var activeDownloadTasks: [URLSessionDownloadTask] = []
    var activeDownloadSessions: [URLSession] = []
    
    init(name: String, config: TrainingConfiguration, configPath: URL) {
        self.id = UUID()
        self.name = name
        self.config = config
        self.configPath = configPath
        self.createdAt = Date()
        self.status = .pending
        self.output = ""
        self.error = nil
        self.metrics = [:]
    }
    
    // Private initializer for loading persisted jobs
    private init(id: UUID, name: String, config: TrainingConfiguration, configPath: URL, createdAt: Date, status: JobStatus, output: String, error: String?, startedAt: Date?, completedAt: Date?, metrics: [String: Double]) {
        self.id = id
        self.name = name
        self.config = config
        self.configPath = configPath
        self.createdAt = createdAt
        self.status = status
        self.output = output
        self.error = error
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.metrics = metrics
    }
    
    // MARK: - Codable Support (via separate struct for persistence)
    struct PersistedJob: Codable {
        let id: UUID
        let name: String
        let config: TrainingConfiguration
        let configPath: String
        let createdAt: Date
        let status: JobStatus
        let output: String
        let error: String?
        let startedAt: Date?
        let completedAt: Date?
        let metrics: [String: Double]
    }
    
    var persistedData: PersistedJob {
        PersistedJob(
            id: id,
            name: name,
            config: config,
            configPath: configPath.path,
            createdAt: createdAt,
            status: status,
            output: output,
            error: error,
            startedAt: startedAt,
            completedAt: completedAt,
            metrics: metrics
        )
    }
    
    convenience init(from persisted: PersistedJob) {
        self.init(
            id: persisted.id,
            name: persisted.name,
            config: persisted.config,
            configPath: URL(fileURLWithPath: persisted.configPath),
            createdAt: persisted.createdAt,
            status: persisted.status,
            output: persisted.output,
            error: persisted.error,
            startedAt: persisted.startedAt,
            completedAt: persisted.completedAt,
            metrics: persisted.metrics
        )
    }
    
    var duration: TimeInterval? {
        guard let started = startedAt else { return nil }
        let end = completedAt ?? Date()
        return end.timeIntervalSince(started)
    }
    
    var formattedDuration: String {
        guard let duration = duration else { return "N/A" }
        let hours = Int(duration) / 3600
        let minutes = (Int(duration) % 3600) / 60
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        } else {
            return "\(minutes)m"
        }
    }
    
    // MARK: - Equatable
    static func == (lhs: TrainingJob, rhs: TrainingJob) -> Bool {
        // Compare by ID since each job has a unique UUID
        return lhs.id == rhs.id
    }
}


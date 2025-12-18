//
//  ModelListManager.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

/// Manages the list of popular models, loading from bundled JSON and user customizations
@MainActor
class ModelListManager {
    static let shared = ModelListManager()
    
    private var cachedModels: [PopularModel]?
    
    private init() {}
    
    /// Load models from bundled JSON file and user customizations
    /// Returns the combined list of models
    func loadModels() -> [PopularModel] {
        // Return cached models if available
        if let cached = cachedModels {
            return cached
        }
        
        var allModels: [PopularModel] = []
        
        // Load default models from bundle
        if let bundledModels = loadBundledModels() {
            allModels.append(contentsOf: bundledModels)
        }
        
        // Load user customizations (if any)
        if let userModels = loadUserModels() {
            // Merge user models, avoiding duplicates by identifier
            let existingIdentifiers = Set(allModels.map { $0.identifier })
            let newUserModels = userModels.filter { !existingIdentifiers.contains($0.identifier) }
            allModels.append(contentsOf: newUserModels)
        }
        
        // Cache the result
        cachedModels = allModels
        
        return allModels
    }
    
    /// Reload models (clears cache and reloads from files)
    func reloadModels() {
        cachedModels = nil
    }
    
    /// Get the path to the user's custom models file
    private var userModelsURL: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let baseDir = appSupport.appendingPathComponent("MLX Training Studio", isDirectory: true)
        return baseDir.appendingPathComponent("custom_models.json")
    }
    
    /// Load models from the bundled JSON file
    private func loadBundledModels() -> [PopularModel]? {
        guard let url = Bundle.main.url(forResource: "popular_models", withExtension: "json") else {
            print("[ModelListManager] Warning: Could not find popular_models.json in bundle")
            return nil
        }
        
        guard let data = try? Data(contentsOf: url) else {
            print("[ModelListManager] Warning: Could not read popular_models.json from bundle")
            return nil
        }
        
        let decoder = JSONDecoder()
        guard let modelList = try? decoder.decode(ModelList.self, from: data) else {
            print("[ModelListManager] Warning: Could not decode popular_models.json")
            return nil
        }
        
        return modelList.models
    }
    
    /// Load user customizations from Application Support
    private func loadUserModels() -> [PopularModel]? {
        let url = userModelsURL
        
        guard FileManager.default.fileExists(atPath: url.path) else {
            // No user customizations file - that's fine
            return nil
        }
        
        guard let data = try? Data(contentsOf: url) else {
            print("[ModelListManager] Warning: Could not read custom_models.json from Application Support")
            return nil
        }
        
        let decoder = JSONDecoder()
        guard let modelList = try? decoder.decode(ModelList.self, from: data) else {
            print("[ModelListManager] Warning: Could not decode custom_models.json")
            return nil
        }
        
        return modelList.models
    }
    
    /// Save user custom models to Application Support
    /// This allows users to add their own models without modifying the app bundle
    func saveUserModels(_ models: [PopularModel]) throws {
        let url = userModelsURL
        
        // Create directory if it doesn't exist
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        
        let modelList = ModelList(models: models)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        let data = try encoder.encode(modelList)
        try data.write(to: url)
        
        // Clear cache so changes are reflected
        reloadModels()
    }
    
    /// Get the path to the user's custom models file (for display purposes)
    var userModelsPath: String {
        userModelsURL.path
    }
}

/// Codable structure for the JSON file
private struct ModelList: Codable {
    let models: [PopularModel]
}

/// Model information structure
struct PopularModel: Codable, Identifiable, Equatable {
    let identifier: String
    let displayName: String
    
    var id: String { identifier }
}


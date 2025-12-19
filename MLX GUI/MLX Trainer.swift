//
//  MLX Trainer.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import SwiftUI
import AppKit

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillTerminate(_ notification: Notification) {
        // Stop all running jobs before app terminates
        // Run synchronously on main thread to ensure cleanup completes before app quits
        // This prevents termination handlers from accessing process info after shutdown
        if Thread.isMainThread {
            TrainingJobManager.shared.stopAllRunningJobs()
        } else {
            DispatchQueue.main.sync {
                TrainingJobManager.shared.stopAllRunningJobs()
            }
        }
    }
}

@main
struct MLX_GUIApp: App {
    @StateObject private var env = EnvironmentManager()
    @StateObject private var jobManager = TrainingJobManager.shared
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(env)
                .environmentObject(jobManager)
        }

        // Standard macOS behavior: App Menu → Settings… opens this window.
        Settings {
            EnvironmentSettingsView()
                .environmentObject(env)
        }
        
        // Training Wizard Window
        WindowGroup("New Training Job", id: "training-wizard") {
            TrainingWizardView()
                .environmentObject(env)
                .environmentObject(jobManager)
        }
        .defaultSize(width: 800, height: 700)
        .commandsRemoved()
    }
}

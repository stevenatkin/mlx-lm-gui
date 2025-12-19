//
//  TrainingWizardView.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import SwiftUI
import UniformTypeIdentifiers
import AppKit

// MARK: - Popular Models List
// Models are loaded from popular_models.json in the app bundle.
// Users can add custom models by creating custom_models.json in Application Support.
// Visit https://huggingface.co/mlx-community to see all available models.

struct TrainingWizardView: View {
    @EnvironmentObject var jobManager: TrainingJobManager
    @EnvironmentObject var envManager: EnvironmentManager
    
    @State private var config: TrainingConfiguration = {
        // Check if there's a preloaded config from TrainingJobManager
        if let preloaded = TrainingJobManager.shared.preloadedConfig {
            // Clear it after using it
            TrainingJobManager.shared.preloadedConfig = nil
            return preloaded
        }
        
        // Otherwise, create default config
        var cfg = TrainingConfiguration()
        // Set default model to first popular model
        let models = ModelListManager.shared.loadModels()
        if let firstModel = models.first {
            cfg.model = firstModel.identifier
        }
        return cfg
    }()
    @State private var currentStep = 0
    @State private var jobName = ""
    
    private let steps = ["Basic", "Training", "LoRA", "Optimizer", "Mode-Specific", "Advanced", "Review"]
    
    var body: some View {
        VStack(spacing: 0) {
            // Title bar
            HStack {
                Text(jobManager.jobBeingEdited == nil ? "New Training Job" : "Edit Training Job")
                    .font(.headline)
                Spacer()
                Button(action: { 
                    if let window = NSApplication.shared.windows.first(where: { $0.identifier?.rawValue == "training-wizard" }) {
                        window.close()
                    }
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Progress indicator
            HStack {
                ForEach(0..<steps.count, id: \.self) { index in
                    Circle()
                        .fill(index <= currentStep ? Color.accentColor : Color.gray.opacity(0.3))
                        .frame(width: 8, height: 8)
                    if index < steps.count - 1 {
                        Rectangle()
                            .fill(index < currentStep ? Color.accentColor : Color.gray.opacity(0.3))
                        .frame(height: 2)
                    }
                }
            }
            .padding()
            
            Divider()
            
            // Step content
            ScrollView {
                VStack {
                    Group {
                        switch currentStep {
                        case 0:
                            BasicSettingsStep(config: $config)
                        case 1:
                            TrainingParamsStep(config: $config)
                        case 2:
                            LoRAStep(config: $config)
                        case 3:
                            OptimizerStep(config: $config)
                        case 4:
                            ModeSpecificStep(config: $config)
                        case 5:
                            AdvancedStep(config: $config)
                                .environmentObject(envManager)
                        case 6:
                            ReviewStep(config: $config, jobName: $jobName)
                        default:
                            BasicSettingsStep(config: $config)
                        }
                    }
                }
                .frame(maxWidth: .infinity)
            }
            .frame(minHeight: 400)
            
            Divider()
            
            // Navigation buttons
            HStack {
                Button("Cancel") {
                    // Clear any edit context and close the wizard
                    jobManager.jobBeingEdited = nil
                    if let window = NSApplication.shared.windows.first(where: { $0.identifier?.rawValue == "training-wizard" }) {
                        window.close()
                    }
                }
                
                Spacer()
                
                if currentStep > 0 {
                    Button("Previous") {
                        withAnimation {
                            currentStep -= 1
                        }
                    }
                }
                
                // Show Save YAML button on the last step, next to Create Job
                if currentStep == steps.count - 1 {
                    Button("Save YAML Configuration…") {
                        saveYAMLConfiguration()
                    }
                    .buttonStyle(.bordered)
                }
                
                let isLastStep = currentStep == steps.count - 1
                let primaryLabel = isLastStep
                    ? (jobManager.jobBeingEdited == nil ? "Create Job" : "Save Changes")
                    : "Next"
                
                Button(primaryLabel) {
                    if isLastStep {
                        createJob()
                    } else {
                        withAnimation {
                            currentStep += 1
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isLastStep && (jobName.isEmpty || !config.validate().isEmpty))
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
        }
        .frame(width: 800, height: 700)
        .background(Color(NSColor.windowBackgroundColor))
        .onAppear {
            // Set window identifier for closing
            if let window = NSApplication.shared.windows.first(where: { $0.title == "New Training Job" || $0.title == "Edit Training Job" }) {
                window.identifier = NSUserInterfaceItemIdentifier("training-wizard")
            }
            
            // If editing an existing job, prefill the job name with the current job's name
            if let editingJob = jobManager.jobBeingEdited, jobName.isEmpty {
                jobName = editingJob.name
            }
        }
    }
    
    private func createJob() {
        do {
            if let editingJob = jobManager.jobBeingEdited {
                // Update existing job in-place
                try jobManager.updateJob(editingJob, newName: jobName, newConfig: config)
                jobManager.jobBeingEdited = nil
            } else {
                // Create a brand new job
                _ = try jobManager.createJob(name: jobName, config: config)
            }
            
            if let window = NSApplication.shared.windows.first(where: { $0.identifier?.rawValue == "training-wizard" }) {
                window.close()
            }
        } catch {
            // Handle error - for now just print
            print("Error creating/updating job: \(error)")
        }
    }
    
    private func saveYAMLConfiguration() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.yaml, .text]
        panel.nameFieldStringValue = jobName.isEmpty ? "training_config.yaml" : "\(jobName.replacingOccurrences(of: " ", with: "_")).yaml"
        panel.canCreateDirectories = true
        panel.title = "Save Training Configuration"
        panel.message = "Choose where to save the YAML configuration file"
        
        panel.begin { response in
            if response == .OK, let url = panel.url {
                do {
                    try YAMLManager.shared.saveConfiguration(self.config, to: url)
                } catch {
                    print("Error saving YAML: \(error)")
                }
            }
        }
    }
}

// MARK: - Wizard Steps

struct BasicSettingsStep: View {
    @Binding var config: TrainingConfiguration
    
    private func autoDetectQuantization() {
        let modelLower = config.model.lowercased()
        // If model is pre-quantized, set quantization to None (don't quantize an already-quantized model)
        if modelLower.contains("4bit") || modelLower.contains("8bit") || modelLower.contains("6bit") {
            config.quantization = nil
        }
        // If model doesn't contain quantization info, leave the current setting as-is
        // (user can manually choose quantization level for full-precision models)
    }
    
    var body: some View {
        Form {
            Section("Training Mode") {
                HStack {
                    Text("Mode:")
                    Picker("", selection: $config.trainMode) {
                        ForEach(TrainingMode.allCases, id: \.self) { mode in
                            Text(mode.displayName).tag(mode)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                Text(config.trainMode.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Section("Model") {
                HStack(alignment: .center, spacing: 8) {
                    Text("Model:")
                    Menu {
                        ForEach(ModelListManager.shared.loadModels(), id: \.identifier) { model in
                            Button(model.displayName) {
                                config.model = model.identifier
                                // Auto-detect quantization from model name
                                autoDetectQuantization()
                            }
                        }
                    } label: {
                        Image(systemName: "list.bullet")
                            .frame(width: 20, height: 20)
                            .padding(4)
                    }
                    .buttonStyle(.bordered)
                    .help("Select a popular model")
                    TextField("", text: $config.model)
                        .textFieldStyle(.roundedBorder)
                        .onChange(of: config.model) { _, _ in
                            autoDetectQuantization()
                        }
                }
                Text("Enter a Hugging Face model identifier or select from popular models. The model will be downloaded automatically if not already cached.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Section("Quantization") {
                let isPreQuantized = config.model.lowercased().contains("4bit") || config.model.lowercased().contains("6bit") || config.model.lowercased().contains("8bit")
                
                HStack {
                    Text("Quantization:")
                    Picker("", selection: $config.quantization) {
                        Text("None").tag(nil as QuantizationType?)
                        ForEach([QuantizationType.bits4, QuantizationType.bits6, QuantizationType.bits8], id: \.self) { q in
                            Text(q.displayName).tag(q as QuantizationType?)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .disabled(isPreQuantized)
                }
                
                if isPreQuantized {
                    Text("This model is already quantized (pre-quantized). The quantization option is disabled because quantizing an already-quantized model is not needed and may cause errors.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                } else {
                    Text("Load the model in quantized format to reduce memory usage. 4-bit uses less memory but may have slightly lower quality. 6-bit provides a balance between memory and quality. 8-bit offers better quality with moderate memory savings. Not applicable for pre-quantized models (those with '4bit', '6bit', or '8bit' in the name).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            
            Section("Data") {
                HStack {
                    Text("Data Path:")
                    TextField("", text: $config.data)
                        .textFieldStyle(.roundedBorder)
                    Button("Browse…") {
                        let panel = NSOpenPanel()
                        panel.allowsMultipleSelection = false
                        panel.canChooseDirectories = true
                        panel.canChooseFiles = true
                        panel.allowedContentTypes = [.item]
                        
                        if panel.runModal() == .OK {
                            if let url = panel.url {
                                config.data = url.path
                            }
                        }
                    }
                }
                Text("Enter a Hugging Face dataset identifier (e.g., 'mlx-community/wikisql') or a local path to a directory containing JSONL files. Note: Hugging Face datasets must already be in a supported format (with train/valid/test splits). Pre-formatted datasets from mlx-community are recommended.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Section("Adapter Output") {
                HStack {
                    Text("Adapter Path:")
                    TextField("", text: Binding(
                        get: { config.adapterPath ?? "" },
                        set: { config.adapterPath = $0.isEmpty ? nil : $0 }
                    ))
                        .textFieldStyle(.roundedBorder)
                    Button("Browse…") {
                        let panel = NSOpenPanel()
                        panel.allowsMultipleSelection = false
                        panel.canChooseDirectories = true
                        panel.canChooseFiles = false
                        
                        if panel.runModal() == .OK {
                            if let url = panel.url {
                                config.adapterPath = url.path
                            }
                        }
                    }
                }
                Text("Where to save the trained adapter. If not specified, a default location will be used (optional).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct TrainingParamsStep: View {
    @Binding var config: TrainingConfiguration
    
    var body: some View {
        Form {
            Section("Training Parameters") {
                HStack {
                    Text("Batch Size:")
                    TextField("", value: $config.batchSize, format: .number)
                        .textFieldStyle(.roundedBorder)
                    Stepper("", value: $config.batchSize, in: 1...128)
                }
                
                HStack {
                    Text("Learning Rate:")
                    TextField("", value: $config.learningRate, format: FloatingPointFormatStyle<Double>.number.notation(.scientific))
                        .textFieldStyle(.roundedBorder)
                }
                Text("Learning rate in scientific notation (e.g., 1e-5).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                HStack {
                    Text("Iterations:")
                    TextField("", value: $config.iterations, format: .number)
                        .textFieldStyle(.roundedBorder)
                    Stepper("", value: $config.iterations, in: 1...100000)
                }
                
                HStack {
                    Text("Max Sequence Length:")
                    TextField("", value: $config.maxSeqLength, format: .number)
                        .textFieldStyle(.roundedBorder)
                }
                Text("Maximum sequence length for training. Default: 2048 (optional).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                HStack {
                    Text("Gradient Accumulation Steps:")
                    TextField("", value: $config.gradientAccumulationSteps, format: .number)
                        .textFieldStyle(.roundedBorder)
                }
                Text("Number of gradient accumulation steps. Default: 1. Effective batch size = batch size × accumulation steps (optional).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                HStack {
                    Text("Gradient Checkpointing:")
                    Toggle("", isOn: $config.gradCheckpoint)
                }
                
                HStack {
                    Text("Seed:")
                    TextField("", value: $config.seed, format: .number)
                        .textFieldStyle(.roundedBorder)
                }
                Text("PRNG seed for reproducibility. Default: 0.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Section("Validation & Reporting") {
                HStack {
                    Text("Validation Batches:")
                    TextField("", value: $config.valBatches, format: .number)
                        .textFieldStyle(.roundedBorder)
                }
                Text("-1 uses entire validation set.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                HStack {
                    Text("Steps Per Report:")
                    TextField("", value: $config.stepsPerReport, format: .number)
                        .textFieldStyle(.roundedBorder)
                }
                
                HStack {
                    Text("Steps Per Eval:")
                    TextField("", value: $config.stepsPerEval, format: .number)
                        .textFieldStyle(.roundedBorder)
                }
                
                HStack {
                    Text("Save Every:")
                    TextField("", value: $config.saveEvery, format: .number)
                        .textFieldStyle(.roundedBorder)
                }
            }
            
            Section("Resume Training") {
                HStack {
                    Text("Resume Adapter File:")
                    TextField("", text: Binding(
                        get: { config.resumeAdapterFile ?? "" },
                        set: { config.resumeAdapterFile = $0.isEmpty ? nil : $0 }
                    ))
                        .textFieldStyle(.roundedBorder)
                    Button("Browse…") {
                        let panel = NSOpenPanel()
                        panel.allowsMultipleSelection = false
                        panel.canChooseDirectories = false
                        panel.canChooseFiles = true
                        panel.allowedContentTypes = [.item]
                        
                        if panel.runModal() == .OK {
                            if let url = panel.url {
                                config.resumeAdapterFile = url.path
                            }
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct LoRAStep: View {
    @Binding var config: TrainingConfiguration
    
    var body: some View {
        Form {
            Section("LoRA Parameters") {
                HStack {
                    Text("Enable LoRA:")
                    Toggle("", isOn: Binding(
                        get: { config.loraParameters != nil },
                        set: { enabled in
                            if enabled {
                                config.loraParameters = LoRAParameters()
                            } else {
                                config.loraParameters = nil
                            }
                        }
                    ))
                }
                
                if config.loraParameters != nil {
                    HStack {
                        Text("Rank:")
                        Stepper("", value: Binding(
                            get: { config.loraParameters!.rank },
                            set: { newRank in
                                let oldRank = config.loraParameters!.rank
                                config.loraParameters!.rank = newRank
                                // Update alpha to 2x rank when rank changes, but only if alpha is currently at the default (2x old rank)
                                let expectedAlpha = Double(oldRank * 2)
                                if abs(config.loraParameters!.alpha - expectedAlpha) < 0.001 {
                                    config.loraParameters!.alpha = Double(newRank * 2)
                                }
                            }
                        ), in: 1...256)
                        Text("\(config.loraParameters!.rank)")
                            .frame(minWidth: 50, alignment: .trailing)
                    }
                    
                    HStack {
                        Text("Alpha:")
                        TextField("", value: Binding(
                            get: { config.loraParameters!.alpha },
                            set: { config.loraParameters!.alpha = $0 }
                        ), format: .number)
                        .textFieldStyle(.roundedBorder)
                    }
                    Text("LoRA alpha parameter. Typically set to 2x rank. Default: \(Int(config.loraParameters!.rank * 2))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    HStack {
                        Text("Dropout:")
                        TextField("", value: Binding(
                            get: { config.loraParameters!.dropout },
                            set: { config.loraParameters!.dropout = $0 }
                        ), format: .number)
                        .textFieldStyle(.roundedBorder)
                    }
                    
                    HStack {
                        Text("Scale:")
                        TextField("", value: Binding(
                            get: { config.loraParameters!.scale },
                            set: { config.loraParameters!.scale = $0 }
                        ), format: .number)
                        .textFieldStyle(.roundedBorder)
                    }
                }
            }
            
            Section("Advanced LoRA") {
                HStack {
                    Text("Number of Layers:")
                    TextField("", text: Binding(
                        get: {
                            if let numLayers = config.numLayers {
                                return String(numLayers)
                            }
                            return ""
                        },
                        set: { newValue in
                            if newValue.isEmpty {
                                config.numLayers = -1
                            } else if let intValue = Int(newValue) {
                                config.numLayers = intValue
                            }
                        }
                    ))
                        .textFieldStyle(.roundedBorder)
                }
                Text("Number of layers to fine-tune. Use -1 to fine-tune all layers (mlx-lm-lora default).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                VStack(alignment: .leading, spacing: 12) {
                    Text("Target Modules:")
                    
                    let keysBinding = Binding<[String]>(
                        get: { 
                            return config.loraParameters?.keys ?? []
                        },
                        set: { 
                            if config.loraParameters == nil {
                                config.loraParameters = LoRAParameters()
                            }
                            config.loraParameters?.keys = $0
                        }
                    )
                    
                    // Self-Attention modules
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Self-Attention")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 4) {
                            ModuleCheckbox(title: "Query Projection (q_proj)", key: "self_attn.q_proj", selectedKeys: keysBinding)
                            ModuleCheckbox(title: "Key Projection (k_proj)", key: "self_attn.k_proj", selectedKeys: keysBinding)
                            ModuleCheckbox(title: "Value Projection (v_proj)", key: "self_attn.v_proj", selectedKeys: keysBinding)
                            ModuleCheckbox(title: "Output Projection (o_proj)", key: "self_attn.o_proj", selectedKeys: keysBinding)
                        }
                    }
                    
                    // MLP modules
                    VStack(alignment: .leading, spacing: 8) {
                        Text("MLP (Multi-Layer Perceptron)")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 4) {
                            ModuleCheckbox(title: "Gate Projection (gate_proj)", key: "mlp.gate_proj", selectedKeys: keysBinding)
                            ModuleCheckbox(title: "Up Projection (up_proj)", key: "mlp.up_proj", selectedKeys: keysBinding)
                            ModuleCheckbox(title: "Down Projection (down_proj)", key: "mlp.down_proj", selectedKeys: keysBinding)
                        }
                    }
                    
                    // Summary field showing selected keys
                    if let lora = config.loraParameters, !lora.keys.isEmpty {
                        HStack {
                            Text("Selected:")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(lora.keys.joined(separator: ", "))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                        }
                        .padding(.top, 4)
                    }
                }
                Text("Select which modules to apply LoRA to. Default: q_proj, k_proj, v_proj")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
        .onAppear {
            // Initialize loraParameters if nil to avoid state modification during view update
            if config.loraParameters == nil {
                config.loraParameters = LoRAParameters()
            }
        }
    }
}

struct OptimizerStep: View {
    @Binding var config: TrainingConfiguration
    
    var body: some View {
        Form {
            Section("Optimizer") {
                HStack {
                    Text("Optimizer:")
                    Picker("", selection: $config.optimizer) {
                        ForEach(OptimizerType.allCases, id: \.self) { opt in
                            Text(opt.displayName).tag(opt)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                
                HStack {
                    Text("Weight Decay:")
                    TextField("", value: $config.weightDecay, format: .number)
                        .textFieldStyle(.roundedBorder)
                }
            }
            
            Section("Learning Rate Schedule") {
                HStack {
                    Text("Schedule:")
                    Picker("", selection: $config.lrSchedule) {
                        ForEach(LRScheduleType.allCases, id: \.self) { schedule in
                            Text(schedule.displayName).tag(schedule)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                
                HStack {
                    Text("Warmup Steps:")
                    TextField("", value: Binding(
                        get: { config.lrScheduleParams?.warmup },
                        set: { 
                            if config.lrScheduleParams == nil {
                                config.lrScheduleParams = LRScheduleParameters()
                            }
                            config.lrScheduleParams?.warmup = $0
                        }
                    ), format: .number)
                    .textFieldStyle(.roundedBorder)
                }
                Text("Number of warmup steps for learning rate schedule. Default: 100. Set to 0 for no warmup. Leave empty to use mlx-lm-lora default (typically 0).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                HStack {
                    Text("Warmup Init:")
                    TextField("", value: Binding(
                        get: { config.lrScheduleParams?.warmupInit },
                        set: { 
                            if config.lrScheduleParams == nil {
                                config.lrScheduleParams = LRScheduleParameters()
                            }
                            config.lrScheduleParams?.warmupInit = $0
                        }
                    ), format: FloatingPointFormatStyle<Double>.number.notation(.scientific))
                    .textFieldStyle(.roundedBorder)
                }
                Text("Initial learning rate for warmup phase. Default: 1e-7. Set to 0 if not specified. (Optional)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                // Decay Steps and Final LR for cosine and linear schedules
                if config.lrSchedule == .cosine || config.lrSchedule == .linear {
                    HStack {
                        Text("Decay Steps:")
                        TextField("", value: Binding(
                            get: { config.lrScheduleParams?.decaySteps },
                            set: { 
                                if config.lrScheduleParams == nil {
                                    config.lrScheduleParams = LRScheduleParameters()
                                }
                                config.lrScheduleParams?.decaySteps = $0
                            }
                        ), format: .number)
                        .textFieldStyle(.roundedBorder)
                    }
                    Text("Total training steps for decay. Defaults to iterations if not set. For cosine_decay and linear_schedule.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    HStack {
                        Text("Final Learning Rate:")
                        TextField("", value: Binding(
                            get: { config.lrScheduleParams?.finalLR },
                            set: { 
                                if config.lrScheduleParams == nil {
                                    config.lrScheduleParams = LRScheduleParameters()
                                }
                                config.lrScheduleParams?.finalLR = $0
                            }
                        ), format: FloatingPointFormatStyle<Double>.number.notation(.scientific))
                        .textFieldStyle(.roundedBorder)
                    }
                    if config.lrSchedule == .cosine {
                        Text("Final learning rate (end parameter). Default: 0.0. For cosine_decay: [init, decay_steps, end]")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Final learning rate (end parameter). Default: 10% of initial LR or 1e-7. For linear_schedule: [init, end, steps]")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                
                HStack {
                    Text("Arguments (Advanced):")
                    TextField("", text: Binding(
                        get: { 
                            guard let args = config.lrScheduleParams?.arguments else { return "" }
                            return args.map { String($0) }.joined(separator: ", ")
                        },
                        set: { 
                            if config.lrScheduleParams == nil {
                                config.lrScheduleParams = LRScheduleParameters()
                            }
                            if $0.isEmpty {
                                config.lrScheduleParams?.arguments = nil
                            } else {
                                let arguments = $0.split(separator: ",").compactMap { arg in
                                    Double(arg.trimmingCharacters(in: .whitespaces))
                                }
                                config.lrScheduleParams?.arguments = arguments.isEmpty ? nil : arguments
                            }
                        }
                    ))
                    .textFieldStyle(.roundedBorder)
                }
                Text("Manual arguments override Decay Steps and Final LR. Comma-separated values (e.g., \"1e-5, 2500, 1e-7\"). (Optional)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct ModeSpecificStep: View {
    @Binding var config: TrainingConfiguration
    
    var body: some View {
        Form {
            judgeModelSection
            dpoCpoSection
            orpoSection
            groupBasedSection
            noParametersSection
        }
        .formStyle(.grouped)
        .padding()
    }
    
    @ViewBuilder
    private var judgeModelSection: some View {
        if config.trainMode.requiresJudgeModel {
                Section("Judge Model (Required)") {
                    HStack(alignment: .center, spacing: 8) {
                        Text("Judge Model:")
                        Menu {
                            ForEach(ModelListManager.shared.loadModels(), id: \.identifier) { model in
                                Button(model.displayName) {
                                    if config.onlineParams == nil {
                                        config.onlineParams = OnlineParameters()
                                    }
                                    config.onlineParams?.judgeModel = model.identifier
                                }
                            }
                        } label: {
                            Image(systemName: "list.bullet")
                                .frame(width: 20, height: 20)
                                .padding(4)
                        }
                        .buttonStyle(.bordered)
                        .help("Select a popular model")
                        TextField("Enter model identifier", text: Binding(
                            get: { config.onlineParams?.judgeModel ?? "" },
                            set: {
                                if config.onlineParams == nil {
                                    config.onlineParams = OnlineParameters()
                                }
                                config.onlineParams?.judgeModel = $0
                            }
                        ))
                        .textFieldStyle(.roundedBorder)
                    }
                    Text("Enter a Hugging Face model identifier or select from popular models for the judge/reward model.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    HStack {
                        Text("Alpha:")
                        TextField("", value: Binding(
                            get: { config.onlineParams?.alpha ?? 1e-5 },
                            set: {
                                if config.onlineParams == nil {
                                    config.onlineParams = OnlineParameters()
                                }
                                config.onlineParams?.alpha = $0
                            }
                        ), format: FloatingPointFormatStyle<Double>.number.notation(.scientific))
                        .textFieldStyle(.roundedBorder)
                    }
                    Text("Alpha parameter in scientific notation (e.g., 1e-5).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    if config.trainMode == .xpo || config.trainMode == .rlhfReinforce {
                        HStack {
                            Text("Beta:")
                            TextField("", value: Binding(
                                get: { config.onlineParams?.beta ?? 0.1 },
                                set: {
                                    if config.onlineParams == nil {
                                        config.onlineParams = OnlineParameters()
                                    }
                                    config.onlineParams?.beta = $0
                                }
                            ), format: .number)
                            .textFieldStyle(.roundedBorder)
                        }
                        Text("KL penalty strength (default: 0.1)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    
                    if config.trainMode == .ppo {
                        HStack {
                            Text("Epsilon:")
                            TextField("", value: Binding(
                                get: { config.onlineParams?.epsilon ?? 0.2 },
                                set: {
                                    if config.onlineParams == nil {
                                        config.onlineParams = OnlineParameters()
                                    }
                                    config.onlineParams?.epsilon = $0
                                }
                            ), format: .number)
                            .textFieldStyle(.roundedBorder)
                        }
                        Text("Epsilon for numerical stability (default: 0.2)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    
                    HStack {
                        Text("Judge Config:")
                        TextField("", text: Binding(
                            get: { config.onlineParams?.judgeConfig ?? "" },
                            set: {
                                if config.onlineParams == nil {
                                    config.onlineParams = OnlineParameters()
                                }
                                config.onlineParams?.judgeConfig = $0.isEmpty ? nil : $0
                            }
                        ))
                        .textFieldStyle(.roundedBorder)
                    }
                    Text("Additional configuration for judge model (optional).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
    }
    
    @ViewBuilder
    private var orpoSection: some View {
        if config.trainMode == .orpo {
            Section("ORPO Parameters") {
                HStack {
                    Text("Beta:")
                    TextField("", value: Binding(
                        get: { config.orpoParams?.beta ?? 0.1 },
                        set: {
                            if config.orpoParams == nil {
                                config.orpoParams = ORPOParameters()
                            }
                            config.orpoParams?.beta = $0
                        }
                    ), format: .number)
                    .textFieldStyle(.roundedBorder)
                }
                Text("Beta parameter for ORPO (default: 0.1)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                HStack {
                    Text("Reward Scaling:")
                    TextField("", value: Binding(
                        get: { config.orpoParams?.rewardScaling ?? 1.0 },
                        set: {
                            if config.orpoParams == nil {
                                config.orpoParams = ORPOParameters()
                            }
                            config.orpoParams?.rewardScaling = $0
                        }
                    ), format: .number)
                    .textFieldStyle(.roundedBorder)
                }
                Text("Reward scaling factor (default: 1.0)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .onAppear {
                if config.orpoParams == nil {
                    config.orpoParams = ORPOParameters()
                }
            }
        }
    }
    
    @ViewBuilder
    private var dpoCpoSection: some View {
        if config.trainMode.requiresReferenceModel {
                Section("DPO/CPO Parameters") {
                    HStack {
                        Text("Beta:")
                        TextField("", value: Binding(
                            get: { config.dpoParams?.beta ?? 0.1 },
                            set: {
                                if config.dpoParams == nil {
                                    config.dpoParams = DPOParameters()
                                }
                                config.dpoParams?.beta = $0
                            }
                        ), format: .number)
                        .textFieldStyle(.roundedBorder)
                    }
                    
                    HStack {
                        Text("Loss Type:")
                        TextField("", text: Binding(
                            get: { config.dpoParams?.lossType ?? "sigmoid" },
                            set: {
                                if config.dpoParams == nil {
                                    config.dpoParams = DPOParameters()
                                }
                                config.dpoParams?.lossType = $0
                            }
                        ))
                        .textFieldStyle(.roundedBorder)
                    }
                    Text("Loss function: sigmoid (default), hinge, ipo, or dpop")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    if config.dpoParams?.lossType == "hinge" {
                        HStack {
                            Text("Delta:")
                            TextField("", value: Binding(
                                get: { config.dpoParams?.delta ?? 50.0 },
                                set: {
                                    if config.dpoParams == nil {
                                        config.dpoParams = DPOParameters()
                                    }
                                    config.dpoParams?.delta = $0
                                }
                            ), format: .number)
                            .textFieldStyle(.roundedBorder)
                        }
                        Text("Margin for hinge loss (default: 50.0)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    
                    HStack {
                        Text("Reference Model Path:")
                        TextField("", text: Binding(
                            get: { config.referenceModelPath ?? "" },
                            set: { config.referenceModelPath = $0.isEmpty ? nil : $0 }
                        ))
                        .textFieldStyle(.roundedBorder)
                    }
                    Text("Path to reference model (optional). Uses main model if not specified.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .onAppear {
                    // Auto-initialize if nil
                    if config.dpoParams == nil {
                        config.dpoParams = DPOParameters()
                    }
                }
            }
    }
    
    @ViewBuilder
    private var groupBasedSection: some View {
        if config.trainMode.supportsGroupSize {
            Section("Group-Based Parameters") {
                basicGRPOParameters
                epsilonField
                rewardFunctionsSection
                grpoLossTypePicker
            }
            .onAppear {
                // Auto-initialize if nil
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
            }
        }
    }
    
    @ViewBuilder
    private var basicGRPOParameters: some View {
        HStack {
            Text("Group Size:")
            TextField("", value: groupSizeBinding, format: .number)
                .textFieldStyle(.roundedBorder)
            Stepper("", value: groupSizeBinding, in: 1...32)
        }
        
        HStack {
            Text("Temperature:")
            TextField("", value: temperatureBinding, format: .number)
                .textFieldStyle(.roundedBorder)
        }
        
        HStack {
            Text("Max Completion Length:")
            TextField("", value: maxCompletionLengthBinding, format: .number)
                .textFieldStyle(.roundedBorder)
        }
    }
    
    @ViewBuilder
    private var epsilonField: some View {
        if config.trainMode == .grpo || config.trainMode == .drGrpo {
            HStack {
                Text("Epsilon:")
                TextField("", value: epsilonBinding, format: .number)
                    .textFieldStyle(.roundedBorder)
            }
            Text("Numerical stability constant (default: 1e-4)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
    
    @ViewBuilder
    private var rewardFunctionsSection: some View {
        if config.trainMode.supportsRewardFunctions {
            HStack {
                Text("Reward Functions File:")
                TextField("", text: rewardFunctionsFileBinding)
                    .textFieldStyle(.roundedBorder)
                Button("Browse…") {
                    let panel = NSOpenPanel()
                    panel.allowsMultipleSelection = false
                    panel.canChooseDirectories = false
                    panel.canChooseFiles = true
                    panel.allowedContentTypes = [.pythonScript, .text, .item]
                    
                    if panel.runModal() == .OK {
                        if let url = panel.url {
                            if config.grpoParams == nil {
                                config.grpoParams = GRPOParameters()
                            }
                            config.grpoParams?.rewardFunctionsFile = url.path
                        }
                    }
                }
            }
            Text("Path to custom reward functions Python file (optional). If provided, use --reward-functions to specify which functions to use.")
                .font(.caption)
                .foregroundStyle(.secondary)
            
            HStack {
                Text("Reward Functions:")
                TextField("", text: rewardFunctionsBinding)
                    .textFieldStyle(.roundedBorder)
            }
            Text("Comma-separated reward function names (e.g., \"accuracy_reward,format_reward\")")
                .font(.caption)
                .foregroundStyle(.secondary)
            
            HStack {
                Text("Reward Weights:")
                TextField("", text: rewardWeightsBinding)
                    .textFieldStyle(.roundedBorder)
            }
            Text("Comma-separated weights for each reward function (e.g., \"0.7, 0.3\"). Must match the number of reward functions.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
    
    @ViewBuilder
    private var grpoLossTypePicker: some View {
        if config.trainMode == .drGrpo {
            HStack {
                Text("GRPO Loss Type:")
                Picker("", selection: grpoLossTypeBinding) {
                    Text("grpo").tag("grpo")
                    Text("bnpo").tag("bnpo")
                    Text("dr_grpo").tag("dr_grpo")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Text("Loss variant: grpo (default), bnpo, or dr_grpo")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
    
    // MARK: - Binding Helpers
    
    private var groupSizeBinding: Binding<Int> {
        Binding(
            get: { config.grpoParams?.groupSize ?? 4 },
            set: {
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.groupSize = $0
            }
        )
    }
    
    private var temperatureBinding: Binding<Double> {
        Binding(
            get: { config.grpoParams?.temperature ?? 0.8 },
            set: {
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.temperature = $0
            }
        )
    }
    
    private var maxCompletionLengthBinding: Binding<Int> {
        Binding(
            get: { config.grpoParams?.maxCompletionLength ?? 512 },
            set: {
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.maxCompletionLength = $0
            }
        )
    }
    
    private var epsilonBinding: Binding<Double> {
        Binding(
            get: { config.grpoParams?.epsilon ?? 1e-4 },
            set: {
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.epsilon = $0
            }
        )
    }
    
    private var rewardFunctionsFileBinding: Binding<String> {
        Binding(
            get: { config.grpoParams?.rewardFunctionsFile ?? "" },
            set: {
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.rewardFunctionsFile = $0.isEmpty ? nil : $0
            }
        )
    }
    
    private var rewardFunctionsBinding: Binding<String> {
        Binding(
            get: { config.grpoParams?.rewardFunctions.joined(separator: ", ") ?? "" },
            set: {
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                // Parse comma-separated function names
                let functions = $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                config.grpoParams?.rewardFunctions = functions
            }
        )
    }
    
    private var rewardWeightsBinding: Binding<String> {
        Binding(
            get: {
                let weights = config.grpoParams?.rewardWeights ?? []
                return weights.map { String($0) }.joined(separator: ", ")
            },
            set: {
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                // Parse comma-separated weights
                let weights = $0.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
                config.grpoParams?.rewardWeights = weights
            }
        )
    }
    
    private var grpoLossTypeBinding: Binding<String> {
        Binding(
            get: { config.grpoParams?.grpoLossType ?? "grpo" },
            set: {
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.grpoLossType = $0
            }
        )
    }
    
    @ViewBuilder
    private var noParametersSection: some View {
        // Show message if no mode-specific parameters are needed
        if !config.trainMode.requiresJudgeModel && 
           !config.trainMode.requiresReferenceModel && 
           !config.trainMode.supportsGroupSize {
            Section("Mode-Specific Parameters") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("No additional parameters required")
                        .font(.headline)
                    Text("The selected training mode (\(config.trainMode.displayName)) does not require any mode-specific parameters. You can proceed to the next step.")
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 8)
            }
        }
    }
}

struct AdvancedStep: View {
    @Binding var config: TrainingConfiguration
    @EnvironmentObject var envManager: EnvironmentManager
    
    var body: some View {
        Form {
            if config.trainMode == .sft {
                Section("Training Type") {
                    HStack {
                        Text("Train Type:")
                        Picker("", selection: Binding(
                            get: { config.trainType ?? .lora },
                            set: { config.trainType = $0 }
                        )) {
                            ForEach(TrainType.allCases, id: \.self) { type in
                                Text(type.displayName).tag(type)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    Text("LoRA: Low-rank adaptation (default, efficient). DoRA: Weight-decomposed LoRA. Full: Full fine-tuning (requires more memory).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    HStack {
                        Text("Mask Prompt:")
                        Toggle("", isOn: $config.maskPrompt)
                    }
                    Text("Apply loss only to assistant responses (useful for instruction tuning).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            
            Section("WandB") {
                HStack {
                    Text("Project Name:")
                    TextField("", text: Binding(
                        get: { config.wandbProject ?? "" },
                        set: { config.wandbProject = $0.isEmpty ? nil : $0 }
                    ))
                        .textFieldStyle(.roundedBorder)
                }
                Text("Weights & Biases project name for logging (optional).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Section("Testing") {
                HStack {
                    Text("Test Mode:")
                    Toggle("", isOn: $config.testMode)
                }
                if config.testMode {
                    HStack {
                        Text("Test Batches:")
                        TextField("", value: Binding(
                            get: { config.testBatches ?? 100 },
                            set: { config.testBatches = $0 }
                        ), format: .number)
                        .textFieldStyle(.roundedBorder)
                        Stepper("", value: Binding(
                            get: { config.testBatches ?? 100 },
                            set: { config.testBatches = $0 }
                        ), in: 1...1000)
                    }
                }
            }
            
            Section("Other") {
                HStack {
                    Text("Fuse Adapters:")
                    Toggle("", isOn: $config.fuse)
                        .disabled(config.enablePostTrainingQuantization)
                }
                if config.enablePostTrainingQuantization {
                    Text("Fuse is automatically enabled when post-training quantization is enabled.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            
            Section("Post-Training Quantization") {
                HStack {
                    Text("Enable Post-Training Quantization:")
                    Toggle("", isOn: Binding(
                        get: { config.enablePostTrainingQuantization },
                        set: { newValue in
                            config.enablePostTrainingQuantization = newValue
                            // Automatically enable fuse when quantization is enabled
                            if newValue {
                                config.fuse = true
                            }
                        }
                    ))
                }
                Text("Quantize the model after training completes to reduce memory usage and improve inference speed. Fuse will be automatically enabled.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                if config.enablePostTrainingQuantization {
                    HStack {
                        Text("Quantization Type:")
                        Picker("", selection: $config.postTrainingQuantization) {
                            Text("4-bit").tag(QuantizationType.bits4 as QuantizationType?)
                            Text("8-bit").tag(QuantizationType.bits8 as QuantizationType?)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    Text("4-bit uses less memory but may have slightly lower quality. 8-bit provides a better quality/size balance.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    HStack {
                        Text("Quantized Model Path:")
                        TextField("", text: Binding(
                            get: { 
                                if let path = config.quantizedModelPath {
                                    return path
                                } else if let adapterPath = config.adapterPath {
                                    return URL(fileURLWithPath: adapterPath).appendingPathComponent("quantized").path
                                } else {
                                    return ""
                                }
                            },
                            set: { config.quantizedModelPath = $0.isEmpty ? nil : $0 }
                        ))
                        .textFieldStyle(.roundedBorder)
                        Button("Browse…") {
                            let panel = NSOpenPanel()
                            panel.allowsMultipleSelection = false
                            panel.canChooseDirectories = true
                            panel.canChooseFiles = false
                            
                            if panel.runModal() == .OK {
                                if let url = panel.url {
                                    config.quantizedModelPath = url.path
                                }
                            }
                        }
                    }
                    Text("Where to save the quantized model. Defaults to adapter path + 'quantized' subfolder if not specified. If no adapter path is set, the base model will be quantized directly.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            
            Section("GGUF Conversion (Optional)") {
                if envManager.convertHfToGgufScriptPath == nil {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("GGUF conversion is not available")
                            .font(.headline)
                        Text("To enable GGUF conversion, set the llama.cpp repository path in Settings. This allows you to convert trained models to GGUF format for use with llama.cpp and compatible tools.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 8)
                } else {
                    HStack {
                        Text("Convert to GGUF:")
                        Toggle("", isOn: Binding(
                            get: { config.enableGGUFConversion },
                            set: { newValue in
                                config.enableGGUFConversion = newValue
                                // Automatically enable fuse when GGUF conversion is enabled
                                if newValue {
                                    config.fuse = true
                                }
                            }
                        ))
                    }
                    Text("Convert the trained model to GGUF format after training completes. Fuse will be automatically enabled. Requires llama.cpp repository path to be set in Settings.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    if config.enableGGUFConversion {
                        HStack {
                            Text("GGUF Output Type:")
                            Picker("", selection: $config.ggufOutType) {
                                ForEach(GGUFOutType.allCases, id: \.self) { outType in
                                    Text(outType.displayName).tag(outType)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        Text("Choose how the GGUF file should be stored. \"Auto\" lets the converter pick based on the model; numeric types (f16, bf16, q8_0, etc.) control precision and size.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        
                        HStack {
                            Text("GGUF Output Path:")
                            TextField("", text: Binding(
                                get: {
                                    if let path = config.ggufOutputPath {
                                        return path
                                    }
                                    
                                    // Default preview: descriptive file name under the adapter directory
                                    guard let adapterPath = config.adapterPath, !adapterPath.isEmpty else {
                                        return ""
                                    }
                                    
                                    let baseModelId = config.model
                                    let rawModelName = baseModelId.components(separatedBy: "/").last ?? baseModelId
                                    let cleanedModelName = rawModelName
                                        .replacingOccurrences(of: "-4bit", with: "")
                                        .replacingOccurrences(of: "-8bit", with: "")
                                        .replacingOccurrences(of: "-6bit", with: "")
                                        .replacingOccurrences(of: "-3bit", with: "")
                                    let defaultStemBase = cleanedModelName.isEmpty ? "model-trained" : "\(cleanedModelName)-mlx-trained"
                                    let safeStem = defaultStemBase
                                        .replacingOccurrences(of: " ", with: "_")
                                        .replacingOccurrences(of: "/", with: "_")
                                    
                                    return URL(fileURLWithPath: adapterPath).appendingPathComponent("\(safeStem).gguf").path
                                },
                                set: { config.ggufOutputPath = $0.isEmpty ? nil : $0 }
                            ))
                            .textFieldStyle(.roundedBorder)
                            Button("Browse…") {
                                let panel = NSOpenPanel()
                                panel.allowsMultipleSelection = false
                                panel.canChooseDirectories = true
                                panel.canChooseFiles = false
                                
                                if panel.runModal() == .OK {
                                    if let url = panel.url {
                                        config.ggufOutputPath = url.path
                                    }
                                }
                            }
                        }
                        Text("Where to save the GGUF model. By default, it will be saved next to the fused adapters with a descriptive name based on the base model (for example, \"Phi-3-mini-128k-instruct-mlx-trained.gguf\"). If you enter a directory path, the app will create the GGUF file inside that directory.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct ReviewStep: View {
    @Binding var config: TrainingConfiguration
    @Binding var jobName: String
    
    // Generate a default job name from the model identifier
    private var defaultJobName: String {
        let modelId = config.model
        // Extract model name (e.g., "mlx-community/DeepSeek-V3-4bit" -> "DeepSeek-V3-4bit")
        let modelName = modelId.components(separatedBy: "/").last ?? modelId
        // Remove common suffixes for cleaner name
        let cleanName = modelName
            .replacingOccurrences(of: "-4bit", with: "")
            .replacingOccurrences(of: "-8bit", with: "")
            .replacingOccurrences(of: "-6bit", with: "")
            .replacingOccurrences(of: "-3bit", with: "")
        // Add timestamp for uniqueness
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, HH:mm"
        let timestamp = formatter.string(from: Date())
        return "\(cleanName) - \(timestamp)"
    }
    
    var body: some View {
        Form {
            Section("Job Name") {
                HStack(alignment: .firstTextBaseline) {
                    HStack(spacing: 2) {
                        Text("Job Name")
                            .fontWeight(.semibold)
                        Text("*")
                            .foregroundStyle(.red)
                    }
                    Spacer()
                }
                TextField("", text: $jobName)
                    .textFieldStyle(.roundedBorder)
                    .onAppear {
                        // Set default name if empty
                        if jobName.isEmpty {
                            jobName = defaultJobName
                        }
                    }
            }
            
            Section("Configuration Preview") {
                ScrollView {
                    Text(YAMLManager.shared.configurationToYAML(config))
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(8)
                        .background(Color(NSColor.controlBackgroundColor))
                }
                .frame(minHeight: 300)
            }
            
            if !config.validate().isEmpty {
                Section("Validation Errors") {
                    ForEach(config.validate(), id: \.self) { error in
                        Text(error)
                            .foregroundStyle(.red)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - Helper Views

struct ModuleCheckbox: View {
    let title: String
    let key: String
    @Binding var selectedKeys: [String]
    
    var isSelected: Bool {
        selectedKeys.contains(key)
    }
    
    var body: some View {
        Toggle(isOn: Binding(
            get: { isSelected },
            set: { isOn in
                if isOn {
                    if !selectedKeys.contains(key) {
                        selectedKeys.append(key)
                    }
                } else {
                    selectedKeys.removeAll { $0 == key }
                }
            }
        )) {
            Text(title)
                .font(.system(size: 13))
        }
    }
}


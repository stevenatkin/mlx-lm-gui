//
//  YAMLManager.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

class YAMLManager {
    static let shared = YAMLManager()
    
    private init() {}
    
    // MARK: - Helper Functions
    
    /// Quotes a string value for YAML output, escaping internal quotes
    private func quoteString(_ value: String) -> String {
        // Escape any double quotes in the string
        let escaped = value.replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }
    
    /// Unquotes a string value from YAML input, handling escaped quotes
    private func unquoteString(_ value: String) -> String {
        var unquoted = value.trimmingCharacters(in: .whitespaces)
        
        // Remove surrounding quotes if present
        if unquoted.hasPrefix("\"") && unquoted.hasSuffix("\"") {
            unquoted = String(unquoted.dropFirst().dropLast())
        } else if unquoted.hasPrefix("'") && unquoted.hasSuffix("'") {
            unquoted = String(unquoted.dropFirst().dropLast())
        }
        
        // Unescape quotes
        unquoted = unquoted.replacingOccurrences(of: "\\\"", with: "\"")
        
        return unquoted
    }
    
    /// Format a number for YAML to ensure it's parsed as a number, not a string
    /// Matches the format used in mlx-lm-lora example YAML files
    private func formatNumberForYAML(_ value: Double) -> String {
        // For integers (like iterations), format as integer (no decimal point)
        if value.truncatingRemainder(dividingBy: 1.0) == 0.0 {
            return String(Int(value))
        }
        
        // For very small numbers (< 0.001), use scientific notation like the examples
        // Format as "1e-5" not "1e-05" to match mlx-lm-lora examples
        if abs(value) < 0.001 {
            let formatter = NumberFormatter()
            formatter.numberStyle = .scientific
            formatter.exponentSymbol = "e"
            formatter.usesGroupingSeparator = false
            formatter.maximumFractionDigits = 6
            formatter.minimumFractionDigits = 0
            // Ensure exponent doesn't have leading zeros (e.g., "1e-5" not "1e-05")
            if let formatted = formatter.string(from: NSNumber(value: value)) {
                // Normalize the format: replace "e+0" with "e+", "e-0" with "e-", etc.
                let normalized = formatted.replacingOccurrences(of: "e+0", with: "e+")
                    .replacingOccurrences(of: "e-0", with: "e-")
                return normalized
            }
        }
        
        // For regular numbers, use decimal format
        // Use enough precision but avoid unnecessary trailing zeros
        let formatted = String(format: "%.10g", value)
        return formatted
    }
    
    // MARK: - Configuration to YAML
    
    func configurationToYAML(_ config: TrainingConfiguration) -> String {
        var yaml = ""
        
        // Basic settings
        yaml += "model: \(quoteString(config.model))\n"
        yaml += "data: \(quoteString(config.data))\n"
        yaml += "train: true\n"
        
        // train_type
        if let trainType = config.trainType {
            yaml += "train_type: \(trainType.rawValue)\n"
        } else if config.loraParameters != nil {
            // Default to lora when LoRA parameters are present
            yaml += "train_type: lora\n"
        }
        
        if config.maskPrompt {
            yaml += "mask_prompt: true\n"
        }
        
        yaml += "train_mode: \(config.trainMode.rawValue)\n"
        
        if let adapterPath = config.adapterPath {
            yaml += "adapter_path: \(quoteString(adapterPath))\n"
        }
        
        if let referenceModelPath = config.referenceModelPath {
            yaml += "reference_model_path: \(quoteString(referenceModelPath))\n"
        }
        
        // Training parameters
        yaml += "batch_size: \(config.batchSize)\n"
        yaml += "learning_rate: \(config.learningRate)\n"
        yaml += "iters: \(config.iterations)\n"
        
        if let maxSeqLength = config.maxSeqLength {
            yaml += "max_seq_length: \(maxSeqLength)\n"
        } else {
            yaml += "# max_seq_length: (using mlx-lm-lora default, typically model-dependent)\n"
        }
        
        if let gradAccum = config.gradientAccumulationSteps {
            yaml += "gradient_accumulation_steps: \(gradAccum)\n"
        } else {
            yaml += "# gradient_accumulation_steps: (using mlx-lm-lora default, typically 1)\n"
        }
        
        if config.gradCheckpoint {
            yaml += "grad_checkpoint: true\n"
        }
        
        yaml += "seed: \(config.seed)  # The PRNG seed\n"
        
        // Validation & Reporting
        if let valBatches = config.valBatches {
            yaml += "val_batches: \(valBatches)\n"
        }
        if let stepsPerReport = config.stepsPerReport {
            yaml += "steps_per_report: \(stepsPerReport)\n"
        }
        if let stepsPerEval = config.stepsPerEval {
            yaml += "steps_per_eval: \(stepsPerEval)\n"
        }
        if let saveEvery = config.saveEvery {
            yaml += "save_every: \(saveEvery)\n"
        }
        
        // Resume training
        if let resumeAdapter = config.resumeAdapterFile, !resumeAdapter.isEmpty {
            yaml += "resume_adapter_file: \(quoteString(resumeAdapter))\n"
        }
        
        // LoRA parameters
        if let lora = config.loraParameters {
            yaml += "lora_parameters:\n"
            // Output keys as YAML array
            yaml += "  keys: ["
            let keysString = lora.keys.map { quoteString($0) }.joined(separator: ", ")
            yaml += keysString
            yaml += "]\n"
            yaml += "  rank: \(lora.rank)\n"
            yaml += "  alpha: \(lora.alpha)\n"
            yaml += "  scale: \(lora.scale)\n"
            yaml += "  dropout: \(lora.dropout)\n"
        }
        
        if let numLayers = config.numLayers {
            if numLayers == -1 {
                yaml += "num_layers: \(numLayers)  # -1 means all layers (default)\n"
            } else {
                yaml += "num_layers: \(numLayers)\n"
            }
        } else {
            yaml += "# num_layers: (using mlx-lm-lora default, all layers)\n"
        }
        
        // Optimizer and optimizer_config
        yaml += "optimizer: \(config.optimizer.rawValue)\n"
        // weight_decay is only supported by adamw optimizer
        if config.optimizer == .adamw, let weightDecay = config.weightDecay, weightDecay != 0.0 {
            yaml += "optimizer_config:\n"
            yaml += "  adamw:\n"
            yaml += "    weight_decay: \(weightDecay)\n"
        }
        
        // Learning rate schedule - output as nested structure
        // Only output if user has customized it (not default values)
        if let lrSchedule = config.lrScheduleParams {
            yaml += "lr_schedule:\n"
            yaml += "  name: \(lrSchedule.name.yamlName)\n"
            if let warmup = lrSchedule.warmup {
                yaml += "  warmup: \(warmup)  # 0 for no warmup\n"
            }
            if let warmupInit = lrSchedule.warmupInit {
                yaml += "  warmup_init: \(formatNumberForYAML(warmupInit))\n"
            }
            // arguments is REQUIRED by build_schedule, so always include it
            // decaySteps and finalLR are internal helper fields - they construct the arguments array
            if let arguments = lrSchedule.arguments, !arguments.isEmpty {
                // Format numbers properly to ensure YAML parses them as numbers, not strings
                let argsString = arguments.map { arg in
                    formatNumberForYAML(arg)
                }.joined(separator: ", ")
                yaml += "  arguments: [\(argsString)]  # passed to scheduler\n"
            } else {
                // Generate arguments from decaySteps/finalLR if set, otherwise use defaults
                // cosine_decay(init, decay_steps, end=0.0) needs: [init, decay_steps, end]
                // linear_schedule(init, end, steps) needs: [init, end, steps] - note different order!
                let defaultArgs: [Double]
                switch lrSchedule.name {
                case .cosine:
                    // cosine_decay(init, decay_steps, end=0.0)
                    // Arguments: [initial_lr, decay_steps, end_lr]
                    let decaySteps = Double(lrSchedule.decaySteps ?? config.iterations)
                    let endLR = lrSchedule.finalLR ?? 0.0  // Default to 0.0 for cosine_decay
                    defaultArgs = [config.learningRate, decaySteps, endLR]
                case .linear:
                    // linear_schedule(init, end, steps) - note: end comes before steps!
                    // Arguments: [initial_lr, end_lr, steps]
                    let endLR = lrSchedule.finalLR ?? max(config.learningRate * 0.1, 1e-7)  // Default to 10% of init or 1e-7
                    let steps = Double(lrSchedule.decaySteps ?? config.iterations)
                    defaultArgs = [config.learningRate, endLR, steps]
                case .constant:
                    // For constant, just use the learning rate
                    defaultArgs = [config.learningRate]
                }
                let argsString = defaultArgs.map { formatNumberForYAML($0) }.joined(separator: ", ")
                yaml += "  arguments: [\(argsString)]  # passed to scheduler\n"
            }
        }
        
        // Mode-specific parameters
        if let dpoParams = config.dpoParams {
            yaml += "beta: \(dpoParams.beta)\n"
            yaml += "dpo_cpo_loss_type: \(quoteString(dpoParams.lossType))\n"
            if let delta = dpoParams.delta {
                yaml += "delta: \(delta)\n"
            }
        }
        
        if let orpoParams = config.orpoParams {
            yaml += "beta: \(orpoParams.beta)\n"
            yaml += "reward_scaling: \(orpoParams.rewardScaling)\n"
        }
        
        if let grpoParams = config.grpoParams {
            yaml += "group_size: \(grpoParams.groupSize)\n"
            yaml += "temperature: \(grpoParams.temperature)\n"
            yaml += "max_completion_length: \(grpoParams.maxCompletionLength)\n"
            
            if let epsilon = grpoParams.epsilon {
                yaml += "epsilon: \(epsilon)\n"
            }
            
            if let rewardFunctionsFile = grpoParams.rewardFunctionsFile, !rewardFunctionsFile.isEmpty {
                yaml += "reward_functions_file: \(quoteString(rewardFunctionsFile))\n"
            }
            
            if !grpoParams.rewardFunctions.isEmpty {
                yaml += "reward_functions: \"\(grpoParams.rewardFunctions.joined(separator: ","))\"\n"
            }
            
            if !grpoParams.rewardWeights.isEmpty {
                let weightsString = grpoParams.rewardWeights.map { String($0) }.joined(separator: ", ")
                yaml += "reward_weights: [\(weightsString)]\n"
            }
            
            if let importanceSampling = grpoParams.importanceSamplingLevel {
                yaml += "importance_sampling_level: \(importanceSampling)\n"
            }
            
            if let lossType = grpoParams.grpoLossType {
                yaml += "grpo_loss_type: \(quoteString(lossType))\n"
            }
            
            if let epsilonLow = grpoParams.epsilonLow {
                yaml += "epsilon: \(epsilonLow)\n"
            }
            
            if let epsilonHigh = grpoParams.epsilonHigh {
                yaml += "epsilon_high: \(epsilonHigh)\n"
            }
        }
        
        if let onlineParams = config.onlineParams {
            if !onlineParams.judgeModel.isEmpty {
                yaml += "judge: \(quoteString(onlineParams.judgeModel))\n"
            }
            yaml += "alpha: \(onlineParams.alpha)\n"
            
            if let beta = onlineParams.beta {
                yaml += "beta: \(beta)\n"
            }
            
            if let epsilon = onlineParams.epsilon {
                yaml += "epsilon: \(epsilon)\n"
            }
            
            if let groupSize = onlineParams.groupSize {
                yaml += "group_size: \(groupSize)\n"
            }
            
            if let judgeConfig = onlineParams.judgeConfig, !judgeConfig.isEmpty {
                yaml += "judge_config: \(quoteString(judgeConfig))\n"
            }
        }
        
        // Advanced options
        if let quantization = config.quantization {
            switch quantization {
            case .bits4:
                yaml += "load_in_4bits: true\n"
            case .bits6:
                yaml += "load_in_6bits: true\n"
            case .bits8:
                yaml += "load_in_8bits: true\n"
            case .none:
                break
            }
        }
        
        if let wandb = config.wandbProject {
            yaml += "wand: \(quoteString(wandb))\n"
        }
        
        if config.testMode {
            yaml += "test: true\n"
            if let testBatches = config.testBatches {
                yaml += "test_batches: \(testBatches)\n"
            }
        }
        
        if config.fuse {
            yaml += "fuse: true\n"
        }
        
        // App-specific options (not part of mlx-lm-lora, used by MLX Training Studio only)
        // Post-training quantization
        if config.enablePostTrainingQuantization {
            yaml += "\n# MLX Training Studio options (not used by mlx-lm-lora directly)\n"
            yaml += "enable_post_training_quantization: true\n"
            if let postQuant = config.postTrainingQuantization {
                switch postQuant {
                case .bits4:
                    yaml += "post_training_quantization: 4bits\n"
                case .bits6:
                    yaml += "post_training_quantization: 6bits\n"
                case .bits8:
                    yaml += "post_training_quantization: 8bits\n"
                case .none:
                    break
                }
            }
            if let quantizedPath = config.quantizedModelPath {
                yaml += "quantized_model_path: \(quoteString(quantizedPath))\n"
            }
        }
        
        // GGUF conversion
        if config.enableGGUFConversion {
            if !config.enablePostTrainingQuantization {
                yaml += "\n# MLX Training Studio options (not used by mlx-lm-lora directly)\n"
            }
            yaml += "enable_gguf_conversion: true\n"
            if let ggufPath = config.ggufOutputPath {
                yaml += "gguf_output_path: \(quoteString(ggufPath))\n"
            }
            yaml += "gguf_outtype: \(quoteString(config.ggufOutType.rawValue))\n"
        }
        
        // Additional CLI arguments
        if !config.additionalArgs.isEmpty {
            let argsString = config.additionalArgs.map { quoteString($0) }.joined(separator: ", ")
            yaml += "additional_args: [\(argsString)]\n"
        }
        
        return yaml
    }
    
    // MARK: - YAML to Configuration
    
    func yamlToConfiguration(_ yamlString: String) throws -> TrainingConfiguration {
        // This is a simplified parser - for production, consider using a proper YAML library
        var config = TrainingConfiguration()
        let lines = yamlString.components(separatedBy: .newlines)
        var inLoraParameters = false
        var inLRSchedule = false
        var inOptimizerConfig = false
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") {
                continue
            }
            
            // Check if we're entering/exiting lora_parameters section
            if trimmed == "lora_parameters:" {
                inLoraParameters = true
                inLRSchedule = false
                inOptimizerConfig = false
                if config.loraParameters == nil {
                    config.loraParameters = LoRAParameters()
                }
                continue
            }
            
            // Check if we're entering/exiting lr_schedule section
            if trimmed == "lr_schedule:" {
                inLRSchedule = true
                inLoraParameters = false
                inOptimizerConfig = false
                if config.lrScheduleParams == nil {
                    config.lrScheduleParams = LRScheduleParameters()
                }
                continue
            }
            
            // Check if we're entering optimizer_config section
            if trimmed == "optimizer_config:" {
                inOptimizerConfig = true
                inLoraParameters = false
                inLRSchedule = false
                continue
            }
            
            // Check if we're exiting nested sections (next top-level key)
            if (inLoraParameters || inLRSchedule || inOptimizerConfig) && !line.hasPrefix("  ") && !line.hasPrefix("\t") {
                inLoraParameters = false
                inLRSchedule = false
                inOptimizerConfig = false
            }
            
            let parts = trimmed.split(separator: ":", maxSplits: 1)
            guard parts.count == 2 else { continue }
            
            let key = parts[0].trimmingCharacters(in: .whitespaces)
            var value = parts[1].trimmingCharacters(in: .whitespaces)
            
            // Handle optimizer_config nested keys (weight_decay is inside optimizer name)
            if inOptimizerConfig {
                switch key {
                case "weight_decay":
                    config.weightDecay = Double(value)
                default:
                    break // Skip optimizer name keys like "adamw:", other config params
                }
                continue
            }
            
            // Handle lora_parameters nested keys
            if inLoraParameters {
                if config.loraParameters == nil {
                    config.loraParameters = LoRAParameters()
                }
                
                switch key {
                case "keys":
                    // Parse array format: ["key1", "key2"] or [key1, key2]
                    var keysString = value.trimmingCharacters(in: .whitespaces)
                    // Remove brackets
                    if keysString.hasPrefix("[") {
                        keysString.removeFirst()
                    }
                    if keysString.hasSuffix("]") {
                        keysString.removeLast()
                    }
                    // Split by comma and unquote
                    let keys = keysString.split(separator: ",").map { key in
                        unquoteString(key.trimmingCharacters(in: .whitespaces))
                    }.filter { !$0.isEmpty }
                    if !keys.isEmpty {
                        config.loraParameters?.keys = keys
                    }
                case "rank":
                    if let rank = Int(value) {
                        config.loraParameters?.rank = rank
                        // Update alpha to 2x rank if not explicitly set
                        if config.loraParameters?.alpha == 16.0 && rank != 8 {
                            config.loraParameters?.alpha = Double(rank * 2)
                        }
                    }
                case "alpha":
                    if let alpha = Double(value) {
                        config.loraParameters?.alpha = alpha
                    }
                case "dropout":
                    if let dropout = Double(value) {
                        config.loraParameters?.dropout = dropout
                    }
                case "scale":
                    if let scale = Double(value) {
                        config.loraParameters?.scale = scale
                    }
                default:
                    break
                }
                continue
            }
            
            // Handle lr_schedule nested keys
            if inLRSchedule {
                if config.lrScheduleParams == nil {
                    config.lrScheduleParams = LRScheduleParameters()
                }
                
                switch key {
                case "name":
                    let unquotedValue = unquoteString(value)
                    // Map YAML names back to enum values
                    var scheduleType: LRScheduleType = .cosine
                    switch unquotedValue {
                    case "cosine_decay":
                        scheduleType = .cosine
                    case "linear_decay", "linear_schedule":  // linear_decay doesn't exist, use linear_schedule
                        scheduleType = .linear
                    case "constant":
                        scheduleType = .constant
                    default:
                        // Try direct match
                        if let direct = LRScheduleType(rawValue: unquotedValue) {
                            scheduleType = direct
                        }
                    }
                    config.lrScheduleParams?.name = scheduleType
                case "warmup":
                    config.lrScheduleParams?.warmup = Int(value)
                case "warmup_init":
                    config.lrScheduleParams?.warmupInit = Double(value)
                // Note: decay_steps and final_lr are not mlx-lm-lora parameters - they're internal helpers
                // We parse them for backwards compatibility and manual YAML editing, but they're used to construct arguments
                case "decay_steps":
                    config.lrScheduleParams?.decaySteps = Int(value)
                case "final_lr", "end_lr":
                    config.lrScheduleParams?.finalLR = Double(value)
                case "arguments":
                    // Parse array format: [1e-5, 1000, 1e-7]
                    var argsString = value.trimmingCharacters(in: .whitespaces)
                    // Remove brackets
                    if argsString.hasPrefix("[") {
                        argsString.removeFirst()
                    }
                    if argsString.hasSuffix("]") {
                        argsString.removeLast()
                    }
                    // Split by comma and parse as doubles
                    let arguments = argsString.split(separator: ",").compactMap { arg in
                        Double(arg.trimmingCharacters(in: .whitespaces))
                    }
                    if !arguments.isEmpty {
                        config.lrScheduleParams?.arguments = arguments
                        
                        // Extract decaySteps and finalLR from arguments for UI display
                        // This preserves the user's settings when loading YAML files
                        if let scheduleName = config.lrScheduleParams?.name {
                            switch scheduleName {
                            case .cosine:
                                // cosine_decay: [init, decay_steps, end]
                                if arguments.count >= 3 {
                                    config.lrScheduleParams?.decaySteps = Int(arguments[1])
                                    config.lrScheduleParams?.finalLR = arguments[2]
                                }
                            case .linear:
                                // linear_schedule: [init, end, steps]
                                if arguments.count >= 3 {
                                    config.lrScheduleParams?.finalLR = arguments[1]
                                    config.lrScheduleParams?.decaySteps = Int(arguments[2])
                                }
                            case .constant:
                                // constant: [init] - no decay steps or final LR
                                break
                            }
                        }
                    }
                default:
                    break
                }
                continue
            }
            
            // Unquote string values
            value = unquoteString(value)
            
            switch key {
            case "model":
                config.model = value
            case "data":
                config.data = value
            case "train_type":
                // train_type is "lora", "dora", or "full"
                if let trainType = TrainType(rawValue: value) {
                    config.trainType = trainType
                    if trainType == .lora && config.loraParameters == nil {
                        config.loraParameters = LoRAParameters()
                    }
                }
            case "mask_prompt":
                config.maskPrompt = value.lowercased() == "true"
            case "reference_model_path":
                config.referenceModelPath = value.isEmpty ? nil : value
            case "train_mode":
                if let mode = TrainingMode(rawValue: value) {
                    config.trainMode = mode
                }
            case "adapter_path":
                config.adapterPath = value
            case "batch_size":
                config.batchSize = Int(value) ?? config.batchSize
            case "learning_rate":
                config.learningRate = Double(value) ?? config.learningRate
            case "iters":
                config.iterations = Int(value) ?? config.iterations
            case "max_seq_length":
                config.maxSeqLength = Int(value)
            case "gradient_accumulation_steps":
                config.gradientAccumulationSteps = Int(value)
            case "grad_checkpoint":
                config.gradCheckpoint = value.lowercased() == "true"
            case "seed":
                config.seed = Int(value) ?? 0
            case "val_batches":
                config.valBatches = Int(value)
            case "steps_per_report":
                config.stepsPerReport = Int(value)
            case "steps_per_eval":
                config.stepsPerEval = Int(value)
            case "save_every":
                config.saveEvery = Int(value)
            case "resume_adapter_file":
                let unquoted = unquoteString(value)
                config.resumeAdapterFile = unquoted == "null" ? nil : unquoted
            case "num_layers":
                config.numLayers = Int(value)
            // Note: target_modules should be in lora_parameters.keys, not at top level
            case "optimizer":
                if let opt = OptimizerType(rawValue: value) {
                    config.optimizer = opt
                }
            case "lr_schedule":
                // Handle both old format (simple string) and new format (nested structure)
                // If it's a simple string value, parse it directly
                let unquotedValue = unquoteString(value)
                if let schedule = LRScheduleType(rawValue: unquotedValue) {
                    if config.lrScheduleParams == nil {
                        config.lrScheduleParams = LRScheduleParameters()
                    }
                    config.lrScheduleParams?.name = schedule
                }
            // Note: warmup_steps should be in lr_schedule.warmup, not at top level
            // Note: weight_decay should be in optimizer_config.<optimizer>.weight_decay
            case "beta":
                // Beta can be for DPO, ORPO, XPO, or RLHF Reinforce
                if config.trainMode == .orpo {
                    if config.orpoParams == nil {
                        config.orpoParams = ORPOParameters()
                    }
                    config.orpoParams?.beta = Double(value) ?? 0.1
                } else if config.trainMode == .xpo || config.trainMode == .rlhfReinforce {
                    if config.onlineParams == nil {
                        config.onlineParams = OnlineParameters()
                    }
                    config.onlineParams?.beta = Double(value)
                } else if config.trainMode == .dpo || config.trainMode == .cpo {
                    if config.dpoParams == nil {
                        config.dpoParams = DPOParameters()
                    }
                    config.dpoParams?.beta = Double(value) ?? 0.1
                }
            case "dpo_cpo_loss_type":
                if config.dpoParams == nil {
                    config.dpoParams = DPOParameters()
                }
                config.dpoParams?.lossType = value
            case "delta":
                if config.dpoParams == nil {
                    config.dpoParams = DPOParameters()
                }
                config.dpoParams?.delta = Double(value)
            case "reward_scaling":
                if config.orpoParams == nil {
                    config.orpoParams = ORPOParameters()
                }
                config.orpoParams?.rewardScaling = Double(value) ?? 1.0
            case "group_size":
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.groupSize = Int(value) ?? 4
            case "temperature":
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.temperature = Double(value) ?? 0.8
            case "max_completion_length":
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.maxCompletionLength = Int(value) ?? 512
            case "epsilon":
                // Handle epsilon for both GRPO and Online DPO
                if config.trainMode.supportsGroupSize {
                    if config.grpoParams == nil {
                        config.grpoParams = GRPOParameters()
                    }
                    config.grpoParams?.epsilon = Double(value)
                } else if config.trainMode == .onlineDpo {
                    if config.onlineParams == nil {
                        config.onlineParams = OnlineParameters()
                    }
                    config.onlineParams?.epsilon = Double(value)
                }
            case "reward_functions_file":
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.rewardFunctionsFile = value.isEmpty ? nil : value
            case "reward_functions":
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                // Parse comma-separated reward function names
                let functions = value.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                config.grpoParams?.rewardFunctions = functions
            case "reward_weights":
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                // Parse JSON array format: [0.7, 0.3] or 0.7, 0.3
                var weightsString = value.trimmingCharacters(in: .whitespaces)
                if weightsString.hasPrefix("[") {
                    weightsString.removeFirst()
                }
                if weightsString.hasSuffix("]") {
                    weightsString.removeLast()
                }
                let weights = weightsString.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
                config.grpoParams?.rewardWeights = weights
            case "grpo_loss_type":
                if config.grpoParams == nil {
                    config.grpoParams = GRPOParameters()
                }
                config.grpoParams?.grpoLossType = value.isEmpty ? nil : value
            case "judge":
                if config.onlineParams == nil {
                    config.onlineParams = OnlineParameters()
                }
                config.onlineParams?.judgeModel = value
            case "alpha":
                if config.onlineParams == nil {
                    config.onlineParams = OnlineParameters()
                }
                config.onlineParams?.alpha = Double(value) ?? 1e-5
            case "judge_config":
                if config.onlineParams == nil {
                    config.onlineParams = OnlineParameters()
                }
                config.onlineParams?.judgeConfig = value.isEmpty ? nil : value
            case "wand":
                config.wandbProject = value
            case "load_in_4bits":
                if value.lowercased() == "true" {
                    config.quantization = .bits4
                }
            case "load_in_6bits":
                if value.lowercased() == "true" {
                    config.quantization = .bits6
                }
            case "load_in_8bits":
                if value.lowercased() == "true" {
                    config.quantization = .bits8
                }
            case "test":
                config.testMode = value.lowercased() == "true"
            case "test_batches":
                config.testBatches = Int(value)
            case "fuse":
                config.fuse = value.lowercased() == "true"
            case "enable_post_training_quantization":
                config.enablePostTrainingQuantization = value.lowercased() == "true"
            case "post_training_quantization":
                if value.lowercased() == "4bits" || value == "4" {
                    config.postTrainingQuantization = .bits4
                } else if value.lowercased() == "6bits" || value == "6" {
                    config.postTrainingQuantization = .bits6
                } else if value.lowercased() == "8bits" || value == "8" {
                    config.postTrainingQuantization = .bits8
                }
            case "quantized_model_path":
                config.quantizedModelPath = value.isEmpty ? nil : value
            case "enable_gguf_conversion":
                config.enableGGUFConversion = value.lowercased() == "true"
            case "gguf_output_path":
                config.ggufOutputPath = value.isEmpty ? nil : value
            case "gguf_outtype":
                let unquoted = unquoteString(value)
                if let outType = GGUFOutType(rawValue: unquoted) {
                    config.ggufOutType = outType
                }
            default:
                // Handle advanced additional CLI arguments
                if key == "additional_args" {
                    var argsString = value.trimmingCharacters(in: .whitespaces)
                    // Remove brackets
                    if argsString.hasPrefix("[") {
                        argsString.removeFirst()
                    }
                    if argsString.hasSuffix("]") {
                        argsString.removeLast()
                    }
                    // Split by comma and unquote each argument
                    let parts = argsString.split(separator: ",").map { part in
                        unquoteString(String(part.trimmingCharacters(in: .whitespaces)))
                    }
                    let filtered = parts.filter { !$0.isEmpty }
                    if !filtered.isEmpty {
                        config.additionalArgs = filtered
                    }
                    continue
                }
                
                // Handle nested LoRA parameters and other complex cases
                if key == "rank" && config.loraParameters == nil {
                    config.loraParameters = LoRAParameters()
                }
                if key == "rank" {
                    config.loraParameters?.rank = Int(value) ?? 8
                }
                if key == "dropout" {
                    if config.loraParameters == nil {
                        config.loraParameters = LoRAParameters()
                    }
                    config.loraParameters?.dropout = Double(value) ?? 0.0
                }
                if key == "scale" {
                    if config.loraParameters == nil {
                        config.loraParameters = LoRAParameters()
                    }
                    config.loraParameters?.scale = Double(value) ?? 10.0
                }
                break
            }
        }
        
        return config
    }
    
    // MARK: - File Operations
    
    func saveConfiguration(_ config: TrainingConfiguration, to url: URL) throws {
        let yaml = configurationToYAML(config)
        
        // Ensure the YAML is not empty
        guard !yaml.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw NSError(domain: "YAMLManager", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Cannot save empty configuration"
            ])
        }
        
        try yaml.write(to: url, atomically: true, encoding: .utf8)
    }
    
    func loadConfiguration(from url: URL) throws -> TrainingConfiguration {
        let yamlString = try String(contentsOf: url, encoding: .utf8)
        return try yamlToConfiguration(yamlString)
    }
}


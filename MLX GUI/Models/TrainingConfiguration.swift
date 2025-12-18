//
//  TrainingConfiguration.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

enum OptimizerType: String, Codable, CaseIterable {
    case adam = "adam"
    case adamw = "adamw"
    case qhadam = "qhadam"
    case muon = "muon"
    
    var displayName: String {
        return rawValue.capitalized
    }
}

enum LRScheduleType: String, Codable, CaseIterable {
    case cosine = "cosine"
    case linear = "linear"
    case constant = "constant"
    
    var displayName: String {
        switch self {
        case .cosine: return "Cosine Annealing"
        case .linear: return "Linear Decay"
        case .constant: return "Constant Rate"
        }
    }
    
    // Convert to YAML format name
    // Note: mlx-lm-lora uses getattr(opt.schedulers, name), so we need actual scheduler names
    var yamlName: String {
        switch self {
        case .cosine: return "cosine_decay"  // cosine_decay(init, decay_steps, end=0.0)
        case .linear: return "linear_schedule"  // linear_schedule(init, end, steps) - note: linear_decay doesn't exist
        case .constant: return "constant"  // May need to check if this exists or use a different approach
        }
    }
}

struct LRScheduleParameters: Codable, Equatable {
    var name: LRScheduleType = .cosine
    var warmup: Int? = 100  // Default: 100 warmup steps
    var warmupInit: Double? = 1e-7  // Default: 1e-7 initial learning rate for warmup
    var arguments: [Double]? = nil  // Optional arguments array for scheduler
    
    init(name: LRScheduleType = .cosine, warmup: Int? = 100, warmupInit: Double? = 1e-7, arguments: [Double]? = nil) {
        self.name = name
        self.warmup = warmup
        self.warmupInit = warmupInit
        self.arguments = arguments
    }
}

enum QuantizationType: String, Codable {
    case none = "none"
    case bits4 = "4bits"
    case bits8 = "8bits"
    
    var displayName: String {
        switch self {
        case .none: return "None"
        case .bits4: return "4-bit"
        case .bits8: return "8-bit"
        }
    }
}

enum GGUFOutType: String, Codable, CaseIterable {
    case auto = "auto"
    case f32 = "f32"
    case f16 = "f16"
    case bf16 = "bf16"
    case q8_0 = "q8_0"
    case tq1_0 = "tq1_0"
    case tq2_0 = "tq2_0"
    
    var displayName: String {
        switch self {
        case .auto:
            return "Auto (match model dtype)"
        case .f32:
            return "FP32 (f32)"
        case .f16:
            return "FP16 (f16)"
        case .bf16:
            return "BF16 (bf16)"
        case .q8_0:
            return "Q8_0 (8-bit)"
        case .tq1_0:
            return "TQ1_0 (tiered quant, smaller than Q8_0)"
        case .tq2_0:
            return "TQ2_0 (tiered quant, more aggressive)"
        }
    }
}

struct TrainingConfiguration: Codable, Equatable {
    // MARK: - Basic Settings
    var model: String = ""
    var data: String = ""
    var trainMode: TrainingMode = .sft
    var adapterPath: String? = nil
    
    // MARK: - Training Parameters
    var batchSize: Int = 4
    var learningRate: Double = 1e-5
    var iterations: Int = 1000
    var maxSeqLength: Int? = 2048
    var gradientAccumulationSteps: Int? = 1
    var gradCheckpoint: Bool = false
    var seed: Int = 0  // PRNG seed for reproducibility
    
    // MARK: - Validation & Reporting
    var valBatches: Int? = -1  // Number of validation batches, -1 uses entire set
    var stepsPerReport: Int? = 10  // Steps between loss reporting
    var stepsPerEval: Int? = 200  // Steps between validations
    var saveEvery: Int? = 100  // Save model every N iterations
    
    // MARK: - Resume Training
    var resumeAdapterFile: String? = nil  // Path to adapter weights to resume from
    
    // MARK: - LoRA Configuration
    var loraParameters: LoRAParameters? = LoRAParameters()
    var numLayers: Int? = -1  // -1 means all layers (mlx-lm-lora default)
    
    // MARK: - Optimizer & Schedule
    var optimizer: OptimizerType = .adamw  // Default: AdamW (supports weight decay)
    var lrScheduleParams: LRScheduleParameters? = LRScheduleParameters()  // Detailed LR schedule parameters
    var weightDecay: Double? = 0.01  // Default: 0.01 weight decay (common for regularization)
    
    // Convenience accessors for backward compatibility and UI
    var lrSchedule: LRScheduleType {
        get { lrScheduleParams?.name ?? .cosine }
        set {
            if lrScheduleParams == nil {
                lrScheduleParams = LRScheduleParameters()
            }
            lrScheduleParams?.name = newValue
        }
    }
    
    var warmupSteps: Int? {
        get { lrScheduleParams?.warmup }
        set {
            if lrScheduleParams == nil {
                lrScheduleParams = LRScheduleParameters()
            }
            lrScheduleParams?.warmup = newValue
        }
    }
    
    // MARK: - Mode-Specific Parameters
    var dpoParams: DPOParameters? = nil
    var grpoParams: GRPOParameters? = nil
    var onlineParams: OnlineParameters? = nil
    
    // MARK: - Advanced Options
    var quantization: QuantizationType? = nil
    var wandbProject: String? = nil
    var testMode: Bool = false
    var testBatches: Int? = nil
    var fuse: Bool = false
    var additionalArgs: [String] = []
    
    // MARK: - Post-Training Quantization
    var enablePostTrainingQuantization: Bool = false
    var postTrainingQuantization: QuantizationType? = .bits4  // Default to 4-bit
    var quantizedModelPath: String? = nil  // If nil, defaults to adapterPath/quantized
    
    // MARK: - GGUF Conversion
    var enableGGUFConversion: Bool = false
    var ggufOutputPath: String? = nil  // If nil, defaults to adapterPath/gguf or quantizedModelPath/gguf
    var ggufOutType: GGUFOutType = .auto  // Passed to convert_hf_to_gguf.py --outtype
    
    // MARK: - Initialization
    init() {
        // Default initialization
    }
    
    // MARK: - Validation
    func validate() -> [String] {
        var errors: [String] = []
        
        if model.isEmpty {
            errors.append("Model path is required")
        }
        
        if data.isEmpty {
            errors.append("Data path is required")
        }
        
        if batchSize <= 0 {
            errors.append("Batch size must be greater than 0")
        }
        
        if learningRate <= 0 {
            errors.append("Learning rate must be greater than 0")
        }
        
        if iterations <= 0 {
            errors.append("Iterations must be greater than 0")
        }
        
        // Mode-specific validation
        if trainMode.requiresJudgeModel {
            if let onlineParams = onlineParams, onlineParams.judgeModel.isEmpty {
                errors.append("Judge model is required for \(trainMode.displayName)")
            }
        }
        
        if trainMode.requiresReferenceModel && adapterPath == nil {
            // Note: Reference model is typically the base model, but adapter path might be needed
            // This is a simplified check
        }
        
        return errors
    }
}


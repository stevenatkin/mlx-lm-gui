//
//  CommandBuilder.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

class CommandBuilder {
    static func buildCommand(from config: TrainingConfiguration, pythonPath: String, configFilePath: String? = nil) -> (executable: String, arguments: [String]) {
        var args: [String] = ["-m", "mlx_lm_lora.train"]
        
        // Use config file if provided (preferred method for complex parameters)
        if let configFile = configFilePath {
            args.append("-c")
            args.append(configFile)
            // When using config file, we can still override with command-line args if needed
        }
        
        // Basic arguments (these can override config file values)
        args.append("--model")
        args.append(config.model)
        
        args.append("--data")
        args.append(config.data)
        
        args.append("--train-mode")
        args.append(config.trainMode.rawValue)
        
        if let adapterPath = config.adapterPath {
            args.append("--adapter-path")
            args.append(adapterPath)
        }
        
        // Training parameters
        args.append("--batch-size")
        args.append(String(config.batchSize))
        
        args.append("--learning-rate")
        args.append(String(config.learningRate))
        
        args.append("--iters")
        args.append(String(config.iterations))
        
        if let maxSeqLength = config.maxSeqLength {
            args.append("--max-seq-length")
            args.append(String(maxSeqLength))
        }
        
        if let gradAccum = config.gradientAccumulationSteps {
            args.append("--gradient-accumulation-steps")
            args.append(String(gradAccum))
        }
        
        if config.gradCheckpoint {
            args.append("--grad-checkpoint")
        }
        
        args.append("--seed")
        args.append(String(config.seed))
        
        // LoRA parameters, LR schedule, and weight decay are in the config file
        // Only include command-line args that are actually supported
        
        if let numLayers = config.numLayers {
            args.append("--num-layers")
            args.append(String(numLayers))
        }
        
        // Optimizer
        args.append("--optimizer")
        args.append(config.optimizer.rawValue)
        
        // Note: --lr-schedule, --weight-decay, and --warmup-steps are not in the help output
        // These should be specified in the config file only
        
        // Mode-specific parameters
        if let dpoParams = config.dpoParams {
            args.append("--beta")
            args.append(String(dpoParams.beta))
            
            args.append("--dpo-cpo-loss-type")
            args.append(dpoParams.lossType)
        }
        
        if let grpoParams = config.grpoParams {
            args.append("--group-size")
            args.append(String(grpoParams.groupSize))
            
            args.append("--temperature")
            args.append(String(grpoParams.temperature))
            
            args.append("--max-completion-length")
            args.append(String(grpoParams.maxCompletionLength))
            
            if !grpoParams.rewardFunctions.isEmpty {
                args.append("--reward-functions")
                args.append(grpoParams.rewardFunctions.joined(separator: ","))
            }
            
            if !grpoParams.rewardWeights.isEmpty {
                args.append("--reward-weights")
                let weightsString = grpoParams.rewardWeights.map { String($0) }.joined(separator: ",")
                args.append("[\(weightsString)]")
            }
            
            if let importanceSampling = grpoParams.importanceSamplingLevel {
                args.append("--importance-sampling-level")
                args.append(importanceSampling)
            }
            
            if let lossType = grpoParams.grpoLossType {
                args.append("--grpo-loss-type")
                args.append(lossType)
            }
            
            if let epsilonLow = grpoParams.epsilonLow {
                args.append("--epsilon")
                args.append(String(epsilonLow))
            }
            
            if let epsilonHigh = grpoParams.epsilonHigh {
                args.append("--epsilon-high")
                args.append(String(epsilonHigh))
            }
        }
        
        if let onlineParams = config.onlineParams {
            if !onlineParams.judgeModel.isEmpty {
                args.append("--judge")
                args.append(onlineParams.judgeModel)
            }
            
            args.append("--alpha")
            args.append(String(onlineParams.alpha))
            
            if let epsilon = onlineParams.epsilon {
                args.append("--epsilon")
                args.append(String(epsilon))
            }
            
            if let groupSize = onlineParams.groupSize {
                args.append("--group-size")
                args.append(String(groupSize))
            }
        }
        
        // Advanced options
        if let quantization = config.quantization {
            switch quantization {
            case .bits4:
                args.append("--load-in-4bits")
            case .bits8:
                args.append("--load-in-8bits")
            case .none:
                break
            }
        }
        
        if let wandb = config.wandbProject {
            args.append("--wandb")
            args.append(wandb)
        }
        
        if config.testMode {
            args.append("--test")
            if let testBatches = config.testBatches {
                args.append("--test-batches")
                args.append(String(testBatches))
            }
        }
        
        if config.fuse {
            args.append("--fuse")
        }
        
        // Additional arguments
        args.append(contentsOf: config.additionalArgs)
        
        return (executable: pythonPath, arguments: args)
    }
    
    static func commandString(from config: TrainingConfiguration, pythonPath: String, configFilePath: String? = nil) -> String {
        let (executable, arguments) = buildCommand(from: config, pythonPath: pythonPath, configFilePath: configFilePath)
        let argsString = arguments.map { arg in
            arg.contains(" ") ? "\"\(arg)\"" : arg
        }.joined(separator: " ")
        return "\(executable) \(argsString)"
    }
}


//
//  TrainingMode.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

enum TrainingMode: String, CaseIterable, Codable {
    case sft = "sft"
    case dpo = "dpo"
    case cpo = "cpo"
    case orpo = "orpo"
    case grpo = "grpo"
    case gspo = "gspo"
    case drGrpo = "dr_grpo"
    case dapo = "dapo"
    case onlineDpo = "online_dpo"
    case xpo = "xpo"
    case rlhfReinforce = "rlhf-reinforce"
    case ppo = "ppo"
    
    var displayName: String {
        switch self {
        case .sft: return "SFT (Supervised Fine-Tuning)"
        case .dpo: return "DPO (Direct Preference Optimization)"
        case .cpo: return "CPO (Contrastive Preference Optimization)"
        case .orpo: return "ORPO (Odds Ratio Preference Optimization)"
        case .grpo: return "GRPO (Group Relative Policy Optimization)"
        case .gspo: return "GSPO (GRPO with Importance Sampling)"
        case .drGrpo: return "Dr. GRPO (Decoupled Rewards)"
        case .dapo: return "DAPO (Dynamic Adaptive Policy Optimization)"
        case .onlineDpo: return "Online DPO"
        case .xpo: return "XPO (Extended Preferences)"
        case .rlhfReinforce: return "RLHF Reinforce"
        case .ppo: return "PPO (Proximal Policy Optimization)"
        }
    }
    
    var description: String {
        switch self {
        case .sft: return "Simple, fast supervised fine-tuning. No reference or judge model needed."
        case .dpo: return "Direct preference optimization. Requires reference model."
        case .cpo: return "Contrastive preference optimization. Better for structured tasks. Requires reference model."
        case .orpo: return "Odds ratio preference optimization. Monolithic optimization approach."
        case .grpo: return "Group relative policy optimization. Group-based learning with multiple generations."
        case .gspo: return "GRPO with importance sampling. More efficient group-based training."
        case .drGrpo: return "Decoupled rewards GRPO. Separates reward computation."
        case .dapo: return "Dynamic adaptive policy optimization. Adaptive clipping with epsilon ranges."
        case .onlineDpo: return "Online DPO with real-time feedback. Requires judge model."
        case .xpo: return "Extended preferences. Requires judge model."
        case .rlhfReinforce: return "RLHF with REINFORCE algorithm. Requires reward model."
        case .ppo: return "Proximal Policy Optimization. Full RL pipeline. Requires reward model."
        }
    }
    
    var requiresReferenceModel: Bool {
        switch self {
        case .dpo, .cpo, .onlineDpo, .xpo, .rlhfReinforce, .ppo:
            return true
        default:
            return false
        }
    }
    
    var requiresJudgeModel: Bool {
        switch self {
        case .onlineDpo, .xpo, .rlhfReinforce, .ppo:
            return true
        default:
            return false
        }
    }
    
    var supportsGroupSize: Bool {
        switch self {
        case .grpo, .gspo, .drGrpo, .dapo, .rlhfReinforce, .ppo:
            return true
        default:
            return false
        }
    }
    
    var supportsRewardFunctions: Bool {
        switch self {
        case .grpo, .gspo, .drGrpo, .dapo:
            return true
        default:
            return false
        }
    }
}


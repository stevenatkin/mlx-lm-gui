//
//  OnlineParameters.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

struct OnlineParameters: Codable, Equatable {
    var judgeModel: String = ""
    var alpha: Double = 1e-5
    var beta: Double? = nil  // For XPO and RLHF Reinforce (KL penalty strength)
    var epsilon: Double? = nil // for PPO (default: 0.2)
    var groupSize: Int? = nil // for RLHF Reinforce and PPO
    var judgeConfig: String? = nil  // Additional configuration for judge model
    
    init(judgeModel: String = "", alpha: Double = 1e-5) {
        self.judgeModel = judgeModel
        self.alpha = alpha
    }
}


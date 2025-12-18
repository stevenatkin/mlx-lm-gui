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
    var epsilon: Double? = nil // for PPO
    var groupSize: Int? = nil // for RLHF Reinforce and PPO
    
    init(judgeModel: String = "", alpha: Double = 1e-5) {
        self.judgeModel = judgeModel
        self.alpha = alpha
    }
}


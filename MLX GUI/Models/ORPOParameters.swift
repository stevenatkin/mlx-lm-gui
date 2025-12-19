//
//  ORPOParameters.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

struct ORPOParameters: Codable, Equatable {
    var beta: Double = 0.1
    var rewardScaling: Double = 1.0
    
    init(beta: Double = 0.1, rewardScaling: Double = 1.0) {
        self.beta = beta
        self.rewardScaling = rewardScaling
    }
}


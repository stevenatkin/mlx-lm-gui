//
//  GRPOParameters.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

struct GRPOParameters: Codable, Equatable {
    var groupSize: Int = 4
    var temperature: Double = 0.8
    var maxCompletionLength: Int = 512
    var rewardFunctions: [String] = []
    var rewardWeights: [Double] = []
    var importanceSamplingLevel: String? = nil // token, sequence, none - for GSPO
    var grpoLossType: String? = nil // for Dr. GRPO
    var epsilonLow: Double? = nil // for DAPO
    var epsilonHigh: Double? = nil // for DAPO
    
    init(groupSize: Int = 4, temperature: Double = 0.8, maxCompletionLength: Int = 512) {
        self.groupSize = groupSize
        self.temperature = temperature
        self.maxCompletionLength = maxCompletionLength
    }
}


//
//  DPOParameters.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

struct DPOParameters: Codable, Equatable {
    var beta: Double = 0.1
    var lossType: String = "sigmoid" // sigmoid, hinge, ipo, dpop
    var delta: Double? = nil  // Margin for hinge loss (default: 50.0)
    
    init(beta: Double = 0.1, lossType: String = "sigmoid", delta: Double? = nil) {
        self.beta = beta
        self.lossType = lossType
        self.delta = delta
    }
}


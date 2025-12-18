//
//  DPOParameters.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

struct DPOParameters: Codable, Equatable {
    var beta: Double = 0.1
    var lossType: String = "sigmoid" // sigmoid, exponential, etc.
    
    init(beta: Double = 0.1, lossType: String = "sigmoid") {
        self.beta = beta
        self.lossType = lossType
    }
}


//
//  LoRAParameters.swift
//  MLX GUI
//
//  Created by Steven Atkin on 12/13/25.
//

import Foundation

struct LoRAParameters: Codable, Equatable {
    var rank: Int = 8
    var alpha: Double = 16.0  // Default: 2x rank (8 * 2 = 16)
    var dropout: Double = 0.0
    var scale: Double = 10.0
    // Default keys include attention and MLP layers for comprehensive LoRA coverage
    var keys: [String] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"]
    
    init(rank: Int = 8, alpha: Double? = nil, dropout: Double = 0.0, scale: Double = 10.0, keys: [String] = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"]) {
        self.rank = rank
        self.alpha = alpha ?? Double(rank * 2)  // Default to 2x rank if not specified
        self.dropout = dropout
        self.scale = scale
        self.keys = keys
    }
    
    func toDictionary() -> [String: Any] {
        return [
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "scale": scale,
            "keys": keys
        ]
    }
}


#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace miia::runtime {

struct PredictionResult {
    std::unordered_map<std::string, std::vector<float>> outputs;
    std::string error_message;
};

struct PredictionRequest {
    std::string type;
    std::string model_id;
    std::unordered_map<std::string, std::vector<float>> inputs;
};

}
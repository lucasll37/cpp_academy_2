#pragma once

#include "miia/runtime/inference_orchestrator.hpp"
#include <string>
#include <map>
#include <vector>

namespace miia::runtime {

struct PredictionResult {
    std::unordered_map<std::string, std::vector<float>> outputs;
    std::string error_message;
};

struct PredictionRequest {
    std::string type; //numerical (rede neural)
    std::string model_id; //ppooptimizationflight1
    std::unordered_map<std::string, std::vector<float>> inputs;
};

class InferenceProvider {
public:
    InferenceProvider();
    ~InferenceProvider();

    PredictionResult predict(const PredictionRequest& request);

private:
    InferenceOrchestrator orchestrator_;
};
#pragma once

#include "miia/runtime/prediction_contract.hpp"

namespace miia::runtime {

class IRuntime {
public:
    virtual ~IRuntime() = default;

    virtual PredictionResult run(const std::unordered_map<std::string, std::vector<float>>& inputs) = 0;
};

}
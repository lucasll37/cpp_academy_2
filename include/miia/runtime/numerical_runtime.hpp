#pragma once

#include "miia/runtime/iruntime.hpp"

namespace miia::runtime {

class NumericalRuntime : public IRuntime {
public:
    PredictionResult run(const std::unordered_map<std::string, std::vector<float>>& inputs) override;
};

}
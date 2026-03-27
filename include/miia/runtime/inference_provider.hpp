#pragma once

#include "miia/runtime/inference_orchestrator.hpp"
#include <string>

namespace miia::runtime {

class InferenceProvider {
public:
    // Public API: user-facing entry point
    std::string infer(const std::string& type,
                      const std::string& input);

private:
    InferenceOrchestrator orchestrator_;
};

}
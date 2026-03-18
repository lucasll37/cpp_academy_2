#pragma once

#include "miia/runtime/inference_orchestrator.hpp"
#include "miia/runtime/iruntime.hpp"
#include <memory>

namespace miia::runtime {

class InferenceProvider {
public:
    void set_runtime(std::shared_ptr<IRuntime> runtime);
    std::string infer(const std::string& input);

private:
    InferenceOrchestrator orchestrator_;
};

}
#include "miia/runtime/inference_orchestrator.hpp"

namespace miia::runtime {

void InferenceOrchestrator::set_runtime(std::shared_ptr<IRuntime> runtime) {
    runtime_ = std::move(runtime);
}

std::string InferenceOrchestrator::execute(const std::string& input) {
    if (!runtime_) return "No runtime set";
    return runtime_->run(input);
}

}
#include "miia/runtime/inference_provider.hpp"

namespace miia::runtime {

void InferenceProvider::set_runtime(std::shared_ptr<IRuntime> runtime) {
    orchestrator_.set_runtime(std::move(runtime));
}

std::string InferenceProvider::infer(const std::string& input) {
    return orchestrator_.execute(input);
}

}
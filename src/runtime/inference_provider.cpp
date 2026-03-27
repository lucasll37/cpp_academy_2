#include "miia/runtime/inference_provider.hpp"

namespace miia::runtime {

std::string InferenceProvider::infer(
    const std::string& type,
    const std::string& input
) {
    // Minimal sanity checks — this is the "receptionist"
    if (type.empty()) {
        return "Invalid model type";
    }

    if (input.empty()) {
        return "Empty input";
    }

    // Delegate to the "technical boss"
    return orchestrator_.execute(type, input);
}

}
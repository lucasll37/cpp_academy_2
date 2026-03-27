#include "miia/runtime/inference_orchestrator.hpp"
#include "miia/runtime/numerical_runtime.hpp"
#include "miia/runtime/procedural_runtime.hpp"
#include "miia/runtime/symbolic_runtime.hpp"

#include <future>
#include <stdexcept>

namespace miia::runtime {

std::shared_ptr<InferenceOrchestrator::RuntimeEntry>
InferenceOrchestrator::get_or_create(const std::string& type) {
    {
        // First check (fast path)
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = runtimes_.find(type);
        if (it != runtimes_.end()) {
            return it->second;
        }
    }

    // Create outside lock? No. Keep it simple and safe.
    std::lock_guard<std::mutex> lock(map_mutex_);

    // Double-check (another thread may have created it)
    auto it = runtimes_.find(type);
    if (it != runtimes_.end()) {
        return it->second;
    }

    auto entry = std::make_shared<RuntimeEntry>();

    // Factory logic (yes, this is now your responsibility)
    if (type == "numerical") {
        entry->runtime = std::make_shared<NumericalRuntime>();
        entry->is_stateful = false;
    } else if (type == "symbolic") {
        entry->runtime = std::make_shared<SymbolicRuntime>();
        entry->is_stateful = true;
    } else if (type == "procedural") {
        entry->runtime = std::make_shared<ProceduralRuntime>();
        entry->is_stateful = true;
    } else {
        throw std::runtime_error("Unknown runtime type: " + type);
    }

    runtimes_[type] = entry;
    return entry;
}

std::string InferenceOrchestrator::execute(
    const std::string& type,
    const std::string& input
) {
    auto entry = get_or_create(type);

    // Stateless → full concurrency
    if (!entry->is_stateful) {
        auto future = std::async(std::launch::async, [entry, input]() {
            return entry->runtime->run(input);
        });
        return future.get();
    }

    // Stateful → serialized per type
    auto future = std::async(std::launch::async, [entry, input]() {
        std::lock_guard<std::mutex> lock(entry->mutex);
        return entry->runtime->run(input);
    });

    return future.get();
}

}
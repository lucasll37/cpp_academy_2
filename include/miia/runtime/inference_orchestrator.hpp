#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>

#include "miia/runtime/iruntime.hpp"

namespace miia::runtime {

class InferenceOrchestrator {
public:
    std::string execute(const std::string& type,
                        const std::string& input);

private:
    struct RuntimeEntry {
        std::shared_ptr<IRuntime> runtime;
        bool is_stateful;
        std::mutex mutex; // for stateful execution
    };

    std::shared_ptr<RuntimeEntry> get_or_create(const std::string& type);

    std::unordered_map<std::string, std::shared_ptr<RuntimeEntry>> runtimes_;
    std::mutex map_mutex_; // protects runtimes_ during creation
};

}
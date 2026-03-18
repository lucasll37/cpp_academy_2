#pragma once

#include <memory>
#include "miia/runtime/iruntime.hpp"

namespace miia::runtime {

class InferenceOrchestrator {
public:
    void set_runtime(std::shared_ptr<IRuntime> runtime);
    std::string execute(const std::string& input);

private:
    std::shared_ptr<IRuntime> runtime_;
};

}
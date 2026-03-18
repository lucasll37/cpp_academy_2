#pragma once

#include "miia/runtime/iruntime.hpp"

namespace miia::runtime {

class ProceduralRuntime : public IRuntime {
public:
    std::string run(const std::string& input) override;
};

}
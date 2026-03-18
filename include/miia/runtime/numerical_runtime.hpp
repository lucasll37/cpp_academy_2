#pragma once

#include "miia/runtime/iruntime.hpp"

namespace miia::runtime {

class NumericalRuntime : public IRuntime {
public:
    std::string run(const std::string& input) override;
};

}
#pragma once

#include <string>

namespace miia::runtime {

class IRuntime {
public:
    virtual ~IRuntime() = default;

    virtual std::string run(const std::string& input) = 0;
};

}
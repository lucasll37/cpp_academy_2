#include "miia/runtime/procedural_runtime.hpp"

namespace miia::runtime {

std::string ProceduralRuntime::run(const std::string& input) {
    return "Procedural: " + input;
}

}
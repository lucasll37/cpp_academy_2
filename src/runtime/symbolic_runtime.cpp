#include "miia/runtime/symbolic_runtime.hpp"

namespace miia::runtime {

std::string SymbolicRuntime::run(const std::string& input) {
    return "Symbolic: " + input;
}

}
#include <iostream>
#include "miia/runtime/engine.hpp"

namespace miia::runtime {

Engine::Engine() = default;

void Engine::run() { std::cout << "MIIA!" << std::endl; }

}
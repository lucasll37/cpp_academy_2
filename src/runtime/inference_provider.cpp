#include "miia/runtime/inference_provider.hpp"

namespace miia::runtime {

std::string InferenceProvider::infer(
    const std::string& type,
    const std::string& input
) {
    // Validação básica de entrada (responsabilidade da "fachada")
    if (type.empty()) {
        return "Tipo de modelo inválido";
    }

    if (input.empty()) {
        return "Entrada vazia";
    }

    // Delega a execução ao Orchestrator,
    // que cuidará da escolha do runtime e execução.
    return orchestrator_.execute(type, input);
}

}
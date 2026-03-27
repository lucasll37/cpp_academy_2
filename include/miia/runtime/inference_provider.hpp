#pragma once

#include "miia/runtime/inference_orchestrator.hpp"
#include <string>

namespace miia::runtime {

// Classe responsável por ser a "porta de entrada" do sistema.
// Atua como uma fachada (Facade), escondendo a complexidade interna
// do Orchestrator e dos runtimes.
class InferenceProvider {
public:
    // Método principal exposto ao usuário.
    // Recebe o tipo do modelo (ex: "numerical", "symbolic", etc)
    // e o input a ser processado.
    std::string infer(const std::string& type,
                      const std::string& input);

private:
    // Orchestrator responsável por decidir qual runtime usar
    // e como executar (concorrência, estado, etc).
    InferenceOrchestrator orchestrator_;
};

}
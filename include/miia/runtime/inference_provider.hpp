// Garante inclusão única do header durante a compilação (evita múltiplas definições)
#pragma once

#include "miia/runtime/prediction_contract.hpp"
#include "miia/runtime/inference_orchestrator.hpp" // Orquestrador responsável por executar a inferência
#include <string> // Uso de std::string
#include <map> // (Possivelmente não utilizado - pode ser removido se não houver uso)
#include <vector> // Uso de std::vector
#include <unordered_map>

namespace miia::runtime {

// Classe responsável por receber requisições e delegar execução ao orquestrador
class InferenceProvider {
public:
    // Construtor padrão
    InferenceProvider();
     // Destrutor padrão
    ~InferenceProvider();

    // Método principal que valida a requisição e executa a inferência
    PredictionResult predict(const PredictionRequest& request);

private:
    // Componente interno responsável por orquestrar a execução da inferência
    InferenceOrchestrator orchestrator_;
};

}
// Garante inclusão única do header durante a compilação (evita múltiplas definições)
#pragma once

#include "miia/runtime/inference_orchestrator.hpp" // Orquestrador responsável por executar a inferência
#include <string> // Uso de std::string
#include <map> // (Possivelmente não utilizado - pode ser removido se não houver uso)
#include <vector> // Uso de std::vector
#include <unordered_map>

namespace miia::runtime {

// Estrutura que representa o resultado da predição    
struct PredictionResult {
    // Saídas da inferência: chave (nome da saída) -> vetor de valores numéricos
    std::unordered_map<std::string, std::vector<float>> outputs;
    // Mensagem de erro (vazia em caso de sucesso)
    std::string error_message;
};

// Estrutura que representa a requisição de predição
struct PredictionRequest {
    std::string type; // Tipo do modelo (ex: "numerical", "symbolical", "procedural")
    std::string model_id; // Identificador único do modelo (ex: "ppooptimizationflight1")
    // Entradas da inferência: chave (nome da feature) -> vetor de valores
    std::unordered_map<std::string, std::vector<float>> inputs;
};

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
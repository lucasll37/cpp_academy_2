#include "miia/runtime/inference_provider.hpp"  // Interface principal do provider de inferência
#include <unordered_set> 
#include <chrono>

namespace miia::runtime {

// Construtor padrão da classe InferenceProvider    
InferenceProvider::InferenceProvider() = default;

// Destrutor padrão
InferenceProvider::~InferenceProvider() = default;

// Método principal responsável por executar uma predição
PredictionResult InferenceProvider::predict(const PredictionRequest& request) {
    PredictionResult result;

    // Valida se o tipo do modelo foi informado
    if (request.type.empty()) {
        result.error_message = "Tipo do modelo não especificado";
        return result;
    }

    // Valida se o tipo do modelo é um dos permitidos
    if (request.type != "numerical" && request.type != "symbolical" && request.type != "procedural") {
        result.error_message = "Tipo do modelo inválido";
        return result;
    }

    // Valida se o tipo contém apenas caracteres permitidos (a-z, A-Z, 0-9, _ e -)
    if (request.type.find_first_not_of(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    ) != std::string::npos) {
        result.error_message = "Tipo contém caracteres inválidos";
        return result;
    }

    // Valida se há entradas para a predição
    if (request.inputs.empty()) {
        result.error_message = "Entrada vazia";
        return result;
    }

    // Valida se o identificador do modelo foi informado
     if (request.model_id.empty()) {
        result.error_message = "Modelo inválido";
        return result;
    }
    
    try {

        // Executa a inferência através do orchestrator
        result = orchestrator_.execute(request);

    } catch (const std::exception& e) {
        // Captura erros padrão e adiciona mensagem detalhada
        result.error_message = std::string("Erro durante inferência: ") + e.what();
    } catch (...) {
        // Captura qualquer outro erro desconhecido
        result.error_message = "Erro desconhecido durante inferência";
    }
    // Retorna o resultado da predição ou erro
    return result;
}

}
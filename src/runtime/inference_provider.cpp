#include "miia/runtime/inference_provider.hpp"
#include <unordered_set>
#include <chrono>

namespace miia::runtime {
InferenceProvider::InferenceProvider() = default;

InferenceProvider::~InferenceProvider() = default;

PredictionResult InferenceProvider::predict(const PredictionRequest& request) {
    PredictionResult result;

    if (request.type.empty()) {
        result.error_message = "Tipo do modelo não especificado";
        return result;
    }

    if (request.type != "numerical" && request.type != "symbolical" && request.type && "procedural"  ) {
        result.error_message = "Tipo do modelo inválido";
        return result;
    }

    if (request.type.find_first_not_of(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    ) != std::string::npos) {
        result.error_message = "Tipo contém caracteres inválidos";
        return result;
    }

    if (request.inputs.empty()) {
        result.error_message = "Entrada vazia";
        return result;
    }

     if (request.model_id.empty()) {
        result.error_message = "Modelo inválido";
        return result;
    }

    try {
      
        result = orchestrator_.execute(request);

    } catch (const std::exception& e) {
       
        result.error_message = std::string("Erro durante inferência: ") + e.what();
    } catch (...) {
        
        result.error_message = "Erro desconhecido durante inferência";
    }

    return result;
}

}
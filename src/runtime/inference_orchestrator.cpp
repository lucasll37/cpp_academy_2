#include "miia/runtime/inference_orchestrator.hpp"
#include "miia/runtime/numerical_runtime.hpp"
#include "miia/runtime/procedural_runtime.hpp"
#include "miia/runtime/symbolic_runtime.hpp"

#include <future>
#include <stdexcept>

namespace miia::runtime {

std::string InferenceOrchestrator::resolve_model_path(const std::string model_id){

    //Implementar busca no sistema de arquivos baseado no model_id;

    // Retorna uma string vazia por enquanto!
    return "";
}

// Método responsável por obter ou criar um runtime de forma thread-safe
std::shared_ptr<InferenceOrchestrator::RuntimeEntry>
InferenceOrchestrator::get_or_create(const std::string& type, const std::string& model_id) {
    {
        // Primeira verificação rápida (evita lock prolongado)
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = runtimes_.find(type);
        if (it != runtimes_.end()) {
            return it->second;
        }
    }

    // Lock para criação (double-check)
    std::lock_guard<std::mutex> lock(map_mutex_);

    auto it = runtimes_.find(type);
    if (it != runtimes_.end()) {
        return it->second;
    }

    auto entry = std::make_shared<RuntimeEntry>();

    // Criação do runtime baseada no tipo (Factory interna)
    if (type == "numerical") {
        entry->runtime = std::make_shared<NumericalRuntime>();
        entry->is_stateful = false;
    } else if (type == "symbolic") {
        entry->runtime = std::make_shared<SymbolicRuntime>();
        entry->is_stateful = true;
    } else if (type == "procedural") {
        entry->runtime = std::make_shared<ProceduralRuntime>(resolve_model_path(model_id));
        entry->is_stateful = true;
    } else {
        throw std::runtime_error("Tipo de runtime desconhecido: " + type);
    }

    // Armazena no cache
    runtimes_[type] = entry;
    return entry;
}

PredictionResult InferenceOrchestrator::execute(const PredictionRequest& request) {

    // Obtém ou cria o runtime com base no tipo do request
    auto entry = get_or_create(request.type, request.model_id);

    // Referência para os inputs (evita cópia desnecessária)
    const auto& inputs = request.inputs;

    // Caso stateless: pode executar em paralelo sem problemas
    if (!entry->is_stateful) {
        auto future = std::async(std::launch::async, [entry, inputs]() {
            return entry->runtime->run(inputs);
        });
        return future.get();
    }

    // Caso stateful: protege com mutex (execução serializada)
    auto future = std::async(std::launch::async, [entry, inputs]() {
        std::lock_guard<std::mutex> lock(entry->mutex);
        return entry->runtime->run(inputs);
    });

    return future.get();
}

}
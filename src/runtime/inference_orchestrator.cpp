#include "miia/runtime/inference_orchestrator.hpp"
#include "miia/runtime/numerical_runtime.hpp"
#include "miia/runtime/procedural_runtime.hpp"
#include "miia/runtime/symbolic_runtime.hpp"

#include <future>
#include <stdexcept>

namespace miia::runtime {

// Método responsável por obter ou criar um runtime de forma thread-safe
std::shared_ptr<InferenceOrchestrator::RuntimeEntry>
InferenceOrchestrator::get_or_create(const std::string& type) {
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
        entry->runtime = std::make_shared<ProceduralRuntime>();
        entry->is_stateful = true;
    } else {
        throw std::runtime_error("Tipo de runtime desconhecido: " + type);
    }

    // Armazena no cache
    runtimes_[type] = entry;
    return entry;
}

// Método principal de execução
std::string InferenceOrchestrator::execute(
    const std::string& type,
    const std::string& input
) {
    // Obtém ou cria o runtime correspondente
    auto entry = get_or_create(type);

    // Caso stateless: pode executar em paralelo sem problemas
    if (!entry->is_stateful) {
        auto future = std::async(std::launch::async, [entry, input]() {
            return entry->runtime->run(input);
        });
        return future.get();
    }

    // Caso stateful: protege com mutex (execução serializada)
    auto future = std::async(std::launch::async, [entry, input]() {
        std::lock_guard<std::mutex> lock(entry->mutex);
        return entry->runtime->run(input);
    });

    return future.get();
}

}
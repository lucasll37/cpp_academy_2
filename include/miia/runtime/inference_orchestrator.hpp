#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>

#include "miia/runtime/iruntime.hpp"

namespace miia::runtime {

// Classe responsável por coordenar a execução dos diferentes runtimes.
// Também atua como fábrica (lazy) e cache dos runtimes.
class InferenceOrchestrator {
public:
    // Executa uma inferência com base no tipo do runtime
    std::string execute(const std::string& type,
                        const std::string& input);

private:
    // Estrutura interna que encapsula um runtime
    struct RuntimeEntry {
        std::shared_ptr<IRuntime> runtime; // Implementação concreta
        bool is_stateful;                  // Indica se precisa de proteção
        std::mutex mutex;                  // Usado se for stateful
    };

    // Retorna um runtime existente ou cria um novo (lazy initialization)
    std::shared_ptr<RuntimeEntry> get_or_create(const std::string& type);

    // Mapa de runtimes por tipo
    std::unordered_map<std::string, std::shared_ptr<RuntimeEntry>> runtimes_;

    // Mutex para proteger criação/acesso ao mapa
    std::mutex map_mutex_;
};

}
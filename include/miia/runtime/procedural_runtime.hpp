#pragma once

#include "miia/runtime/iruntime.hpp"

// STL básico
#include <map>
#include <string>
#include <vector>

// pybind11 - versão muito mais amigável da API do Python
#include <pybind11/embed.h>

namespace miia::runtime {

// Alias para evitar escrever pybind11 toda hora
namespace py = pybind11;

class ProceduralRuntime : public IRuntime {
public:
    // Recebe o caminho do arquivo Python (modelo)
    ProceduralRuntime(const std::string& model_path);

    // Destrutor padrão (pybind11 cuida da maior parte da memória)
    ~ProceduralRuntime();

    // Executa o modelo:
    PredictionResult run(const std::unordered_map<std::string, std::vector<float>>& inputs) override;

private:
    // Responsável por carregar o modelo Python (lazy loading)
    bool load_model();

private:
    // Caminho do arquivo .py
    std::string model_path_;

    // Indica se o modelo já foi carregado
    bool loaded_ = false;

    // Objeto Python da instância do modelo
    py::object model_;

    // Referência ao método predict do modelo
    py::object predict_;
};

}
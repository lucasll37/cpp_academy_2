#include "miia/runtime/procedural_runtime.hpp"

#include <filesystem>
#include <iostream>

// Alias para facilitar
namespace py = pybind11;
namespace fs = std::filesystem;

namespace miia::runtime {

// Construtor: só guarda o caminho do modelo
ProceduralRuntime::ProceduralRuntime(const std::string& model_path)
    : model_path_(model_path) {}

// Destrutor padrão
// Não precisamos manualmente liberar py::object (RAII faz isso)
ProceduralRuntime::~ProceduralRuntime() = default;

bool ProceduralRuntime::load_model() {
    // Evita carregar mais de uma vez
    if (loaded_) return true;

    // Inicializa o interpretador Python
    // static garante que isso só acontece uma vez no processo inteiro
    static py::scoped_interpreter guard{};

    // Resolve caminho absoluto do arquivo
    fs::path p = fs::absolute(model_path_);

    if (!fs::exists(p)) {
        std::cerr << "Modelo não encontrado: " << model_path_ << std::endl;
        return false;
    }

    // Nome do módulo Python (arquivo sem .py)
    std::string module = p.stem().string();

    // Diretório onde o modelo está
    std::string dir = p.parent_path().string();

    // Adiciona diretório ao sys.path para permitir import
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, dir);

    // Importa o módulo Python dinamicamente
    py::module_ mod = py::module_::import(module.c_str());

    // Vamos procurar dentro do módulo uma classe com método "predict"
    py::object cls;

    // Percorre todos os itens do módulo (__dict__)
    for (auto item : mod.attr("__dict__")) {
        py::object val = item.second;

        // Verifica:
        // - é uma classe
        // - possui atributo "predict"
        if (py::isinstance<py::type>(val) && py::hasattr(val, "predict")) {
            cls = val;
            break;
        }
    }

    // Se não encontrou nenhuma classe válida
    if (!cls) {
        std::cerr << "Nenhuma classe com predict()" << std::endl;
        return false;
    }

    // Instancia a classe Python (equivalente a: obj = Classe())
    model_ = cls();

    // Obtém o método predict já associado à instância
    // (equivalente a: obj.predict)
    predict_ = model_.attr("predict");

    loaded_ = true;
    return true;
}

std::map<std::string, std::vector<float>>
ProceduralRuntime::run(const std::map<std::string, std::vector<float>>& input) {

    // Garante que o modelo está carregado
    if (!load_model()) return {};

    // Cria um dicionário Python
    py::dict py_inputs;

    // Converte C++ -> Python automaticamente
    for (const auto& [k, v] : input) {
        // pybind11 converte std::vector<float> -> list automaticamente
        py_inputs[py::str(k)] = v;
    }

    // Chama o método predict do modelo
    // equivalente a: result = predict(inputs)
    py::object result = predict_(py_inputs);

    // Converte Python -> C++ automaticamente
    // (dict[str, list[float]] -> map<string, vector<float>>)
    return result.cast<std::map<std::string, std::vector<float>>>();
}

}
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

    // Obtém o __dict__ do módulo como um dict Python explícito
    py::dict dict = mod.attr("__dict__");

    // Percorre todos os itens (chave, valor)
    for (auto item : dict) {
        // item.first e item.second são py::handle → precisamos converter
        py::object val = py::reinterpret_borrow<py::object>(item.second);

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

PredictionResult ProceduralRuntime::run(const std::unordered_map<std::string, std::vector<float>>& inputs) {

    PredictionResult runtime_result;

    // Garante que o modelo está carregado
    if (!load_model()) {
        runtime_result.error_message = "Erro ao carregar modelo";
        return runtime_result;
    }

    try {
        // Cria um dicionário Python
        py::dict py_inputs;

        // Converte C++ -> Python automaticamente
        for (const auto& [k, v] : inputs) {
            py_inputs[py::str(k)] = v;
        }

        // Chama o método predict do modelo
        py::object result = predict_(py_inputs);

        // Converte Python -> C++
        runtime_result.outputs =
            result.cast<std::unordered_map<std::string, std::vector<float>>>();

    } catch (const py::error_already_set& e) {
        // Erro vindo do Python
        runtime_result.error_message = e.what();
    } catch (const std::exception& e) {
        // Qualquer outro erro C++
        runtime_result.error_message = e.what();
    } catch (...) {
        // Porque sempre existe algo pior
        runtime_result.error_message = "Erro desconhecido durante execução do modelo";
    }

    return runtime_result;
}

}
// =============================================================================
/// @file   backend_registry.hpp
/// @brief  Registro singleton de backends de inferência do Miia.
///
/// @details
/// O `BackendRegistry` implementa o padrão *Registry* sobre o padrão
/// *Factory Method* definido em #BackendFactory.  Mantém um mapa de
/// extensão de arquivo → fábrica e permite que o #InferenceEngine crie
/// backends sem depender de tipos concretos.
///
/// ### Registro automático na inicialização do motor
/// O #InferenceEngine registra os backends padrão no seu construtor:
/// @code
/// // Feito automaticamente — não requer intervenção do usuário:
/// registry.register_backend(".onnx", std::make_unique<OnnxBackendFactory>(...));
/// registry.register_backend(".py",   std::make_unique<PythonBackendFactory>());
/// @endcode
///
/// ### Adicionando um novo backend
/// Implemente #ModelBackend + #BackendFactory e registre **uma única vez**,
/// antes de qualquer chamada a `create_for_file()`:
/// @code
/// #include <inference/backend_registry.hpp>
/// #include "meu_backend.hpp"
///
/// BackendRegistry::instance().register_backend(
///     ".meu_ext", std::make_unique<MeuBackendFactory>());
///
/// // A partir daqui, qualquer arquivo ".meu_ext" é tratado automaticamente:
/// auto backend = BackendRegistry::instance().create_for_file("model.meu_ext");
/// @endcode
/// Nenhuma outra alteração no motor, no cliente ou no servidor é necessária.
///
/// ### Thread-safety
/// A instância singleton é criada de forma thread-safe pelo mecanismo
/// *magic static* do C++11 (`static` local em `instance()`).  Porém, as
/// operações de leitura (`create_for_file`, `detect_backend`, etc.) e escrita
/// (`register_backend`) **não** são protegidas por mutex.  O padrão esperado
/// é que o registro ocorra uma única vez durante a inicialização do processo
/// (antes de qualquer uso concorrente).
///
/// @see mlinference::inference::ModelBackend
/// @see mlinference::inference::BackendFactory
/// @see mlinference::inference::InferenceEngine
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_BACKEND_REGISTRY_HPP
#define ML_INFERENCE_BACKEND_REGISTRY_HPP

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_backend.hpp"

namespace mlinference {
namespace inference {

// =============================================================================
// BackendRegistry
// =============================================================================

/// @brief Registro singleton que mapeia extensões de arquivo a fábricas de backend.
///
/// @details
/// Implementa o padrão *Meyers Singleton* — a instância única é criada na
/// primeira chamada a `instance()` e destruída no final do programa.
/// A destruição estática é segura desde que nenhum outro objeto estático
/// dependa do registro após o término de `main()`.
///
/// O mapa interno é `std::map<std::string, std::unique_ptr<BackendFactory>>`,
/// indexado pela extensão com ponto (ex.: `".onnx"`, `".py"`).
/// Extensões sem ponto ou arquivos sem extensão retornam `BACKEND_UNKNOWN`
/// em `detect_backend()` e lançam `std::runtime_error` em `create_for_file()`.
class BackendRegistry {
public:
    // -------------------------------------------------------------------------
    /// @name Acesso à instância
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Retorna a instância única do registro.
    ///
    /// @details
    /// Usa *magic static* (C++11 §6.7): a inicialização é thread-safe e ocorre
    /// exatamente uma vez.  Não requer sincronização explícita para leitura.
    ///
    /// @return Referência para a instância singleton.
    static BackendRegistry& instance() {
        static BackendRegistry registry;
        return registry;
    }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Registro
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Registra uma fábrica para uma extensão de arquivo.
    ///
    /// @details
    /// Se uma fábrica já estiver registrada para a extensão, ela é substituída
    /// silenciosamente.  O #InferenceEngine usa `supports()` para evitar
    /// registros duplicados:
    /// @code
    /// if (!registry.supports(".onnx"))
    ///     registry.register_backend(".onnx", std::make_unique<OnnxBackendFactory>(...));
    /// @endcode
    ///
    /// @param extension  Extensão com ponto (ex.: `".onnx"`, `".py"`).
    /// @param factory    Fábrica que cria instâncias do backend para esta extensão.
    void register_backend(const std::string& extension,
                          std::unique_ptr<BackendFactory> factory) {
        factories_[extension] = std::move(factory);
    }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Criação de backends
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Cria um backend para o arquivo indicado, detectando o tipo pela extensão.
    ///
    /// @details
    /// Extrai a extensão de @p path com `rfind('.')` e procura no mapa de
    /// fábricas.  Retorna uma instância não-carregada do backend correspondente.
    ///
    /// @param path  Caminho do arquivo (somente a extensão é usada).
    ///
    /// @return Nova instância não-carregada do backend adequado.
    ///
    /// @throws std::runtime_error Se nenhuma fábrica estiver registrada para
    ///         a extensão extraída de @p path.
    std::unique_ptr<ModelBackend> create_for_file(const std::string& path) const {
        std::string ext = get_extension(path);
        auto it = factories_.find(ext);
        if (it == factories_.end())
            throw std::runtime_error(
                "No backend registered for extension: " + ext);
        return it->second->create();
    }

    /// @brief Cria um backend pelo tipo enum protobuf, sem depender de extensão.
    ///
    /// @details
    /// Percorre todas as fábricas registradas e retorna a primeira cujo
    /// `BackendFactory::backend_type()` coincide com @p type.
    /// Usado pelo #InferenceEngine quando `force_backend` é especificado.
    ///
    /// @param type  Tipo de backend (ex.: `common::BACKEND_PYTHON`).
    ///
    /// @return Nova instância não-carregada do backend correspondente.
    ///
    /// @throws std::runtime_error Se nenhuma fábrica estiver registrada para
    ///         o tipo informado.
    std::unique_ptr<ModelBackend> create_by_type(common::BackendType type) const {
        for (const auto& [ext, factory] : factories_) {
            if (factory->backend_type() == type)
                return factory->create();
        }
        throw std::runtime_error(
            "No backend registered for type: " + std::to_string(type));
    }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Detecção e introspecção
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Detecta qual backend trataria o arquivo indicado.
    ///
    /// @details
    /// Não lança exceção — retorna `common::BACKEND_UNKNOWN` se a extensão
    /// não estiver registrada.  Útil para verificação antes de `create_for_file()`.
    ///
    /// @param path  Caminho do arquivo (somente a extensão é usada).
    ///
    /// @return Tipo do backend registrado para a extensão, ou
    ///         `common::BACKEND_UNKNOWN` se não houver correspondência.
    common::BackendType detect_backend(const std::string& path) const {
        std::string ext = get_extension(path);
        auto it = factories_.find(ext);
        if (it == factories_.end())
            return common::BACKEND_UNKNOWN;
        return it->second->backend_type();
    }

    /// @brief Lista todas as extensões atualmente registradas.
    ///
    /// @return Vetor de strings com extensões (ex.: `{".onnx", ".py"}`).
    ///         A ordem é determinada pelo `std::map` (lexicográfica).
    std::vector<std::string> registered_extensions() const {
        std::vector<std::string> exts;
        exts.reserve(factories_.size());
        for (const auto& [ext, _] : factories_)
            exts.push_back(ext);
        return exts;
    }

    /// @brief Lista os nomes de todos os backends registrados.
    ///
    /// @return Vetor de strings com nomes (ex.: `{"onnx", "python"}`),
    ///         na mesma ordem de `registered_extensions()`.
    std::vector<std::string> registered_backend_names() const {
        std::vector<std::string> names;
        names.reserve(factories_.size());
        for (const auto& [_, factory] : factories_)
            names.push_back(factory->name());
        return names;
    }

    /// @brief Verifica se uma extensão possui backend registrado.
    ///
    /// @param extension  Extensão com ponto (ex.: `".py"`).
    ///
    /// @return @c true se há uma fábrica registrada para @p extension.
    bool supports(const std::string& extension) const {
        return factories_.find(extension) != factories_.end();
    }

    /// @}

private:
    /// @cond INTERNAL

    /// Construtor privado — acesso somente via `instance()`.
    BackendRegistry() = default;

    /// Cópia proibida — singleton.
    BackendRegistry(const BackendRegistry&) = delete;

    /// Atribuição proibida — singleton.
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    /// @brief Mapa extensão → fábrica.
    std::map<std::string, std::unique_ptr<BackendFactory>> factories_;

    /// @brief Extrai a extensão (com ponto) do caminho informado.
    ///
    /// @return Extensão a partir do último `'.'`, ou string vazia se
    ///         não houver ponto no caminho.
    static std::string get_extension(const std::string& path) {
        auto pos = path.rfind('.');
        if (pos == std::string::npos) return "";
        return path.substr(pos);
    }

    /// @endcond
};

}  // namespace inference
}  // namespace mlinference

#endif  // ML_INFERENCE_BACKEND_REGISTRY_HPP
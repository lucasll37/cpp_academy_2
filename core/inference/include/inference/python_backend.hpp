// =============================================================================
/// @file   python_backend.hpp
/// @brief  Backend de inferência via CPython embutido do AsaMiia.
///
/// @details
/// Implementa #ModelBackend carregando arquivos `.py` que definem uma
/// subclasse de `MiiaModel`.  O interpretador CPython é inicializado uma
/// única vez por processo (contagem de referências via `instance_count_`) e
/// todas as chamadas Python são protegidas pelo GIL.
///
/// ### Fluxo de inferência
/// ```
/// predict(inputs: Object)
///   ├── PyGILState_Ensure()
///   ├── inputs_to_py_dict(inputs)   // Object → PyObject* (dict)
///   ├── PyObject_CallObject(py_predict_method_, args)
///   │     └── model.predict(inputs) em Python
///   ├── py_dict_to_outputs(result)  // PyObject* → Object
///   └── PyGILState_Release()
/// ```
///
/// ### Ciclo de vida do interpretador CPython
/// O CPython é inicializado em `ensure_interpreter()` na primeira chamada
/// a `load()` e nunca finalizado (`Py_Finalize()` é considerado inseguro
/// em processos que embarcam o interpretador junto com bibliotecas nativas).
/// A contagem de instâncias (`instance_count_`) garante que a inicialização
/// ocorra apenas uma vez, mesmo com múltiplos backends ativos.
///
/// ### Gestão do GIL
/// Após `Py_Initialize()`, o GIL é liberado via `PyEval_SaveThread()` para
/// que threads C++ possam executar sem bloquear o interpretador.  Cada
/// operação Python adquire o GIL com `PyGILState_Ensure()` e o libera com
/// `PyGILState_Release()` ao final.
///
/// ### Injeção de venv
/// Na carga de cada modelo, `inject_venv_from_model_dir()` procura por
/// `<model_dir>/.venv/lib/python*/site-packages` e injeta o caminho em
/// `sys.path`, permitindo que o modelo use pacotes instalados no venv
/// (ex.: `numpy`).  A numpy C-API é inicializada via `import_array1()`
/// encapsulada em `init_numpy_capi()` para evitar o problema de `return`
/// prematuro da macro.
///
/// ### Conversão de tipos
/// - **Object → dict Python:** `inputs_to_py_dict()` converte recursivamente
///   cada #client::Value para o tipo Python equivalente (float → `float`,
///   bool → `bool`, string → `str`, Array → `list`, Object → `dict`,
///   Null → `None`).
/// - **dict Python → Object:** `py_dict_to_outputs()` converte arrays numpy
///   para #client::Array ou escalar (single-element collapse).
///
/// ### Cache do schema
/// O schema (`MiiaModel.get_schema()`) é extraído uma única vez em `load()`
/// e armazenado em `cached_schema_`.  Chamadas subsequentes a `get_schema()`
/// retornam o cache sem tocar no interpretador.
///
/// ### Forward-declaration de PyObject
/// `Python.h` **não é incluído** neste header — apenas `struct _object` é
/// forward-declarado.  O include real fica em `python_backend.cpp`, evitando
/// contaminação do namespace global em todas as translation units.
///
/// @see python/models/miia_model.py
/// @see mlinference::inference::OnnxBackend
/// @see mlinference::inference::ModelBackend
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_PYTHON_BACKEND_HPP
#define ML_INFERENCE_PYTHON_BACKEND_HPP

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "model_backend.hpp"

// Forward-declaration de PyObject para não vazar Python.h para todo TU
// que incluir este header.  O #include <Python.h> real fica apenas em
// python_backend.cpp.
struct _object;
typedef _object PyObject;

namespace mlinference {
namespace inference {

// =============================================================================
// PythonBackend
// =============================================================================

/// @brief Backend de inferência que executa modelos `.py` via CPython embutido.
///
/// @details
/// Cada instância gerencia:
/// - Um módulo Python importado (`module_name_`).
/// - Uma instância da subclasse `MiiaModel` encontrada no módulo.
/// - Referências aos bound methods `predict` e `get_schema`.
/// - O schema extraído na carga (`cached_schema_`).
///
/// Instâncias **não** são thread-safe — o #InferenceEngine serializa
/// o acesso via `mutex_`.  O GIL é adquirido internamente em cada
/// operação Python.
class PythonBackend : public ModelBackend {
public:
    // -------------------------------------------------------------------------
    /// @name Construção e destruição
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Constrói o backend (sem inicializar o interpretador).
    ///
    /// @details
    /// O interpretador CPython é inicializado somente em `load()`, via
    /// `ensure_interpreter()`.
    PythonBackend();

    /// @brief Destrói o backend, chamando `unload()` se carregado.
    ~PythonBackend() override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Interface ModelBackend
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Carrega um arquivo `.py` e instancia a subclasse `MiiaModel`.
    ///
    /// @details
    /// Executa os seguintes passos com o GIL adquirido:
    /// 1. `ensure_interpreter()` — inicializa CPython se necessário.
    /// 2. `inject_venv_from_model_dir()` — injeta site-packages do venv.
    /// 3. Insere `model_dir_` no início de `sys.path` (sem duplicatas).
    /// 4. `find_model_class()` — importa o módulo e localiza a subclasse.
    /// 5. Instancia a classe (`model = ModelClass()`).
    /// 6. Chama `model.load()`.
    /// 7. Extrai e cacheia o schema via `extract_schema_from_python()`.
    ///
    /// @param path    Caminho absoluto do arquivo `.py`.
    /// @param config  Ignorado por este backend.
    ///
    /// @return @c true se todos os passos foram bem-sucedidos.
    ///         @c false se o arquivo não existir, a classe não for encontrada,
    ///         ou `model.load()` lançar exceção Python.
    bool load(const std::string& path,
              const std::map<std::string, std::string>& config) override;

    /// @brief Chama `model.unload()`, decrementa `instance_count_` e libera refs Python.
    ///
    /// @details
    /// Adquire o GIL, chama `model.unload()` (ignorando erros), e executa
    /// `Py_CLEAR` em `py_model_instance_`, `py_predict_method_` e
    /// `py_schema_method_`.  Não chama `Py_Finalize()`.
    void unload() override;

    /// @brief Executa inferência chamando `model.predict(inputs)`.
    ///
    /// @details
    /// Converte @p inputs para `dict` Python via `inputs_to_py_dict()`,
    /// chama `model.predict(inputs)` e converte o resultado de volta com
    /// `py_dict_to_outputs()`.  Erros Python são capturados via
    /// `fetch_python_error()` (traceback completo) e retornados em
    /// `InferenceResult::error_message`.
    ///
    /// @param inputs  Mapa de entradas — estrutura arbitrária aninhada.
    ///
    /// @return #InferenceResult com outputs convertidos.
    ///         `success = false` se a conversão ou o `predict()` Python falharem.
    InferenceResult predict(const client::Object& inputs) override;

    /// @brief Retorna o schema cacheado extraído em `load()`.
    ///
    /// @details
    /// Não acessa o interpretador — retorna `cached_schema_` diretamente.
    ///
    /// @return #ModelSchema extraído de `model.get_schema()`.
    ModelSchema get_schema() const override;

    /// @brief Retorna `common::BACKEND_PYTHON`.
    common::BackendType backend_type() const override {
        return common::BACKEND_PYTHON;
    }

    /// @brief Retorna `0` — uso de memória Python não é trivialmente mensurável.
    int64_t memory_usage_bytes() const override;

    /// @brief Verifica existência e extensão `.py` sem carregar o modelo.
    ///
    /// @param path  Caminho do arquivo a validar.
    ///
    /// @return String vazia se válido; mensagem de erro caso contrário.
    std::string validate(const std::string& path) const override;

    /// @brief Executa `n` inferências sintéticas delegando para `ModelBackend::warmup()`.
    ///
    /// @param n  Número de execuções de aquecimento.
    void warmup(uint32_t n) override;

    /// @}

private:
    /// @cond INTERNAL

    // -------------------------------------------------------------------------
    // Gerenciamento global do interpretador (compartilhado entre instâncias)
    // -------------------------------------------------------------------------

    /// @brief Mutex que protege a inicialização do interpretador CPython.
    static std::mutex init_mutex_;

    /// @brief Contagem de instâncias ativas — controla inicialização do CPython.
    ///
    /// Incrementado em `ensure_interpreter()`, decrementado em
    /// `release_interpreter()`.  Quando atinge zero, o interpretador
    /// **não** é finalizado (unsafe em processos com extensões nativas).
    static int instance_count_;

    /// @brief Injeta `<model_dir>/.venv/lib/python*/site-packages` em `sys.path`.
    ///
    /// @details
    /// Também inicializa a numpy C-API via `init_numpy_capi()` (encapsulada
    /// para evitar o `return` prematuro da macro `import_array()`).
    /// Adquire o GIL internamente.
    ///
    /// @param model_dir  Diretório do arquivo `.py` carregado.
    void inject_venv_from_model_dir(const std::string& model_dir);

    /// @brief Inicializa o CPython se `instance_count_ == 0`.
    ///
    /// @details
    /// Chama `Py_Initialize()` e, se a thread atual possuir o GIL,
    /// libera-o via `PyEval_SaveThread()` para não bloquear threads C++.
    /// Se o interpretador já estiver ativo (hospedado externamente), apenas
    /// verifica o estado do GIL.
    static void ensure_interpreter();

    /// @brief Decrementa `instance_count_`.  Não chama `Py_Finalize()`.
    static void release_interpreter();

    // -------------------------------------------------------------------------
    // Handles Python (sem vazar Python.h)
    // -------------------------------------------------------------------------

    /// @brief Instância da subclasse `MiiaModel` encontrada no módulo.
    PyObject* py_model_instance_ = nullptr;

    /// @brief Bound method `model.predict` — cached para evitar lookup repetido.
    PyObject* py_predict_method_ = nullptr;

    /// @brief Bound method `model.get_schema` — usado uma vez em `load()`.
    PyObject* py_schema_method_  = nullptr;

    // -------------------------------------------------------------------------
    // Estado por instância
    // -------------------------------------------------------------------------

    /// @brief Schema extraído em `load()` e retornado por `get_schema()`.
    ModelSchema cached_schema_;

    /// @brief Diretório do arquivo `.py` (inserido em `sys.path`).
    std::string model_dir_;

    /// @brief Nome do módulo Python (nome do arquivo sem `.py`).
    ///
    /// Usado como chave em `sys.modules` para identificar o módulo importado.
    std::string module_name_;

    // -------------------------------------------------------------------------
    // Helpers privados (implementados em python_backend.cpp)
    // -------------------------------------------------------------------------

    /// @brief Importa o módulo e localiza a primeira subclasse de `MiiaModel`.
    ///
    /// @details
    /// Chama `PyImport_ImportModule(module_name_)` e itera sobre os atributos
    /// do módulo buscando a primeira classe que passe em `PyObject_IsSubclass`.
    ///
    /// @param path  Caminho do arquivo (apenas para mensagens de erro).
    ///
    /// @return Referência nova (*new-ref*) para a classe, ou @c nullptr em falha.
    PyObject* find_model_class(const std::string& path);

    /// @brief Converte #client::Object para `PyObject*` (dict Python).
    ///
    /// @details
    /// Converte recursivamente cada #client::Value para o tipo Python
    /// equivalente usando `value_to_py()`:
    /// - `double` → `float`
    /// - `bool`   → `bool`
    /// - `string` → `str`
    /// - #Array   → `list`
    /// - #Object  → `dict`
    /// - #Null    → `None`
    ///
    /// @param inputs  Mapa de entradas a converter.
    ///
    /// @return Nova referência para o dict Python, ou @c nullptr em falha.
    PyObject* inputs_to_py_dict(const client::Object& inputs) const;

    /// @brief Converte `PyObject*` (dict Python) para #client::Object.
    ///
    /// @details
    /// Itera sobre as chaves do dict Python retornado por `model.predict()`.
    /// Arrays numpy (`ndarray`) são convertidos para #client::Array de
    /// `double`; tensores com um único elemento tornam-se escalares.
    ///
    /// @param py_dict  Dict Python retornado por `model.predict()`.
    /// @param outputs  Object de saída a preencher.
    /// @param error    Preenchido com mensagem de erro em caso de falha.
    ///
    /// @return @c true se a conversão foi bem-sucedida.
    bool py_dict_to_outputs(
        PyObject*       py_dict,
        client::Object& outputs,
        std::string&    error) const;

    /// @brief Chama `model.get_schema()` e converte o dataclass Python para #ModelSchema.
    ///
    /// @details
    /// Extrai `inputs`, `outputs`, `description`, `author` e `tags` do
    /// objeto `ModelSchema` Python retornado por `get_schema()`.
    /// Chamado uma única vez em `load()`.
    ///
    /// @return #ModelSchema preenchido; campos default em caso de falha.
    ModelSchema extract_schema_from_python() const;

    /// @brief Converte um `TensorSpec` Python para #TensorSpecData C++.
    ///
    /// @details
    /// Extrai `name`, `dtype`, `shape`, `description` e `structured` do
    /// dataclass Python via `PyObject_GetAttrString()`.
    ///
    /// @param py_spec  Instância de `TensorSpec` Python.
    ///
    /// @return #TensorSpecData preenchido; campos default em caso de falha.
    TensorSpecData parse_tensor_spec(PyObject* py_spec) const;

    /// @endcond
};

// =============================================================================
// PythonBackendFactory
// =============================================================================

/// @brief Fábrica que cria instâncias de #PythonBackend para o #BackendRegistry.
///
/// @details
/// Registrada no #BackendRegistry para a extensão `".py"` pelo
/// #InferenceEngine no construtor.  Não requer parâmetros de hardware —
/// o #PythonBackend sempre executa em CPU via CPython.
class PythonBackendFactory : public BackendFactory {
public:
    /// @brief Cria e retorna uma nova instância de #PythonBackend.
    std::unique_ptr<ModelBackend> create() const override {
        return std::make_unique<PythonBackend>();
    }

    /// @brief Retorna `common::BACKEND_PYTHON`.
    common::BackendType backend_type() const override {
        return common::BACKEND_PYTHON;
    }

    /// @brief Retorna `"python"`.
    std::string name() const override { return "python"; }
};

}  // namespace inference
}  // namespace mlinference

#endif  // ML_INFERENCE_PYTHON_BACKEND_HPP
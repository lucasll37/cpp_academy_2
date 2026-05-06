// =============================================================================
/// @file   onnx_backend.hpp
/// @brief  Backend de inferência ONNX Runtime do AsaMiia.
///
/// @details
/// Implementa #ModelBackend carregando arquivos `.onnx` via ONNX Runtime C++
/// API (`Ort::Session`).  Suporta execução em CPU e, opcionalmente, em GPU
/// via CUDA Execution Provider.
///
/// ### Fluxo de inferência
/// ```
/// predict(inputs: Object)
///   ├── Para cada tensor de entrada:
///   │     value_to_floats(inputs[name])   // Value → vector<float>
///   │     resolve_dynamic_shape(shape, n)  // resolve dims -1
///   │     Ort::Value::CreateTensor<float>  // sem cópia dos dados
///   ├── session_->Run(input_tensors → output_tensors)
///   └── Para cada tensor de saída:
///         floats_to_value(ptr, count)      // float* → Value (escalar ou Array)
/// ```
///
/// ### Conversão de tipos
/// O backend opera exclusivamente com tensores `float32`.  Inputs chegam
/// como #client::Value (escalar `double` ou #client::Array de números) e
/// são convertidos para `vector<float>` antes de criar o `Ort::Value`.
/// Outputs são convertidos de volta: tensores com um único elemento tornam-se
/// escalares; tensores com múltiplos elementos tornam-se #client::Array.
///
/// ### Dimensões dinâmicas
/// Shapes com `−1` (dinâmico) são resolvidos em `resolve_dynamic_shape()`
/// a partir do número de elementos fornecidos.  Se a resolução for
/// ambígua ou inconsistente, a inferência falha com mensagem de erro.
///
/// ### Persistência de buffers
/// `Ort::Value::CreateTensor` **não copia** os dados — mantém referência
/// ao buffer externo.  Os `vector<float>` de entrada são preservados em
/// `input_data_buffers` durante toda a chamada a `session_->Run()` para
/// evitar dangling pointers.
///
/// ### Metadados do modelo
/// `get_schema()` extrai nomes, shapes e dtypes dos tensores via
/// `session_->GetInputTypeInfo()` / `GetOutputTypeInfo()` na carga, e
/// tenta extrair descrição e autor via `session_->GetModelMetadata()`.
///
/// ### GPU (CUDA Execution Provider)
/// Se `enable_gpu = true`, o construtor tenta registrar o CUDA EP.
/// Em caso de falha (CUDA indisponível), faz fallback silencioso para CPU
/// com log de aviso.
///
/// @see mlinference::inference::PythonBackend
/// @see mlinference::inference::ModelBackend
/// @see mlinference::inference::OnnxBackendFactory
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_ONNX_BACKEND_HPP
#define ML_INFERENCE_ONNX_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "model_backend.hpp"

namespace mlinference {
namespace inference {

// =============================================================================
// OnnxBackend
// =============================================================================

/// @brief Backend de inferência que executa modelos `.onnx` via ONNX Runtime.
///
/// @details
/// Cada instância gerencia uma única `Ort::Session` e os metadados de I/O
/// extraídos na carga.  Não é thread-safe — o #InferenceEngine serializa
/// o acesso via `mutex_`.
class OnnxBackend : public ModelBackend {
public:
    // -------------------------------------------------------------------------
    /// @name Construção e destruição
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Constrói o backend e configura as opções de sessão ONNX Runtime.
    ///
    /// @details
    /// Cria o `Ort::Env` e o `Ort::SessionOptions`, define
    /// `SetIntraOpNumThreads(num_threads)` e, se `enable_gpu` for @c true,
    /// tenta registrar o CUDA Execution Provider.  Falha no CUDA resulta
    /// em fallback silencioso para CPU.
    ///
    /// @param enable_gpu   @c true para tentar usar GPU via CUDA EP.
    /// @param gpu_device   Índice do dispositivo CUDA (ignorado se `!enable_gpu`).
    /// @param num_threads  Número de threads intra-op do ONNX Runtime (padrão: 4).
    explicit OnnxBackend(bool     enable_gpu  = false,
                         uint32_t gpu_device  = 0,
                         uint32_t num_threads = 4);

    /// @brief Destrói a sessão e libera recursos do ONNX Runtime.
    ~OnnxBackend() override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Interface ModelBackend
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Carrega um modelo `.onnx` e extrai os metadados de I/O.
    ///
    /// @details
    /// Cria uma `Ort::Session` a partir do arquivo e chama `extract_metadata()`
    /// para popular `input_names_`, `input_shapes_`, `input_dtypes_`,
    /// `output_names_`, `output_shapes_` e `output_dtypes_`.
    ///
    /// Em Windows, o caminho é convertido para `std::wstring` antes de ser
    /// passado ao ONNX Runtime (exigência da API `Ort::Session` no Windows).
    ///
    /// @param path    Caminho absoluto do arquivo `.onnx`.
    /// @param config  Ignorado por este backend.
    ///
    /// @return @c true se a sessão foi criada e os metadados extraídos.
    ///         @c false se o arquivo não existir, for inválido, ou se o
    ///         backend já estiver carregado.
    bool load(const std::string& path,
              const std::map<std::string, std::string>& config) override;

    /// @brief Libera a sessão ONNX Runtime e limpa os metadados.
    void unload() override;

    /// @brief Executa inferência no modelo carregado.
    ///
    /// @details
    /// Para cada tensor de entrada declarado no modelo:
    /// 1. Busca `inputs[name]` — falha com erro se o tensor não for fornecido.
    /// 2. Converte o valor para `vector<float>` via `value_to_floats()`.
    /// 3. Resolve dimensões dinâmicas (`−1`) com `resolve_dynamic_shape()`.
    /// 4. Cria `Ort::Value::CreateTensor<float>` referenciando o buffer.
    ///
    /// Os buffers são mantidos vivos em `input_data_buffers` durante
    /// `session_->Run()` — `Ort::Value` não copia os dados.
    ///
    /// Para cada tensor de saída, converte `float*` para #client::Value via
    /// `floats_to_value()`: tensores com um único elemento tornam-se escalares.
    ///
    /// @param inputs  Mapa de entradas — cada chave deve corresponder a um
    ///                tensor declarado no modelo.
    ///
    /// @return #InferenceResult com outputs e tempo de execução.
    ///         `success = false` se algum tensor estiver ausente ou se
    ///         `session_->Run()` lançar `Ort::Exception`.
    InferenceResult predict(const client::Object& inputs) override;

    /// @brief Retorna o schema de I/O extraído na carga.
    ///
    /// @details
    /// Constrói um #ModelSchema a partir dos metadados em cache
    /// (`input_names_`, `input_shapes_`, etc.) e tenta preencher
    /// `description` e `author` via `session_->GetModelMetadata()`.
    /// Falha na extração de metadados é ignorada silenciosamente.
    ///
    /// @pre `load()` deve ter sido chamado com sucesso.
    ///
    /// @return #ModelSchema com tensores de I/O e metadados do modelo.
    ModelSchema get_schema() const override;

    /// @brief Retorna `common::BACKEND_ONNX`.
    common::BackendType backend_type() const override {
        return common::BACKEND_ONNX;
    }

    /// @brief Estima o uso de memória como o tamanho do arquivo `.onnx`.
    ///
    /// @details
    /// Abre o arquivo em modo binário e retorna `tellg()` após seek para
    /// o fim — aproximação do tamanho dos pesos serializados.
    /// Retorna `0` se o modelo não estiver carregado ou o arquivo não
    /// puder ser aberto.
    ///
    /// @return Tamanho do arquivo em bytes, ou `0`.
    int64_t memory_usage_bytes() const override;

    /// @brief Valida um arquivo `.onnx` sem carregá-lo na sessão principal.
    ///
    /// @details
    /// Cria uma `Ort::Session` temporária com `ORT_LOGGING_LEVEL_ERROR`
    /// e 1 thread.  Se a criação falhar com `Ort::Exception`, o modelo
    /// é considerado inválido.
    ///
    /// @param path  Caminho do arquivo a validar.
    ///
    /// @return String vazia se válido; mensagem de erro caso contrário.
    std::string validate(const std::string& path) const override;

    /// @}

private:
    /// @cond INTERNAL

    // -------------------------------------------------------------------------
    // Objetos ONNX Runtime
    // -------------------------------------------------------------------------

    /// @brief Ambiente global do ONNX Runtime (um por instância de backend).
    Ort::Env env_;

    /// @brief Opções de sessão: threads, execution providers, otimizações.
    Ort::SessionOptions session_options_;

    /// @brief Sessão ONNX Runtime — criada em `load()`, destruída em `unload()`.
    std::unique_ptr<Ort::Session> session_;

    // -------------------------------------------------------------------------
    // Metadados em cache (preenchidos em load() via extract_metadata())
    // -------------------------------------------------------------------------

    /// @brief Nomes dos tensores de entrada na ordem declarada pelo modelo.
    std::vector<std::string> input_names_;

    /// @brief Nomes dos tensores de saída na ordem declarada pelo modelo.
    std::vector<std::string> output_names_;

    /// @brief Shapes dos tensores de entrada; `−1` indica dimensão dinâmica.
    std::vector<std::vector<int64_t>> input_shapes_;

    /// @brief Shapes dos tensores de saída; `−1` indica dimensão dinâmica.
    std::vector<std::vector<int64_t>> output_shapes_;

    /// @brief Tipos de elemento dos tensores de entrada (enum ONNX Runtime).
    std::vector<ONNXTensorElementDataType> input_dtypes_;

    /// @brief Tipos de elemento dos tensores de saída (enum ONNX Runtime).
    std::vector<ONNXTensorElementDataType> output_dtypes_;

    /// @brief Caminho absoluto do arquivo carregado (usado em `memory_usage_bytes()`).
    std::string model_path_;

    // -------------------------------------------------------------------------
    // Configuração
    // -------------------------------------------------------------------------

    bool     enable_gpu_;   ///< @c true se CUDA EP foi solicitado.
    uint32_t gpu_device_;   ///< Índice do dispositivo CUDA.
    uint32_t num_threads_;  ///< Threads intra-op configuradas.

    // -------------------------------------------------------------------------
    // Helpers privados
    // -------------------------------------------------------------------------

    /// @brief Extrai e cacheia nomes, shapes e dtypes de todos os tensores.
    ///
    /// Chamado internamente por `load()` após criar a sessão ONNX Runtime.
    void extract_metadata();

    /// @brief Converte tipo de elemento ONNX Runtime para `common::DataType`.
    ///
    /// Ex.: `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT → common::FLOAT32`.
    static common::DataType ort_to_proto_dtype(ONNXTensorElementDataType ort_type);

    /// @brief Resolve dimensões dinâmicas (`−1`) em um shape dado o total de elementos.
    ///
    /// @details
    /// Se o shape contém exatamente um `−1`, substitui pelo valor que torna
    /// o produto igual a `data_size`.  Se a resolução for ambígua ou
    /// inconsistente, popula `error_msg` e retorna vetor vazio.
    ///
    /// @param shape        Shape com possíveis `−1`.
    /// @param data_size    Número total de elementos disponíveis.
    /// @param tensor_name  Nome do tensor (para mensagem de erro).
    /// @param error_msg    Preenchido com descrição do erro em caso de falha.
    ///
    /// @return Shape resolvido sem `−1`, ou vetor vazio em caso de erro.
    static std::vector<int64_t> resolve_dynamic_shape(
        const std::vector<int64_t>& shape,
        size_t                      data_size,
        const std::string&          tensor_name,
        std::string&                error_msg);

    /// @brief Converte um #client::Value (escalar ou Array) para `vector<float>`.
    ///
    /// @details
    /// - Escalar `double` → `{static_cast<float>(v)}`.
    /// - #client::Array de números → vetor de floats na mesma ordem.
    /// - Qualquer outro tipo → vetor vazio (inferência falhará com erro).
    static std::vector<float> value_to_floats(const client::Value& v);

    /// @brief Converte um buffer de floats para #client::Value.
    ///
    /// @details
    /// - `count == 1` → `Value{static_cast<double>(ptr[0])}` (escalar).
    /// - `count > 1`  → `Value{Array{...}}` com um elemento por float.
    ///
    /// @param ptr    Ponteiro para o início do buffer de saída do ONNX Runtime.
    /// @param count  Número de elementos no buffer.
    static client::Value floats_to_value(const float* ptr, size_t count);

    /// @endcond
};

// =============================================================================
// OnnxBackendFactory
// =============================================================================

/// @brief Fábrica que cria instâncias de #OnnxBackend para o #BackendRegistry.
///
/// @details
/// Registrada no #BackendRegistry para a extensão `".onnx"` pelo
/// #InferenceEngine no construtor.  Encapsula a configuração de GPU,
/// device e threads para que o registry não precise conhecer os parâmetros
/// de construção do backend.
class OnnxBackendFactory : public BackendFactory {
public:
    /// @brief Constrói a fábrica com as configurações de hardware.
    ///
    /// @param enable_gpu   Passar para `OnnxBackend::OnnxBackend()`.
    /// @param gpu_device   Passar para `OnnxBackend::OnnxBackend()`.
    /// @param num_threads  Passar para `OnnxBackend::OnnxBackend()`.
    explicit OnnxBackendFactory(bool     enable_gpu  = false,
                                uint32_t gpu_device  = 0,
                                uint32_t num_threads = 4)
        : enable_gpu_(enable_gpu)
        , gpu_device_(gpu_device)
        , num_threads_(num_threads) {}

    /// @brief Cria e retorna uma nova instância de #OnnxBackend.
    std::unique_ptr<ModelBackend> create() const override {
        return std::make_unique<OnnxBackend>(enable_gpu_, gpu_device_, num_threads_);
    }

    /// @brief Retorna `common::BACKEND_ONNX`.
    common::BackendType backend_type() const override {
        return common::BACKEND_ONNX;
    }

    /// @brief Retorna `"onnx"`.
    std::string name() const override { return "onnx"; }

private:
    bool     enable_gpu_;
    uint32_t gpu_device_;
    uint32_t num_threads_;
};

}  // namespace inference
}  // namespace mlinference

#endif  // ML_INFERENCE_ONNX_BACKEND_HPP
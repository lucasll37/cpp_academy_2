// =============================================================================
/// @file   inference_engine.hpp
/// @brief  Motor de inferência de alto nível do Miia.
///
/// @details
/// `InferenceEngine` é a fachada central que gerencia múltiplos modelos
/// carregados simultaneamente, cada um respaldado pelo backend adequado
/// (#PythonBackend ou #OnnxBackend), selecionado automaticamente pelo
/// #BackendRegistry a partir da extensão do arquivo.
///
/// ### Responsabilidades
/// - Ciclo de vida dos modelos: carga, descarga e verificação de presença.
/// - Despacho de inferência thread-safe para o backend correto.
/// - Introspecção de schema (I/O specs) e métricas de runtime.
/// - Aquecimento (*warmup*) de modelos com dados sintéticos.
/// - Validação estática de arquivos sem carga completa.
///
/// ### Thread-safety
/// Todos os métodos públicos adquirem `mutex_` internamente.
/// É seguro chamar `predict()` de múltiplas threads simultaneamente,
/// mas cada chamada serializa no mutex antes de despachar para o backend.
///
/// ### Detecção de backend
/// A extensão do arquivo determina o backend:
///
/// | Extensão | Backend         |
/// |----------|-----------------|
/// | `.py`    | #PythonBackend  |
/// | `.onnx`  | #OnnxBackend    |
///
/// O parâmetro `force_backend` em load_model() permite sobrescrever a
/// detecção automática quando necessário.
///
/// ### Uso típico (via InProcessBackend)
/// O `InferenceEngine` normalmente não é instanciado diretamente pelo código
/// de aplicação — ele é criado internamente pelo #InProcessBackend quando
/// `InferenceClient("inprocess")` é usado.  Para testes unitários do motor
/// é possível instanciá-lo diretamente:
///
/// @code
/// #include <inference/inference_engine.hpp>
/// using namespace miia::inference;
///
/// InferenceEngine engine(/*gpu=*/false, /*device=*/0, /*threads=*/4);
///
/// engine.load_model("nav", "/app/models/ship_avoidance.py");
///
/// client::Object inputs;
/// inputs["state"] = client::Value{client::Object{{"toHeading", client::Value{45.0}}}};
///
/// InferenceResult r = engine.predict("nav", inputs);
/// if (r.success)
///     double hdg = r.outputs["heading"].as_number();
/// @endcode
///
/// @see miia::client::InferenceClient
/// @see miia::inference::ModelBackend
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_INFERENCE_ENGINE_HPP
#define ML_INFERENCE_INFERENCE_ENGINE_HPP

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common.pb.h"
#include "model_backend.hpp"
#include "client/inference_client.hpp"  // client::Object

namespace miia {

/// @namespace miia::inference
/// @brief     Motor de inferência, backends e tipos internos do servidor Miia.
namespace inference {

// =============================================================================
// InferenceEngine
// =============================================================================

/// @brief Gerenciador de modelos de ML e fachada de inferência thread-safe.
///
/// @details
/// Mantém um mapa `model_id → LoadedModel` protegido por `mutex_`.
/// Cada `LoadedModel` contém um ponteiro único para o #ModelBackend
/// responsável pela execução da inferência naquele modelo.
///
/// A instância é tipicamente criada pelo #InProcessBackend e tem o mesmo
/// tempo de vida do processo cliente quando operando em modo in-process.
class InferenceEngine {
public:
    // -------------------------------------------------------------------------
    /// @name Construção e destruição
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Constrói o motor de inferência.
    ///
    /// @param enable_gpu    Habilita o uso de GPU via ONNX Runtime (CUDA EP).
    ///                      Ignorado para modelos Python.
    /// @param gpu_device_id Índice do dispositivo CUDA a utilizar (padrão: 0).
    /// @param num_threads   Número de threads de inferência no pool interno
    ///                      do ONNX Runtime (padrão: 4).
    explicit InferenceEngine(bool     enable_gpu    = false,
                             uint32_t gpu_device_id = 0,
                             uint32_t num_threads   = 4);

    /// @brief Destrói o motor e descarrega todos os modelos.
    ~InferenceEngine();

    /// @}

    // -------------------------------------------------------------------------
    /// @name Ciclo de vida dos modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Carrega um modelo a partir de um arquivo.
    ///
    /// @details
    /// O backend é selecionado automaticamente pela extensão do arquivo via
    /// #BackendRegistry, a menos que `force_backend` seja especificado.
    /// Um mesmo arquivo pode ser carregado com IDs distintos, criando
    /// instâncias independentes do backend:
    ///
    /// @code
    /// engine.load_model("nav_a", "/models/nav.py");
    /// engine.load_model("nav_b", "/models/nav.py");  // instância separada
    /// @endcode
    ///
    /// @param model_id       Identificador único do modelo neste motor.
    /// @param model_path     Caminho absoluto ou relativo do arquivo.
    /// @param force_backend  Backend a usar; `BACKEND_UNKNOWN` ativa a
    ///                       detecção automática por extensão (padrão).
    /// @param backend_config Parâmetros opcionais passados ao backend na carga
    ///                       (chave-valor; semântica definida por cada backend).
    ///
    /// @return @c true se o modelo foi carregado com sucesso.
    ///         @c false se o arquivo não existir, o ID já estiver em uso ou
    ///         o backend falhar na inicialização.
    bool load_model(
        const std::string& model_id,
        const std::string& model_path,
        common::BackendType force_backend = common::BACKEND_UNKNOWN,
        const std::map<std::string, std::string>& backend_config = {});

    /// @brief Remove um modelo da memória do motor.
    ///
    /// @param model_id  ID do modelo carregado.
    ///
    /// @return @c true se o modelo existia e foi removido.
    ///         @c false se o ID não for encontrado.
    bool unload_model(const std::string& model_id);

    /// @brief Verifica se um modelo está carregado.
    ///
    /// @param model_id  ID a consultar.
    ///
    /// @return @c true se o modelo estiver no mapa interno.
    bool is_model_loaded(const std::string& model_id) const;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Inferência
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Executa inferência em um modelo carregado.
    ///
    /// @details
    /// Adquire o mutex, localiza o backend pelo `model_id` e delega a chamada
    /// para `ModelBackend::predict()`.  O backend interpreta o #client::Object
    /// conforme sua implementação:
    ///
    /// - **PythonBackend:** converte o `Object` para `dict` Python e chama
    ///   `model.predict(inputs)`.
    /// - **OnnxBackend:** extrai arrays de primeiro nível como tensores planos.
    ///
    /// @param model_id  ID do modelo previamente carregado.
    /// @param inputs    Mapa de entradas (nome → #client::Value).
    ///
    /// @return #InferenceResult com os outputs e metadados.
    ///         `success == false` se o ID não existir ou o backend falhar.
    InferenceResult predict(const std::string& model_id,
                            const client::Object& inputs);

    /// @}

    // -------------------------------------------------------------------------
    /// @name Introspecção
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Lista os IDs de todos os modelos carregados.
    ///
    /// @return Vetor de strings com os IDs ativos no momento da chamada.
    std::vector<std::string> get_loaded_model_ids() const;

    /// @brief Retorna o protobuf `ModelInfo` completo de um modelo carregado.
    ///
    /// @details
    /// Converte o #ModelSchema retornado por `ModelBackend::get_schema()` para
    /// o formato protobuf `common::ModelInfo`, incluindo tensores de I/O,
    /// metadados de autoria e timestamp de carga.
    ///
    /// @param model_id  ID do modelo carregado.
    ///
    /// @return `common::ModelInfo` preenchido.  Se o ID não existir, todos os
    ///         campos estarão com valores default (strings vazias, zeros).
    common::ModelInfo get_model_info(const std::string& model_id) const;

    /// @brief Retorna ponteiro para as métricas de runtime de um modelo.
    ///
    /// @details
    /// O ponteiro é válido enquanto o modelo permanecer carregado.
    /// Não adquira o mutex externamente — a leitura de campos individuais de
    /// #RuntimeMetrics é segura para leitores concorrentes por serem
    /// `uint64_t` e `double` alinhados.
    ///
    /// @param model_id  ID do modelo carregado.
    ///
    /// @return Ponteiro const para #RuntimeMetrics, ou @c nullptr se o
    ///         modelo não estiver carregado.
    const RuntimeMetrics* get_model_metrics(const std::string& model_id) const;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Aquecimento (Warmup)
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Resultado de uma operação de aquecimento do motor.
    struct WarmupResult {
        /// @brief @c true se todas as execuções completaram sem erro.
        bool success = false;

        /// @brief Número de execuções efetivamente concluídas.
        uint32_t runs_completed = 0;

        /// @brief Tempo médio por execução de aquecimento, em milissegundos.
        double avg_time_ms = 0.0;

        /// @brief Tempo mínimo observado, em milissegundos.
        double min_time_ms = 0.0;

        /// @brief Tempo máximo observado, em milissegundos.
        double max_time_ms = 0.0;

        /// @brief Mensagem de erro quando #success é @c false.
        std::string error_message;
    };

    /// @brief Aquece um modelo com execuções de inferência sintética.
    ///
    /// @details
    /// Gera dados de entrada aleatórios com base no #ModelSchema do modelo
    /// (usando `std::mt19937` com semente 42 para reprodutibilidade) e executa
    /// @p num_runs inferências para:
    /// - Pré-compilar kernels JIT do ONNX Runtime.
    /// - Preencher caches de alocação de tensores.
    /// - Estabelecer a baseline de latência nos #RuntimeMetrics.
    ///
    /// Para modelos Python com inputs `structured = true`, o warmup injeta
    /// um #client::Object vazio, pois a estrutura exata depende do modelo.
    ///
    /// @param model_id  ID do modelo carregado.
    /// @param num_runs  Número de execuções de aquecimento.
    ///                  Se zero, é ajustado internamente para 5.
    ///
    /// @return #WarmupResult com estatísticas das execuções.
    WarmupResult warmup_model(const std::string& model_id,
                              uint32_t num_runs = 5);

    /// @}

    // -------------------------------------------------------------------------
    /// @name Validação
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Resultado da validação estática de um arquivo de modelo.
    struct ValidationResult {
        /// @brief @c true se o arquivo é válido e pode ser carregado.
        bool valid = false;

        /// @brief Backend detectado (ou forçado) para o arquivo.
        common::BackendType backend = common::BACKEND_UNKNOWN;

        /// @brief Mensagem de erro quando #valid é @c false.
        std::string error_message;

        /// @brief Tensores de entrada reportados pelo modelo (quando disponíveis).
        ///
        /// Preenchido apenas se o backend conseguir carregar e extrair o schema
        /// sem erros.  Pode estar vazio mesmo quando #valid é @c true.
        std::vector<TensorSpecData> inputs;

        /// @brief Tensores de saída reportados pelo modelo (quando disponíveis).
        std::vector<TensorSpecData> outputs;

        /// @brief Avisos não-fatais (ex.: schema ausente, versão depreciada).
        std::vector<std::string> warnings;
    };

    /// @brief Valida um arquivo de modelo sem carregá-lo no mapa de modelos.
    ///
    /// @details
    /// Cria uma instância temporária do backend, chama `validate()` (verificação
    /// estática: existência e extensão) e, em seguida, `load()` + `get_schema()`
    /// + `unload()` para extrair o schema sem registrar o modelo.
    ///
    /// @param model_path    Caminho do arquivo a validar.
    /// @param force_backend Backend a usar; `BACKEND_UNKNOWN` ativa detecção
    ///                      automática por extensão (padrão).
    ///
    /// @return #ValidationResult com resultado e schema (quando disponível).
    ValidationResult validate_model(
        const std::string& model_path,
        common::BackendType force_backend = common::BACKEND_UNKNOWN) const;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Informações do motor
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Configuração e capacidades do motor de inferência.
    struct EngineInfo {
        /// @brief Indica se o suporte a GPU está habilitado.
        bool gpu_enabled;

        /// @brief Índice do dispositivo CUDA em uso.
        uint32_t gpu_device_id;

        /// @brief Número de threads de inferência configuradas.
        uint32_t num_threads;

        /// @brief Lista dos backends registrados (ex.: `"python"`, `"onnx"`).
        std::vector<std::string> supported_backends;
    };

    /// @brief Retorna as configurações e capacidades do motor.
    ///
    /// @return Referência const para #EngineInfo preenchida na construção.
    const EngineInfo& get_engine_info() const { return engine_info_; }

    /// @}

private:
    /// @cond INTERNAL

    /// @brief Registro de um modelo carregado no motor.
    struct LoadedModel {
        std::string model_id;  ///< ID registrado na carga.
        std::string path;      ///< Caminho absoluto do arquivo.
        std::unique_ptr<ModelBackend> backend;  ///< Backend ativo.
    };

    /// @brief Mutex que protege `models_` para acesso concorrente.
    mutable std::mutex mutex_;

    /// @brief Mapa de modelos carregados, indexado por model_id.
    std::map<std::string, std::unique_ptr<LoadedModel>> models_;

    bool     gpu_enabled_;
    uint32_t gpu_device_id_;
    uint32_t num_threads_;

    EngineInfo engine_info_;

    /// @brief Timestamp de inicialização do motor.
    std::chrono::steady_clock::time_point start_time_;

    /// @brief Placeholder para política de descarga automática por LRU/TTL.
    ///
    /// Atualmente sem implementação — reservado para extensões futuras.
    void check_auto_unload();

    /// @brief Converte #ModelSchema + metadados para `common::ModelInfo` protobuf.
    static common::ModelInfo schema_to_proto(
        const std::string& model_id,
        const std::string& path,
        const ModelSchema& schema,
        common::BackendType backend_type,
        int64_t memory_bytes,
        bool is_warmed_up,
        int64_t loaded_at_unix);

    /// @endcond
};

}  // namespace inference
}  // namespace miia

#endif  // ML_INFERENCE_INFERENCE_ENGINE_HPP
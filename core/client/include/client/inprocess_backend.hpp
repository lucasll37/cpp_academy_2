// =============================================================================
/// @file   inprocess_backend.hpp
/// @brief  Backend de transporte in-process do cliente Miia.
///
/// @details
/// Implementa #IClientBackend executando o #InferenceEngine **diretamente
/// no processo do cliente**, sem camada de rede ou serialização gRPC.
/// É a opção recomendada para integração em simuladores como o ASA/MIXR,
/// onde cliente e motor de inferência rodam no mesmo processo.
///
/// ### Comparação com GrpcClientBackend
///
/// | Aspecto              | InProcessBackend            | GrpcClientBackend         |
/// |----------------------|-----------------------------|---------------------------|
/// | Transporte           | Chamada de função direta    | gRPC sobre TCP            |
/// | Serialização         | Nenhuma                     | protobuf                  |
/// | Latência             | Mínima (sem rede)           | RTT de rede               |
/// | Escalabilidade       | Um processo                 | Múltiplos clientes/workers |
/// | Seleção automática   | `"inprocess"` / `"local"`   | `"host:porta"`            |
///
/// ### Ciclo de vida
/// ```
/// InferenceClient client("inprocess");
/// client.connect();      // cria InferenceEngine no processo atual
/// client.load_model("nav", "/models/nav.py");
/// auto r = client.predict("nav", inputs);
/// ```
///
/// ### Resolução de caminhos
/// Todos os caminhos passados a `load_model()` e `validate_model()` são
/// normalizados para absolutos via `std::filesystem::weakly_canonical()`
/// antes de serem passados ao #InferenceEngine.  Caminhos relativos são
/// resolvidos em relação ao diretório de trabalho atual.
///
/// ### Limitações conhecidas
/// - `get_metrics()` retorna um #ServerMetrics vazio — métricas detalhadas
///   por modelo estão disponíveis via `InferenceEngine::get_model_metrics()`,
///   mas não são expostas pelo backend in-process por ora.
/// - `worker_id` fixo em `"inprocess"`.
///
/// @see mlinference::client::GrpcClientBackend
/// @see mlinference::inference::InferenceEngine
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_INPROCESS_BACKEND_HPP
#define ML_INFERENCE_INPROCESS_BACKEND_HPP

#include "client/i_client_backend.hpp"
#include "inference/inference_engine.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace mlinference {
namespace client {

using inference::InferenceEngine;

// =============================================================================
// InProcessBackend
// =============================================================================

/// @brief Implementação in-process de #IClientBackend.
///
/// @details
/// Instancia um #InferenceEngine no construtor de `connect()` e delega todas
/// as operações diretamente a ele, sem conversão de tipos — o #client::Object
/// é repassado ao motor sem serialização.
///
/// Marcado `final` — não deve ser herdado.
class InProcessBackend final : public IClientBackend {
public:
    // -------------------------------------------------------------------------
    /// @name Construção
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Constrói o backend.
    ///
    /// @param  O parâmetro de target é aceito mas ignorado — o backend é sempre
    ///         in-process, independente do valor passado.  Existe apenas para
    ///         uniformidade com a assinatura dos outros backends.
    explicit InProcessBackend(const std::string& /*ignored*/) {}

    /// @}

    // -------------------------------------------------------------------------
    /// @name Conexão
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Inicializa o #InferenceEngine no processo atual.
    ///
    /// @details
    /// Cria uma instância de `InferenceEngine` com GPU desabilitado,
    /// device 0 e 4 threads.  Registra o `start_time_` para cálculo
    /// de uptime em `get_status()`.
    ///
    /// @note Diferente do #GrpcClientBackend, esta operação nunca falha
    ///       por razões de rede — sempre retorna @c true salvo erro de
    ///       inicialização do motor (ex.: falha ao registrar backends).
    ///
    /// @return @c true sempre que o motor for criado com sucesso.
    bool connect() override;

    /// @brief Retorna @c true se `connect()` foi chamado com sucesso.
    bool is_connected() const override { return connected_; }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Ciclo de vida dos modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Carrega um modelo no #InferenceEngine local.
    ///
    /// @details
    /// Normaliza `model_path` para caminho absoluto via
    /// `std::filesystem::weakly_canonical()` antes de passar ao motor.
    /// O parâmetro `version` é aceito mas ignorado pelo motor.
    ///
    /// @param model_id    Identificador único do modelo.
    /// @param model_path  Caminho do arquivo (relativo ou absoluto).
    /// @param version     Ignorado — aceito por uniformidade de interface.
    ///
    /// @return @c true se o motor carregou o modelo com sucesso.
    bool load_model(const std::string& model_id,
                    const std::string& model_path,
                    const std::string& version) override;

    /// @brief Remove um modelo do #InferenceEngine local.
    ///
    /// @param model_id  ID do modelo a descarregar.
    ///
    /// @return @c true se o modelo existia e foi removido.
    bool unload_model(const std::string& model_id) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Inferência
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Executa inferência delegando diretamente ao #InferenceEngine.
    ///
    /// @details
    /// O #client::Object é repassado sem conversão — nenhuma serialização
    /// ocorre.  O resultado do motor é convertido de #InferenceResult para
    /// #PredictionResult copiando os campos correspondentes.
    ///
    /// @param model_id  ID do modelo carregado.
    /// @param inputs    Mapa de entradas.
    ///
    /// @return #PredictionResult com outputs e tempo de execução.
    PredictionResult predict(const std::string& model_id,
                             const Object& inputs) override;

    /// @brief Executa inferência em lote chamando `predict()` sequencialmente.
    ///
    /// @details
    /// Itera sobre `batch_inputs` e chama `predict()` para cada elemento.
    /// Não há paralelismo interno — as chamadas são sequenciais na ordem
    /// do vetor de entrada.
    ///
    /// @param model_id    ID do modelo carregado.
    /// @param batch_inputs Vetor de entradas.
    ///
    /// @return Vetor de #PredictionResult na mesma ordem de `batch_inputs`.
    std::vector<PredictionResult> batch_predict(
        const std::string& model_id,
        const std::vector<Object>& batch_inputs) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Introspecção
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Lista todos os modelos carregados no motor local.
    ///
    /// @return Vetor de #ModelInfo convertidos de `common::ModelInfo` protobuf.
    std::vector<ModelInfo> list_models() override;

    /// @brief Retorna os metadados de um modelo específico.
    ///
    /// @param id  ID do modelo carregado.
    ///
    /// @return #ModelInfo preenchido; campos default se o ID não existir.
    ModelInfo get_model_info(const std::string& id) override;

    /// @brief Valida um arquivo de modelo sem carregá-lo.
    ///
    /// @details
    /// Normaliza o caminho antes de passar ao motor.  O resultado
    /// `backend` é convertido de enum protobuf para string
    /// (`"onnx"`, `"python"`, `"unknown"`).
    ///
    /// @param path  Caminho do arquivo a validar.
    ///
    /// @return #ValidationResult com resultado e schema (quando disponível).
    ValidationResult validate_model(const std::string& path) override;

    /// @brief Aquece um modelo com inferências sintéticas.
    ///
    /// @param id        ID do modelo carregado.
    /// @param num_runs  Número de execuções de aquecimento.
    ///
    /// @return #WarmupResult com estatísticas das execuções.
    WarmupResult warmup_model(const std::string& id,
                              uint32_t num_runs) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Observabilidade
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Retorna @c true se `connect()` foi chamado com sucesso.
    ///
    /// @note Para o backend in-process, health check é equivalente a
    ///       verificar `connected_` — não há rede para testar.
    bool health_check() override;

    /// @brief Retorna o estado operacional do motor local.
    ///
    /// @details
    /// Campos preenchidos:
    /// - `worker_id`: sempre `"inprocess"`.
    /// - `uptime_seconds`: segundos desde `connect()`.
    /// - `loaded_models`: IDs dos modelos ativos no motor.
    /// - `supported_backends`: backends registrados no #BackendRegistry.
    ///
    /// @return #WorkerStatus com estado atual do motor.
    WorkerStatus get_status() override;

    /// @brief Retorna métricas de desempenho.
    ///
    /// @note Retorna um #ServerMetrics vazio — métricas por modelo não são
    ///       expostas por este backend.  Use `InferenceEngine::get_model_metrics()`
    ///       diretamente se precisar de granularidade por modelo em modo in-process.
    ///
    /// @return #ServerMetrics com todos os campos zerados.
    ServerMetrics get_metrics() override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Descoberta de modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Lista arquivos de modelo disponíveis em um diretório.
    ///
    /// @details
    /// Varre `directory` (ou `./models` se vazio) com
    /// `std::filesystem::directory_iterator` e filtra arquivos `.py` e `.onnx`.
    /// Para cada arquivo, verifica se já está carregado comparando o path
    /// com `InferenceEngine::get_model_info(id).model_path()`.
    ///
    /// @param directory  Diretório a varrer; usa `./models` se vazio.
    ///
    /// @return Vetor de #AvailableModel.  Vazio se o diretório não existir.
    std::vector<AvailableModel> list_available_models(
        const std::string& directory) override;

    /// @}

private:
    /// @cond INTERNAL

    bool connected_ = false;

    /// @brief Motor de inferência local — criado em `connect()`.
    std::unique_ptr<InferenceEngine> engine_;

    /// @brief Timestamp de quando `connect()` foi chamado (para uptime).
    std::chrono::steady_clock::time_point start_time_;

    /// @brief IDs dos modelos carregados (espelho de `engine_->get_loaded_model_ids()`).
    ///
    /// Mantido separadamente para evitar chamada ao motor em caminhos críticos.
    std::vector<std::string> loaded_models_;

    /// @brief Converte `common::ModelInfo` protobuf para #ModelInfo público.
    ///
    /// Mapeia o enum `common::BackendType` para string (`"onnx"`, `"python"`,
    /// `"unknown"`) e copia tensores de I/O e tags.
    static ModelInfo proto_to_model_info(const common::ModelInfo& p);

    /// @endcond
};

}  // namespace client
}  // namespace mlinference

#endif  // ML_INFERENCE_INPROCESS_BACKEND_HPP
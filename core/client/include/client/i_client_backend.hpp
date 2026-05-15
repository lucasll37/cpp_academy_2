// =============================================================================
/// @file   i_client_backend.hpp
/// @brief  Interface interna de backend de transporte do cliente MiiaClient.
///
/// @details
/// Define o contrato que todos os backends de transporte do #InferenceClient
/// devem implementar.  É o equivalente, na camada cliente, do que o
/// #ModelBackend é na camada de inferência: uma interface puramente abstrata
/// que desacopla o cliente público do mecanismo de transporte concreto.
///
/// ### Implementações concretas
///
/// | Classe                | Selecionada quando                              |
/// |-----------------------|-------------------------------------------------|
/// | #GrpcClientBackend    | Target é `"host:porta"` (ex.: `"localhost:50052"`) |
/// | #InProcessBackend     | Target é `"inprocess"`, `"in_process"` ou `"local"` |
///
/// A seleção ocorre no construtor de #InferenceClient com base na string
/// passada pelo usuário — nenhuma outra interação com esta interface é
/// necessária pelo código de aplicação.
///
/// ### Adicionando um novo backend de transporte
/// Implemente esta interface e adicione a detecção em `inference_client.cpp`:
/// @code
/// // inference_client.cpp
/// InferenceClient::InferenceClient(const std::string& target) {
///     if (is_inprocess(target))
///         backend_ = std::make_unique<InProcessBackend>(target);
///     else if (meu_criterio(target))
///         backend_ = std::make_unique<MeuBackend>(target);
///     else
///         backend_ = std::make_unique<GrpcClientBackend>(target);
/// }
/// @endcode
///
/// @note Este header **não faz parte da API pública** exposta via
///       `libmiia_client`.  É incluído apenas por
///       `grpc_client_backend.cpp`, `inprocess_backend.cpp` e
///       `inference_client.cpp`.
///
/// @see miia::client::InferenceClient
/// @see miia::client::GrpcClientBackend
/// @see miia::client::InProcessBackend
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_I_CLIENT_BACKEND_HPP
#define ML_INFERENCE_I_CLIENT_BACKEND_HPP

#include "client/inference_client.hpp"

namespace miia {
namespace client {

// =============================================================================
// IClientBackend
// =============================================================================

/// @brief Interface interna de backend de transporte do #InferenceClient.
///
/// @details
/// O #InferenceClient delega todas as operações para uma instância de
/// `IClientBackend` armazenada em `backend_` (via `std::unique_ptr`).
/// A interface espelha exatamente a API pública do #InferenceClient,
/// garantindo que qualquer backend concreto seja substituível sem
/// alterações no código de aplicação (*Liskov Substitution Principle*).
///
/// Instâncias **não** precisam ser thread-safe — o #InferenceClient não
/// realiza chamadas concorrentes ao mesmo backend.
class IClientBackend {
public:
    /// @brief Destrutor virtual — necessário para destruição polimórfica correta.
    virtual ~IClientBackend() = default;

    // -------------------------------------------------------------------------
    /// @name Conexão
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Estabelece a conexão com o backend de transporte.
    ///
    /// @details
    /// - **GrpcClientBackend:** abre o canal gRPC e valida com `HealthCheck`.
    /// - **InProcessBackend:** inicializa o #InferenceEngine embutido.
    ///
    /// @return @c true se a conexão foi estabelecida com sucesso.
    virtual bool connect() = 0;

    /// @brief Verifica se o backend está conectado.
    ///
    /// @return @c true se `connect()` foi chamado com sucesso.
    virtual bool is_connected() const = 0;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Ciclo de vida dos modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Carrega um modelo no backend.
    ///
    /// @param model_id   Identificador único do modelo.
    /// @param model_path Caminho do arquivo do modelo.
    /// @param version    Rótulo de versão (informativo).
    ///
    /// @return @c true se o modelo foi carregado com sucesso.
    virtual bool load_model(const std::string& model_id,
                            const std::string& model_path,
                            const std::string& version) = 0;

    /// @brief Remove um modelo do backend.
    ///
    /// @param model_id  ID do modelo a descarregar.
    ///
    /// @return @c true se o modelo existia e foi removido.
    virtual bool unload_model(const std::string& model_id) = 0;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Inferência
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Executa inferência em um modelo carregado.
    ///
    /// @param model_id  ID do modelo carregado.
    /// @param inputs    Mapa de entradas (nome → #Value).
    ///
    /// @return #PredictionResult com outputs e metadados.
    virtual PredictionResult predict(const std::string& model_id,
                                     const Object& inputs) = 0;

    /// @brief Executa inferência em lote.
    ///
    /// @param model_id    ID do modelo carregado.
    /// @param batch_inputs Vetor de objetos de entrada.
    ///
    /// @return Vetor de #PredictionResult na mesma ordem de @p batch_inputs.
    virtual std::vector<PredictionResult> batch_predict(
        const std::string& model_id,
        const std::vector<Object>& batch_inputs) = 0;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Introspecção
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Lista todos os modelos carregados no backend.
    ///
    /// @return Vetor de #ModelInfo com metadados de cada modelo.
    virtual std::vector<ModelInfo> list_models() = 0;

    /// @brief Retorna os metadados de um modelo específico.
    ///
    /// @param id  ID do modelo carregado.
    ///
    /// @return #ModelInfo preenchido; campos default se o ID não existir.
    virtual ModelInfo get_model_info(const std::string& id) = 0;

    /// @brief Valida um arquivo de modelo sem carregá-lo.
    ///
    /// @param path  Caminho do arquivo a validar.
    ///
    /// @return #ValidationResult com resultado e schema (quando disponível).
    virtual ValidationResult validate_model(const std::string& path) = 0;

    /// @brief Aquece um modelo com inferências sintéticas.
    ///
    /// @param id        ID do modelo carregado.
    /// @param num_runs  Número de execuções de aquecimento.
    ///
    /// @return #WarmupResult com estatísticas das execuções.
    virtual WarmupResult warmup_model(const std::string& id,
                                      uint32_t num_runs) = 0;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Observabilidade
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Verifica se o backend está operacional.
    ///
    /// @return @c true se o backend respondeu ao ping de saúde.
    virtual bool health_check() = 0;

    /// @brief Retorna o estado operacional atual do backend.
    ///
    /// @return #WorkerStatus com contadores e lista de modelos carregados.
    virtual WorkerStatus get_status() = 0;

    /// @brief Retorna métricas de desempenho globais e por modelo.
    ///
    /// @return #ServerMetrics com estatísticas acumuladas.
    virtual ServerMetrics get_metrics() = 0;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Descoberta de modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Lista os arquivos de modelo disponíveis em um diretório.
    ///
    /// @param directory  Caminho do diretório a varrer.
    ///
    /// @return Vetor de #AvailableModel descrevendo cada arquivo encontrado.
    virtual std::vector<AvailableModel> list_available_models(
        const std::string& directory) = 0;

    /// @}
};

}  // namespace client
}  // namespace miia

#endif  // ML_INFERENCE_I_CLIENT_BACKEND_HPP
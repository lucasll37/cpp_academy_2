// =============================================================================
// grpc_client_backend.hpp — Backend gRPC do cliente AsaMiia
//
// Implementa IClientBackend via chamadas gRPC ao WorkerService.
// Converte client::Object ↔ google.protobuf.Struct na fronteira.
//
// Conectividade: o estado do canal é consultado diretamente via
// channel_->GetState() — sem flag connected_ estático.  A detecção de
// queda do servidor acontece assim que o canal gRPC atualiza seu estado
// interno (keepalive ou falha de envio), sem custo de RPC extra.
// =============================================================================

#pragma once

#include "client/i_client_backend.hpp"
#include "client/inference_client.hpp"

#include <grpcpp/grpcpp.h>
#include "server.grpc.pb.h"
#include "common.pb.h"

#include <memory>
#include <string>

namespace mlinference {
namespace client {

// =============================================================================
// GrpcClientBackend
// =============================================================================

/// @brief Implementação de `IClientBackend` sobre transporte gRPC.
///
/// @details
/// Mantém um `grpc::Channel` e um stub `WorkerService::Stub` e traduz cada
/// método da interface para uma chamada RPC unária ao servidor `AsaMiia`.
/// Inputs e outputs são serializados como `google::protobuf::Struct` via
/// `value_convert.hpp`.
///
/// **Conectividade fidedigna:** em vez de um flag booleano `connected_`
/// que ficaria desatualizado após a conexão inicial, o estado do canal é
/// consultado em tempo real via `channel_->GetState()`.  O método privado
/// `is_channel_ready()` encapsula essa consulta e é chamado pelos guards
/// de pré-condição em cada RPC.
///
/// Marcado `final` — não deve ser herdado.
class GrpcClientBackend final : public IClientBackend {
public:
    // -------------------------------------------------------------------------
    /// @name Construção
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Constrói o backend com o endereço do servidor.
    ///
    /// @details
    /// Apenas armazena o endereço — o canal gRPC e o stub são criados
    /// somente em `connect()`.
    ///
    /// @param address  Endereço do servidor no formato `"host:porta"`
    ///                 (ex.: `"localhost:50052"`, `"0.0.0.0:50052"`).
    explicit GrpcClientBackend(const std::string& address)
        : server_address_(address) {}

    /// @}

    // -------------------------------------------------------------------------
    /// @name Conexão
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Cria o canal gRPC e verifica a acessibilidade do servidor.
    ///
    /// @details
    /// Executa os seguintes passos:
    /// 1. `grpc::CreateChannel(address, InsecureChannelCredentials)`.
    /// 2. `WorkerService::NewStub(channel)`.
    /// 3. `HealthCheck` com deadline de **5 segundos**.
    ///
    /// @note O canal usa `InsecureChannelCredentials` — sem TLS.
    ///       Adequado para redes fechadas; substitua por `SslCredentials`
    ///       em ambientes que exijam segurança de transporte.
    ///
    /// @return @c true se o servidor respondeu ao health check com sucesso.
    ///         @c false se o servidor estiver inacessível ou o deadline expirar.
    bool connect() override;

    /// @brief Retorna @c true se o canal gRPC está em estado operacional.
    ///
    /// @details
    /// Consulta `channel_->GetState()` em tempo real — reflete quedas do
    /// servidor sem custo de RPC extra.  Retorna @c false se `connect()`
    /// ainda não foi chamado.
    bool is_connected() const override { return is_channel_ready(); }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Ciclo de vida dos modelos
    /// @{
    // -------------------------------------------------------------------------

    bool load_model(const std::string& model_id,
                    const std::string& model_path,
                    const std::string& version) override;

    bool unload_model(const std::string& model_id) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Inferência
    /// @{
    // -------------------------------------------------------------------------

    PredictionResult predict(const std::string& model_id,
                             const Object& inputs) override;

    std::vector<PredictionResult> batch_predict(
        const std::string& model_id,
        const std::vector<Object>& batch_inputs) override;

    WarmupResult warmup_model(const std::string& model_id,
                              uint32_t           num_runs) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Introspecção
    /// @{
    // -------------------------------------------------------------------------

    std::vector<ModelInfo>  list_models()                              override;
    ModelInfo               get_model_info(const std::string& id)      override;
    ValidationResult        validate_model(const std::string& path)    override;
    WorkerStatus            get_status()                               override;
    ServerMetrics           get_metrics()                              override;

    std::vector<AvailableModel> list_available_models(
        const std::string& directory) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Observabilidade
    /// @{
    // -------------------------------------------------------------------------

    bool health_check() override;

    /// @}

private:
    // -------------------------------------------------------------------------
    // Helpers privados
    // -------------------------------------------------------------------------

    /// @brief Consulta o estado atual do canal gRPC sem custo de RPC.
    ///
    /// @details
    /// `GRPC_CHANNEL_IDLE` e `GRPC_CHANNEL_READY` são considerados
    /// operacionais.  `GRPC_CHANNEL_CONNECTING`, `GRPC_CHANNEL_TRANSIENT_FAILURE`
    /// e `GRPC_CHANNEL_SHUTDOWN` retornam @c false.
    ///
    /// Passa `try_to_connect = false` para não forçar reconexão aqui —
    /// o gRPC reconecta automaticamente quando o próximo RPC é enviado.
    bool is_channel_ready() const;

    static std::string   dtype_str(common::DataType dt);
    static ModelInfo     proto_to_model_info(const common::ModelInfo& p);

    // -------------------------------------------------------------------------
    // Estado
    // -------------------------------------------------------------------------

    std::string                                          server_address_;
    std::shared_ptr<grpc::Channel>                       channel_;   ///< Canal gRPC (criado em connect()).
    std::unique_ptr<server::WorkerService::Stub>         stub_;      ///< Stub gerado pelo protoc.
    bool                                                 ever_connected_ = false;
};

}  // namespace client
}  // namespace mlinference
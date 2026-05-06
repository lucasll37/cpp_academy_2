// =============================================================================
/// @file   worker_server.hpp
/// @brief  Servidor gRPC AsaMiia — serviço de inferência e wrapper de ciclo de vida.
///
/// @details
/// Este header define duas classes que compõem a camada de servidor do sistema:
///
/// - **#WorkerServiceImpl** — implementa todos os RPCs definidos em
///   `server.proto`, delegando operações ao #InferenceEngine e convertendo
///   tipos protobuf ↔ C++ via `value_convert.hpp`.
///
/// - **#WorkerServer** — dono do `WorkerServiceImpl` e do `grpc::Server`.
///   Gerencia o ciclo de vida completo: construção, escuta bloqueante e
///   desligamento gracioso.
///
/// ### Arquitetura
/// ```
/// main()
///   └── WorkerServer::run()          ← bloqueia em grpc::Server::Wait()
///         └── WorkerServiceImpl      ← implementa WorkerService::Service
///               └── InferenceEngine  ← motor de inferência thread-safe
/// ```
///
/// ### Uso típico (em main.cpp)
/// @code
/// #include <server/worker_server.hpp>
/// using mlinference::server::WorkerServer;
///
/// WorkerServer server("worker-1", "0.0.0.0:50052",
///                     /*gpu=*/false, /*threads=*/8, "./models");
///
/// std::signal(SIGINT,  [](int){ server.stop(); });
/// std::signal(SIGTERM, [](int){ server.stop(); });
///
/// server.run();  // bloqueia até stop() ser chamado
/// @endcode
///
/// ### Segurança de concorrência
/// O gRPC cria uma thread por RPC em andamento.  Cada chamada ao
/// `WorkerServiceImpl` é thread-safe porque:
/// - Os contadores de requisição são `std::atomic`.
/// - O `InferenceEngine` protege seu estado interno com `mutex_`.
///
/// @see mlinference::inference::InferenceEngine
/// @see mlinference::client::InferenceClient
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_WORKER_SERVER_HPP
#define ML_INFERENCE_WORKER_SERVER_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include <cerrno>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include "server.grpc.pb.h"
#include "inference/inference_engine.hpp"

namespace mlinference {

/// @namespace mlinference::server
/// @brief     Servidor gRPC e implementação dos RPCs do AsaMiia.
namespace server {

using inference::InferenceEngine;
using inference::RuntimeMetrics;

// =============================================================================
// WorkerServiceImpl
// =============================================================================

/// @brief Implementação concreta dos RPCs do serviço `WorkerService`.
///
/// @details
/// Herda de `WorkerService::Service` (gerado pelo protoc) e sobrescreve
/// todos os métodos virtuais.  Cada método RPC:
///
/// 1. Incrementa `active_requests_` e `total_requests_` (atomicamente).
/// 2. Converte inputs protobuf → `client::Object` via `from_proto_struct()`.
/// 3. Delega ao `InferenceEngine`.
/// 4. Converte outputs `client::Object` → protobuf via `to_proto_struct()`.
/// 5. Decrementa `active_requests_` e atualiza `successful_requests_` ou
///    `failed_requests_`.
///
/// ### Thread-safety
/// Instâncias são compartilhadas entre threads do servidor gRPC.  Os
/// contadores são `std::atomic`; o `InferenceEngine` serializa internamente
/// via `mutex_`.
class WorkerServiceImpl final
    : public mlinference::server::WorkerService::Service {
public:
    // -------------------------------------------------------------------------
    /// @name Construção e destruição
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Constrói o serviço e inicializa o #InferenceEngine.
    ///
    /// @param worker_id   Identificador textual do worker (exibido em logs e
    ///                    no campo `worker_id` de `GetStatusResponse`).
    /// @param enable_gpu  Habilita CUDA EP no ONNX Runtime.
    /// @param num_threads Número de threads de inferência no pool interno.
    /// @param models_dir  Diretório base para `ListAvailableModels` quando
    ///                    a requisição não especifica um diretório.
    explicit WorkerServiceImpl(
        const std::string& worker_id,
        bool               enable_gpu  = false,
        uint32_t           num_threads = 4,
        const std::string& models_dir  = "./models");

    ~WorkerServiceImpl() override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name RPCs de inferência
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Executa inferência unitária em um modelo carregado.
    ///
    /// @details
    /// Desserializa `request->inputs()` (`google::protobuf::Struct`) para
    /// `client::Object`, chama `InferenceEngine::predict()` e serializa
    /// `result.outputs` de volta para `google::protobuf::Struct` na resposta.
    ///
    /// @param context   Contexto gRPC (prazo, metadados, cancelamento).
    /// @param request   Contém `model_id` e `inputs` (Struct).
    /// @param response  Preenchido com `success`, `outputs` e `inference_time_ms`.
    ///
    /// @return `grpc::Status::OK` sempre — erros de inferência são reportados
    ///         em `response->success = false` e `response->error_message`.
    grpc::Status Predict(
        grpc::ServerContext*  context,
        const PredictRequest* request,
        PredictResponse*      response) override;

    /// @brief Executa inferência em modo streaming bidirecional.
    ///
    /// @details
    /// Lê `PredictRequest` do stream de entrada e escreve `PredictResponse`
    /// no stream de saída, em loop, até o cliente fechar o canal.
    /// Útil para cenários de alta frequência onde o overhead de abertura
    /// de chamada gRPC por tick seria significativo.
    ///
    /// @param context  Contexto gRPC.
    /// @param stream   Stream bidirecional de request/response.
    ///
    /// @return `grpc::Status::OK` ao encerrar normalmente.
    grpc::Status PredictStream(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<PredictResponse, PredictRequest>* stream) override;

    /// @brief Executa inferência em lote.
    ///
    /// @details
    /// Itera sobre `request->inputs()` e chama `InferenceEngine::predict()`
    /// para cada elemento, agregando os resultados em `response->results()`.
    ///
    /// @param context   Contexto gRPC.
    /// @param request   Contém `model_id` e vetor de `inputs`.
    /// @param response  Preenchido com vetor de `PredictResponse`.
    ///
    /// @return `grpc::Status::OK` sempre.
    grpc::Status BatchPredict(
        grpc::ServerContext*        context,
        const BatchPredictRequest*  request,
        BatchPredictResponse*       response) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name RPCs de ciclo de vida dos modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Carrega um modelo no #InferenceEngine.
    ///
    /// @param context   Contexto gRPC.
    /// @param request   Contém `model_id`, `model_path` e `version`.
    /// @param response  Preenchido com `success` e `error_message`.
    ///
    /// @return `grpc::Status::OK` sempre; erros em `response->success`.
    grpc::Status LoadModel(
        grpc::ServerContext*    context,
        const LoadModelRequest* request,
        LoadModelResponse*      response) override;

    /// @brief Remove um modelo do #InferenceEngine.
    ///
    /// @param context   Contexto gRPC.
    /// @param request   Contém `model_id`.
    /// @param response  Preenchido com `success` e `error_message`.
    ///
    /// @return `grpc::Status::OK` sempre.
    grpc::Status UnloadModel(
        grpc::ServerContext*      context,
        const UnloadModelRequest* request,
        UnloadModelResponse*      response) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name RPCs de introspecção de modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Lista todos os modelos carregados no motor.
    ///
    /// @return `grpc::Status::OK` com vetor de `ModelInfo` em `response`.
    grpc::Status ListModels(
        grpc::ServerContext*     context,
        const ListModelsRequest* request,
        ListModelsResponse*      response) override;

    /// @brief Retorna os metadados completos de um modelo carregado.
    ///
    /// @return `grpc::Status::OK`; campos default se o ID não existir.
    grpc::Status GetModelInfo(
        grpc::ServerContext*       context,
        const GetModelInfoRequest* request,
        GetModelInfoResponse*      response) override;

    /// @brief Valida um arquivo de modelo sem carregá-lo.
    ///
    /// @details
    /// Cria uma instância temporária do backend, executa validação estática
    /// e tenta extrair o schema sem registrar o modelo.
    ///
    /// @return `grpc::Status::OK` com `valid` e schema (quando disponível).
    grpc::Status ValidateModel(
        grpc::ServerContext*        context,
        const ValidateModelRequest* request,
        ValidateModelResponse*      response) override;

    /// @brief Aquece um modelo com inferências sintéticas.
    ///
    /// @return `grpc::Status::OK` com estatísticas do warmup.
    grpc::Status WarmupModel(
        grpc::ServerContext*      context,
        const WarmupModelRequest* request,
        WarmupModelResponse*      response) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name RPCs de observabilidade
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Retorna o estado operacional do worker.
    ///
    /// @details
    /// Agrega contadores atômicos locais (`total_requests_`,
    /// `successful_requests_`, etc.), lista de modelos carregados e
    /// capacidades do #InferenceEngine.
    ///
    /// @return `grpc::Status::OK` com `GetStatusResponse` preenchido.
    grpc::Status GetStatus(
        grpc::ServerContext*    context,
        const GetStatusRequest* request,
        GetStatusResponse*      response) override;

    /// @brief Retorna métricas de desempenho globais e por modelo.
    ///
    /// @details
    /// Itera sobre os modelos carregados, consulta `InferenceEngine::get_model_metrics()`
    /// e serializa #RuntimeMetrics (avg, min, max, p95, p99) em
    /// `per_model_metrics` da resposta.
    ///
    /// @return `grpc::Status::OK` com `GetMetricsResponse` preenchido.
    grpc::Status GetMetrics(
        grpc::ServerContext*     context,
        const GetMetricsRequest* request,
        GetMetricsResponse*      response) override;

    /// @brief Responde ao ping de saúde do cliente.
    ///
    /// @details
    /// Sempre retorna `healthy = true` enquanto o servidor estiver ativo.
    /// Usado pelo `GrpcClientBackend::connect()` para verificar acessibilidade.
    ///
    /// @return `grpc::Status::OK` com `healthy = true`.
    grpc::Status HealthCheck(
        grpc::ServerContext*      context,
        const HealthCheckRequest* request,
        HealthCheckResponse*      response) override;

    /// @}

    // -------------------------------------------------------------------------
    /// @name RPCs de descoberta de modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Lista os arquivos de modelo disponíveis em um diretório.
    ///
    /// @details
    /// Varre o diretório especificado em `request->directory()` (ou
    /// `models_dir_` se vazio) com `std::filesystem::directory_iterator`
    /// e informa, para cada arquivo suportado (`.py`, `.onnx`), se ele
    /// já está carregado no motor e sob qual `model_id`.
    ///
    /// @return `grpc::Status::OK` com lista de `AvailableModel`.
    ///         `grpc::StatusCode::NOT_FOUND` se o diretório não existir.
    grpc::Status ListAvailableModels(
        grpc::ServerContext*               context,
        const ListAvailableModelsRequest*  request,
        ListAvailableModelsResponse*       response) override;

    /// @}

private:
    /// @cond INTERNAL

    std::string                          worker_id_;
    std::string                          models_dir_;
    std::shared_ptr<InferenceEngine>     inference_engine_;
    std::chrono::steady_clock::time_point start_time_;

    /// @brief Total de requisições recebidas (incluindo falhas e ativas).
    std::atomic<uint64_t> total_requests_{0};

    /// @brief Requisições concluídas com sucesso.
    std::atomic<uint64_t> successful_requests_{0};

    /// @brief Requisições que resultaram em erro de inferência ou protocolo.
    std::atomic<uint64_t> failed_requests_{0};

    /// @brief Requisições atualmente em processamento.
    std::atomic<uint32_t> active_requests_{0};

    /// @brief Copia campos de `common::ModelInfo` src para dst.
    void fill_model_info(const common::ModelInfo& src,
                         common::ModelInfo*       dst) const;

    /// @brief Serializa #RuntimeMetrics para `ModelRuntimeMetrics` protobuf.
    void fill_runtime_metrics(const RuntimeMetrics& src,
                              ModelRuntimeMetrics*  dst) const;

    /// @brief Constrói o `common::WorkerMetrics` a partir dos contadores atômicos.
    common::WorkerMetrics build_worker_metrics() const;

    /// @endcond
};

// =============================================================================
// WorkerServer
// =============================================================================

/// @brief Dono do serviço gRPC e do servidor — gerencia o ciclo de vida completo.
///
/// @details
/// Instancia um #WorkerServiceImpl e um `grpc::Server`, registra o serviço
/// e bloqueia em `run()` até `stop()` ser chamado (tipicamente por um handler
/// de sinal SIGINT/SIGTERM).
///
/// ### Ciclo de vida
/// ```
/// WorkerServer server(...);
/// server.run();   // bloqueia aqui
/// // stop() chamado por signal_handler
/// // destrutor chama stop() automaticamente se necessário
/// ```
///
/// @note `run()` é bloqueante — deve ser chamado da thread principal.
///       `stop()` é seguro para chamar de handlers de sinal.
class WorkerServer {
public:
    // -------------------------------------------------------------------------
    /// @name Construção e destruição
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Constrói o servidor e inicializa o serviço interno.
    ///
    /// @details
    /// Cria um #WorkerServiceImpl com os parâmetros fornecidos.  O servidor
    /// gRPC (`grpc::Server`) é criado apenas em `run()`, quando
    /// `grpc::ServerBuilder::BuildAndStart()` é chamado.
    ///
    /// @param worker_id      Identificador textual do worker.
    /// @param server_address Endereço de escuta (ex.: `"0.0.0.0:50052"`).
    /// @param enable_gpu     Habilita CUDA EP no ONNX Runtime.
    /// @param num_threads    Número de threads de inferência.
    /// @param models_dir     Diretório base de modelos locais.
    WorkerServer(
        const std::string& worker_id,
        const std::string& server_address,
        bool               enable_gpu  = false,
        uint32_t           num_threads = 4,
        const std::string& models_dir  = "./models");

    /// @brief Destrói o servidor, chamando `stop()` se ainda estiver ativo.
    ~WorkerServer();

    /// @}

    // -------------------------------------------------------------------------
    /// @name Operação
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Inicia o servidor gRPC e bloqueia até `stop()` ser chamado.
    ///
    /// @details
    /// Usa `grpc::ServerBuilder` para registrar o serviço e chamar
    /// `BuildAndStart()`.  Se a porta não puder ser aberta, imprime erro
    /// em `stderr` e retorna imediatamente.
    ///
    /// Bloqueia em `grpc::Server::Wait()` — retorna apenas após `stop()`.
    void run();

    /// @brief Encerra o servidor graciosamente.
    ///
    /// @details
    /// Chama `grpc::Server::Shutdown()`, que sinaliza todas as threads
    /// de RPC em andamento para concluírem e desbloqueia `Wait()` em `run()`.
    /// É seguro chamar múltiplas vezes — chamadas adicionais são no-ops.
    ///
    /// @note Projetado para ser chamado de handlers de sinal (`SIGINT`, `SIGTERM`).
    void stop();

    /// @}

private:
    /// @cond INTERNAL
    std::string                            worker_id_;
    std::string                            server_address_;
    std::unique_ptr<WorkerServiceImpl>     service_;
    std::unique_ptr<grpc::Server>          server_;
    /// @endcond
};

}  // namespace server
}  // namespace mlinference

#endif  // ML_INFERENCE_WORKER_SERVER_HPP
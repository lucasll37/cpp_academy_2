#ifndef ML_INFERENCE_WORKER_SERVER_HPP
#define ML_INFERENCE_WORKER_SERVER_HPP

#include <memory>
#include <string>
#include <atomic>
#include <chrono>
#include <grpcpp/grpcpp.h>
#include "worker.grpc.pb.h"
#include "inference_engine.hpp"

namespace mlinference {
namespace worker {

// ============================================
// gRPC Service Implementation
// ============================================

class WorkerServiceImpl final : public mlinference::worker::WorkerService::Service {
public:
    explicit WorkerServiceImpl(
        const std::string& worker_id,
        bool enable_gpu = false,
        uint32_t num_threads = 4,
        const std::string& models_dir = "./models");
    
    ~WorkerServiceImpl() override;
    
    // --- Inference ---
    grpc::Status Predict(
        grpc::ServerContext* context,
        const PredictRequest* request,
        PredictResponse* response) override;
    
    grpc::Status PredictStream(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<PredictResponse, PredictRequest>* stream) override;
    
    grpc::Status BatchPredict(
        grpc::ServerContext* context,
        const BatchPredictRequest* request,
        BatchPredictResponse* response) override;
    
    // --- Model lifecycle ---
    grpc::Status LoadModel(
        grpc::ServerContext* context,
        const LoadModelRequest* request,
        LoadModelResponse* response) override;
    
    grpc::Status UnloadModel(
        grpc::ServerContext* context,
        const UnloadModelRequest* request,
        UnloadModelResponse* response) override;
    
    // --- Model introspection ---
    grpc::Status ListModels(
        grpc::ServerContext* context,
        const ListModelsRequest* request,
        ListModelsResponse* response) override;
    
    grpc::Status GetModelInfo(
        grpc::ServerContext* context,
        const GetModelInfoRequest* request,
        GetModelInfoResponse* response) override;
    
    grpc::Status ValidateModel(
        grpc::ServerContext* context,
        const ValidateModelRequest* request,
        ValidateModelResponse* response) override;
    
    grpc::Status WarmupModel(
        grpc::ServerContext* context,
        const WarmupModelRequest* request,
        WarmupModelResponse* response) override;
    
    // --- Worker observability ---
    grpc::Status GetStatus(
        grpc::ServerContext* context,
        const GetStatusRequest* request,
        GetStatusResponse* response) override;
    
    grpc::Status GetMetrics(
        grpc::ServerContext* context,
        const GetMetricsRequest* request,
        GetMetricsResponse* response) override;
    
    grpc::Status HealthCheck(
        grpc::ServerContext* context,
        const HealthCheckRequest* request,
        HealthCheckResponse* response) override;

    // --- File discovery ---
    grpc::Status ListAvailableModels(
        grpc::ServerContext* context,
        const ListAvailableModelsRequest* request,
        ListAvailableModelsResponse* response) override;

private:
    std::string worker_id_;
    std::string models_dir_;
    std::shared_ptr<InferenceEngine> inference_engine_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Atomic counters for worker-level metrics
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> successful_requests_{0};
    std::atomic<uint64_t> failed_requests_{0};
    std::atomic<uint32_t> active_requests_{0};
    
    // Helpers
    void fill_model_info(const common::ModelInfo& src,
                         common::ModelInfo* dst) const;
    
    void fill_runtime_metrics(const RuntimeMetrics& src,
                              ModelRuntimeMetrics* dst) const;
    
    common::WorkerMetrics build_worker_metrics() const;
};

// ============================================
// Server (owns the service + gRPC server)
// ============================================

class WorkerServer {
public:
    WorkerServer(const std::string& worker_id,
                 const std::string& server_address,
                 bool enable_gpu = false,
                 uint32_t num_threads = 4,
                 const std::string& models_dir = "./models");
    
    ~WorkerServer();
    
    void run();   // Blocking
    void stop();

private:
    std::string worker_id_;
    std::string server_address_;
    std::unique_ptr<WorkerServiceImpl> service_;
    std::unique_ptr<grpc::Server> server_;
};

}  // namespace worker
}  // namespace mlinference

#endif  // ML_INFERENCE_WORKER_SERVER_HPP
// =============================================================================
// inference_client.hpp — Public API of the AsaMiia client library
// =============================================================================

#ifndef ML_INFERENCE_CLIENT_HPP
#define ML_INFERENCE_CLIENT_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <grpcpp/grpcpp.h>
#include "worker.grpc.pb.h"
#include "common.pb.h"

namespace mlinference {
namespace client {

// ============================================
// Result Structs
// ============================================

struct PredictionResult {
    bool success = false;
    std::map<std::string, std::vector<float>> outputs;
    double inference_time_ms = 0.0;
    std::string error_message;
};

struct ModelInfo {
    std::string model_id;
    std::string version;
    std::string backend;       // "onnx", "python", etc.
    std::string description;
    std::string author;
    int64_t memory_usage_bytes = 0;
    bool is_warmed_up = false;

    struct TensorSpec {
        std::string name;
        std::vector<int64_t> shape;
        std::string dtype;
        std::string description;
    };

    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
    std::map<std::string, std::string> tags;
};

struct ValidationResult {
    bool valid = false;
    std::string backend;
    std::vector<std::string> warnings;
    std::string error_message;
    std::vector<ModelInfo::TensorSpec> inputs;
    std::vector<ModelInfo::TensorSpec> outputs;
};

struct WarmupResult {
    bool success = false;
    uint32_t runs_completed = 0;
    double avg_time_ms = 0.0;
    double min_time_ms = 0.0;
    double max_time_ms = 0.0;
    std::string error_message;
};

struct WorkerStatus {
    std::string worker_id;
    uint64_t total_requests       = 0;
    uint64_t successful_requests  = 0;
    uint64_t failed_requests      = 0;
    uint32_t active_requests      = 0;
    int64_t  uptime_seconds       = 0;
    std::vector<std::string> loaded_models;
    std::vector<std::string> supported_backends;
};

// ============================================
// Server-side metrics structs
// ============================================

/// Runtime statistics for a single model, as accumulated by the worker
/// since it was loaded.  These values reflect ALL callers (interactive
/// client, application libraries, direct gRPC calls, etc.).
struct ModelMetrics {
    uint64_t total_inferences  = 0;
    uint64_t failed_inferences = 0;
    double   avg_ms            = 0.0;
    double   min_ms            = 0.0;
    double   max_ms            = 0.0;
    double   p95_ms            = 0.0;
    double   p99_ms            = 0.0;
    double   total_time_ms     = 0.0;
    int64_t  last_used_at_unix = 0;   // 0 = never used
    int64_t  loaded_at_unix    = 0;
};

/// Aggregated view of the worker, obtained via GetMetrics RPC.
/// Unlike WorkerStatus (which is a point-in-time snapshot), this struct
/// carries the full per-model breakdown suitable for statistical display.
struct ServerMetrics {
    // Worker-level counters
    uint64_t total_requests      = 0;
    uint64_t successful_requests = 0;
    uint64_t failed_requests     = 0;
    uint32_t active_requests     = 0;
    int64_t  uptime_seconds      = 0;

    // Per-model statistics keyed by model_id
    std::map<std::string, ModelMetrics> per_model;
};

// ============================================
// Available model (file discovery)
// ============================================

struct AvailableModel {
    std::string filename;
    std::string path;
    std::string extension;
    std::string backend;
    int64_t file_size_bytes = 0;
    bool is_loaded = false;
    std::string loaded_as;
};

// ============================================
// Client
// ============================================

class InferenceClient {
public:
    explicit InferenceClient(const std::string& server_address);
    ~InferenceClient();

    // --- Connection ---
    bool connect();
    bool is_connected() const;

    // --- Model lifecycle ---
    bool load_model(const std::string& model_id,
                    const std::string& model_path,
                    const std::string& version = "1.0.0");

    bool unload_model(const std::string& model_id);

    // --- Inference ---
    PredictionResult predict(const std::string& model_id,
                             const std::map<std::string, std::vector<float>>& inputs);

    std::vector<PredictionResult> batch_predict(
        const std::string& model_id,
        const std::vector<std::map<std::string, std::vector<float>>>& batch_inputs);

    // --- Introspection ---
    std::vector<ModelInfo> list_models();
    ModelInfo get_model_info(const std::string& model_id);
    ValidationResult validate_model(const std::string& model_path);
    WarmupResult warmup_model(const std::string& model_id, uint32_t num_runs = 5);

    // --- Observability ---
    bool health_check();
    WorkerStatus get_status();

    /// Returns server-side accumulated inference metrics for all loaded
    /// models.  Covers every caller, not just this client instance.
    ServerMetrics get_metrics();

    // --- File discovery ---
    std::vector<AvailableModel> list_available_models(const std::string& directory = "");

private:
    std::string server_address_;
    bool connected_ = false;

    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<mlinference::worker::WorkerService::Stub> stub_;

    // Tensor serialisation helpers
    static void set_tensor_data(common::Tensor* tensor,
                                const std::vector<float>& data);

    static std::vector<float> get_tensor_data(const common::Tensor& tensor);

    // Proto → client struct converters
    static ModelInfo proto_to_model_info(const common::ModelInfo& proto);
    static std::string backend_type_to_string(common::BackendType type);
    static std::string dtype_to_string(common::DataType type);
};

}  // namespace client
}  // namespace mlinference

#endif  // ML_INFERENCE_CLIENT_HPP
#ifndef ML_INFERENCE_INFERENCE_ENGINE_HPP
#define ML_INFERENCE_INFERENCE_ENGINE_HPP

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <mutex>
#include <chrono>
#include "common.pb.h"
#include "model_backend.hpp"

namespace mlinference {
namespace worker {

/// High-level facade that manages multiple loaded models, each backed by
/// the appropriate ModelBackend (selected via BackendRegistry).
///
/// Thread-safe: all public methods acquire a mutex.
///
/// This class replaces the old monolithic InferenceEngine that was
/// tightly coupled to ONNX Runtime.  The ONNX-specific logic now
/// lives in OnnxBackend.
class InferenceEngine {
public:
    explicit InferenceEngine(bool enable_gpu = false,
                             uint32_t gpu_device_id = 0,
                             uint32_t num_threads = 4);
    ~InferenceEngine();
    
    // ---- Model lifecycle ----
    
    /// Load a model.  The backend is auto-detected from the file extension
    /// unless `force_backend` is set.
    bool load_model(const std::string& model_id,
                    const std::string& model_path,
                    common::BackendType force_backend = common::BACKEND_UNKNOWN,
                    const std::map<std::string, std::string>& backend_config = {});
    
    bool unload_model(const std::string& model_id);
    
    bool is_model_loaded(const std::string& model_id) const;
    
    // ---- Inference ----
    
    InferenceResult predict(const std::string& model_id,
                            const std::map<std::string, std::vector<float>>& inputs);
    
    // ---- Introspection ----
    
    std::vector<std::string> get_loaded_model_ids() const;
    
    /// Get the full ModelInfo protobuf for a loaded model.
    common::ModelInfo get_model_info(const std::string& model_id) const;
    
    /// Get per-model runtime metrics.
    const RuntimeMetrics* get_model_metrics(const std::string& model_id) const;
    
    // ---- Validation (pre-load) ----
    
    struct ValidationResult {
        bool valid = false;
        common::BackendType backend = common::BACKEND_UNKNOWN;
        std::vector<std::string> warnings;
        std::string error_message;
        ModelSchema schema;  // Populated if backend can preview without full load
    };
    
    ValidationResult validate_model(
        const std::string& model_path,
        common::BackendType force_backend = common::BACKEND_UNKNOWN) const;
    
    // ---- Warmup ----
    
    struct WarmupResult {
        bool success = false;
        uint32_t runs_completed = 0;
        double avg_time_ms = 0.0;
        double min_time_ms = 0.0;
        double max_time_ms = 0.0;
        std::string error_message;
    };
    
    WarmupResult warmup_model(const std::string& model_id, uint32_t num_runs = 5);
    
    // ---- Engine info ----
    
    struct EngineInfo {
        bool gpu_enabled;
        uint32_t num_threads;
        std::vector<std::string> supported_backends;
    };
    
    EngineInfo get_engine_info() const;
    
    // ---- Auto-unload ----
    
    void enable_auto_unload(uint32_t idle_timeout_seconds);
    void disable_auto_unload();

private:
    struct LoadedModel {
        std::string model_id;
        std::string path;
        std::string version;
        std::unique_ptr<ModelBackend> backend;
    };
    
    mutable std::mutex mutex_;
    std::map<std::string, std::unique_ptr<LoadedModel>> models_;
    
    bool enable_gpu_;
    uint32_t gpu_device_id_;
    uint32_t num_threads_;
    
    bool auto_unload_enabled_ = false;
    uint32_t idle_timeout_seconds_ = 0;
    
    void check_auto_unload();  // Called periodically
};

}  // namespace worker
}  // namespace mlinference

#endif  // ML_INFERENCE_INFERENCE_ENGINE_HPP
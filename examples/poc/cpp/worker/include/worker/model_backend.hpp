// =============================================================================
// model_backend.hpp — Abstract backend interface + RuntimeMetrics
// =============================================================================

#ifndef ML_INFERENCE_MODEL_BACKEND_HPP
#define ML_INFERENCE_MODEL_BACKEND_HPP

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <chrono>
#include <limits>
#include "common.pb.h"

namespace mlinference {
namespace worker {

// ============================================
// Data Structures
// ============================================

/// Result of a single inference call.
struct InferenceResult {
    bool success = false;
    std::map<std::string, std::vector<float>> outputs;
    double inference_time_ms = 0.0;
    std::string error_message;
};

/// Specification of a single tensor I/O port.
struct TensorSpecData {
    std::string name;
    std::vector<int64_t> shape;       // -1 for dynamic
    common::DataType dtype = common::FLOAT32;
    std::string description;
    double min_value = 0.0;
    double max_value = 0.0;
    bool has_constraints = false;
};

/// Complete schema returned by a backend after loading.
struct ModelSchema {
    std::vector<TensorSpecData> inputs;
    std::vector<TensorSpecData> outputs;
    std::string description;
    std::string author;
    std::map<std::string, std::string> tags;
};

/// Accumulated per-model runtime metrics.
///
/// Design notes:
///   - min_time_ms is initialised to 0.0 ("not yet observed") instead of
///     +infinity.  The record() method handles the "first sample" case
///     explicitly.  This prevents +inf (or its serialised approximation)
///     from appearing as a large negative value on the client side after
///     protobuf double → float conversion or sign-bit corruption.
///
///   - latency_samples is a circular buffer capped at LATENCY_WINDOW
///     entries; it feeds the p95 / p99 nearest-rank calculations.
struct RuntimeMetrics {
    uint64_t total_inferences   = 0;
    uint64_t failed_inferences  = 0;
    double   total_time_ms      = 0.0;
    double   min_time_ms        = 0.0;   // 0 = "not yet observed"
    double   max_time_ms        = 0.0;

    // Circular buffer for percentile estimation (last N successful samples).
    static constexpr size_t LATENCY_WINDOW = 1000;
    std::vector<double> latency_samples;
    size_t latency_write_pos = 0;

    void record(double time_ms, bool success) {
        total_inferences++;
        if (!success) {
            failed_inferences++;
            return;
        }

        // Clamp: a negative measurement is physically impossible and
        // indicates a clock skew / timer wrap — treat it as zero.
        if (time_ms < 0.0) time_ms = 0.0;

        total_time_ms += time_ms;

        uint64_t ok = total_inferences - failed_inferences;
        if (ok == 1) {
            // First successful sample — initialise min/max.
            min_time_ms = time_ms;
            max_time_ms = time_ms;
        } else {
            if (time_ms < min_time_ms) min_time_ms = time_ms;
            if (time_ms > max_time_ms) max_time_ms = time_ms;
        }

        if (latency_samples.size() < LATENCY_WINDOW) {
            latency_samples.push_back(time_ms);
        } else {
            latency_samples[latency_write_pos % LATENCY_WINDOW] = time_ms;
        }
        latency_write_pos++;
    }

    double avg_time_ms() const {
        uint64_t ok = total_inferences - failed_inferences;
        return (ok > 0) ? total_time_ms / static_cast<double>(ok) : 0.0;
    }

    /// 95th percentile latency (nearest-rank, sorted copy of the window).
    double p95_time_ms() const;

    /// 99th percentile latency (nearest-rank, sorted copy of the window).
    double p99_time_ms() const;
};

// ============================================
// Abstract Backend Interface
// ============================================

/// Every model backend (ONNX, Python, ...) must implement this interface.
/// Instances are **not** thread-safe; the InferenceEngine serialises access.
class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    // --- Lifecycle ---

    /// Load the model from `path`. `config` carries backend-specific options.
    virtual bool load(const std::string& path,
                      const std::map<std::string, std::string>& config) = 0;

    /// Release all resources held by this model.
    virtual void unload() = 0;

    // --- Inference ---

    /// Run inference.  Inputs and outputs are flat float vectors; shape
    /// information lives in the schema.
    virtual InferenceResult predict(
        const std::map<std::string, std::vector<float>>& inputs) = 0;

    // --- Introspection ---

    /// Return the I/O schema.  Available after `load()`.
    virtual ModelSchema get_schema() const = 0;

    /// Return the backend type enum.
    virtual common::BackendType backend_type() const = 0;

    /// Estimate memory footprint in bytes (best-effort).
    virtual int64_t memory_usage_bytes() const { return 0; }

    // --- Warmup ---

    /// Perform `n` dummy inferences to warm JIT / caches.
    /// Default implementation generates random data based on schema.
    virtual void warmup(uint32_t n);

    // --- Validation (static, pre-load) ---

    /// Check if `path` looks like a valid model for this backend
    /// without fully loading it.  Returns empty string on success,
    /// or an error description.
    virtual std::string validate(const std::string& path) const { return ""; }

    // --- State ---

    bool is_loaded() const { return loaded_; }

    std::chrono::steady_clock::time_point load_time() const { return load_time_; }
    std::chrono::steady_clock::time_point last_used() const { return last_used_; }
    void touch() { last_used_ = std::chrono::steady_clock::now(); }

    RuntimeMetrics& metrics()             { return metrics_; }
    const RuntimeMetrics& metrics() const { return metrics_; }

protected:
    bool loaded_ = false;
    std::chrono::steady_clock::time_point load_time_;
    std::chrono::steady_clock::time_point last_used_;
    RuntimeMetrics metrics_;
};

// ============================================
// Backend Factory
// ============================================

/// Factory that creates ModelBackend instances.
/// Each backend registers a factory in the BackendRegistry.
class BackendFactory {
public:
    virtual ~BackendFactory() = default;
    virtual std::unique_ptr<ModelBackend> create() const = 0;
    virtual common::BackendType backend_type() const = 0;
    virtual std::string name() const = 0;
};

}  // namespace worker
}  // namespace mlinference

#endif  // ML_INFERENCE_MODEL_BACKEND_HPP
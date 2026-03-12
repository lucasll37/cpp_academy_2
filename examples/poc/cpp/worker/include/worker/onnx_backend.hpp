#ifndef ML_INFERENCE_ONNX_BACKEND_HPP
#define ML_INFERENCE_ONNX_BACKEND_HPP

#include <memory>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "model_backend.hpp"

namespace mlinference {
namespace worker {

/// ONNX Runtime backend.
///
/// Loads .onnx files, extracts full I/O metadata (names, shapes, dtypes),
/// handles dynamic dimensions, and runs inference via ONNX Runtime.
class OnnxBackend : public ModelBackend {
public:
    /// @param enable_gpu   Attempt CUDA execution provider.
    /// @param gpu_device   CUDA device id (ignored if !enable_gpu).
    /// @param num_threads  Intra-op thread count for ONNX Runtime.
    explicit OnnxBackend(bool enable_gpu = false,
                         uint32_t gpu_device = 0,
                         uint32_t num_threads = 4);
    
    ~OnnxBackend() override;
    
    // --- ModelBackend interface ---
    
    bool load(const std::string& path,
              const std::map<std::string, std::string>& config) override;
    
    void unload() override;
    
    InferenceResult predict(
        const std::map<std::string, std::vector<float>>& inputs) override;
    
    ModelSchema get_schema() const override;
    
    common::BackendType backend_type() const override {
        return common::BACKEND_ONNX;
    }
    
    int64_t memory_usage_bytes() const override;
    
    std::string validate(const std::string& path) const override;

private:
    // ONNX Runtime objects
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    
    // Cached metadata (populated on load)
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    std::vector<ONNXTensorElementDataType> input_dtypes_;
    std::vector<ONNXTensorElementDataType> output_dtypes_;
    
    std::string model_path_;
    
    // Config
    bool enable_gpu_;
    uint32_t gpu_device_;
    uint32_t num_threads_;
    
    // Helpers
    void extract_metadata();
    
    /// Convert ONNX Runtime element type to our proto DataType.
    static common::DataType ort_to_proto_dtype(ONNXTensorElementDataType ort_type);
    
    /// Resolve dynamic dimensions in `shape` given `data_size` elements.
    /// Returns empty vector on error (populates `error_msg`).
    static std::vector<int64_t> resolve_dynamic_shape(
        const std::vector<int64_t>& shape,
        size_t data_size,
        const std::string& tensor_name,
        std::string& error_msg);
};

// ============================================
// Factory
// ============================================

class OnnxBackendFactory : public BackendFactory {
public:
    OnnxBackendFactory(bool enable_gpu = false,
                       uint32_t gpu_device = 0,
                       uint32_t num_threads = 4)
        : enable_gpu_(enable_gpu)
        , gpu_device_(gpu_device)
        , num_threads_(num_threads) {}
    
    std::unique_ptr<ModelBackend> create() const override {
        return std::make_unique<OnnxBackend>(enable_gpu_, gpu_device_, num_threads_);
    }
    
    common::BackendType backend_type() const override {
        return common::BACKEND_ONNX;
    }
    
    std::string name() const override { return "onnx"; }

private:
    bool enable_gpu_;
    uint32_t gpu_device_;
    uint32_t num_threads_;
};

}  // namespace worker
}  // namespace mlinference

#endif  // ML_INFERENCE_ONNX_BACKEND_HPP
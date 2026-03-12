#include "worker/onnx_backend.hpp"
#include <iostream>
#include <fstream>
#include <numeric>
#include <functional>

namespace mlinference {
namespace worker {

// ============================================
// Construction / Destruction
// ============================================

OnnxBackend::OnnxBackend(bool enable_gpu, uint32_t gpu_device, uint32_t num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "MiiaOnnxBackend")
    , enable_gpu_(enable_gpu)
    , gpu_device_(gpu_device)
    , num_threads_(num_threads) {
    
    // Configure session options once; reused for every model loaded by this backend.
    session_options_.SetIntraOpNumThreads(num_threads_);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    if (enable_gpu_) {
#ifdef USE_CUDA
        try {
            OrtCUDAProviderOptions cuda_opts;
            cuda_opts.device_id = gpu_device_;
            session_options_.AppendExecutionProvider_CUDA(cuda_opts);
            std::cout << "[OnnxBackend] CUDA execution provider enabled (device "
                      << gpu_device_ << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[OnnxBackend] WARNING: CUDA init failed: " << e.what()
                      << " — falling back to CPU" << std::endl;
        }
#else
        std::cerr << "[OnnxBackend] WARNING: GPU requested but not compiled with CUDA"
                  << std::endl;
#endif
    }
}

OnnxBackend::~OnnxBackend() {
    if (loaded_) unload();
}

// ============================================
// Lifecycle
// ============================================

bool OnnxBackend::load(const std::string& path,
                       const std::map<std::string, std::string>& /*config*/) {
    if (loaded_) {
        std::cerr << "[OnnxBackend] Already loaded. Unload first." << std::endl;
        return false;
    }
    
    try {
#ifdef _WIN32
        std::wstring wide(path.begin(), path.end());
        session_ = std::make_unique<Ort::Session>(env_, wide.c_str(), session_options_);
#else
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), session_options_);
#endif
        model_path_ = path;
        extract_metadata();
        
        loaded_ = true;
        load_time_ = std::chrono::steady_clock::now();
        last_used_ = load_time_;
        
        std::cout << "[OnnxBackend] Loaded: " << path << std::endl;
        // std::cout << "  Inputs (" << input_names_.size() << "):" << std::endl;
        // for (size_t i = 0; i < input_names_.size(); ++i) {
        //     std::cout << "    " << input_names_[i] << " [";
        //     for (size_t j = 0; j < input_shapes_[i].size(); ++j) {
        //         if (j > 0) std::cout << ", ";
        //         std::cout << input_shapes_[i][j];
        //     }
        //     std::cout << "] dtype=" << input_dtypes_[i] << std::endl;
        // }
        // std::cout << "  Outputs (" << output_names_.size() << "):" << std::endl;
        // for (size_t i = 0; i < output_names_.size(); ++i) {
        //     std::cout << "    " << output_names_[i] << " [";
        //     for (size_t j = 0; j < output_shapes_[i].size(); ++j) {
        //         if (j > 0) std::cout << ", ";
        //         std::cout << output_shapes_[i][j];
        //     }
        //     std::cout << "] dtype=" << output_dtypes_[i] << std::endl;
        // }
        
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[OnnxBackend] ONNX Runtime error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[OnnxBackend] Error: " << e.what() << std::endl;
        return false;
    }
}

void OnnxBackend::unload() {
    session_.reset();
    input_names_.clear();
    output_names_.clear();
    input_shapes_.clear();
    output_shapes_.clear();
    input_dtypes_.clear();
    output_dtypes_.clear();
    model_path_.clear();
    loaded_ = false;
}

// ============================================
// Inference
// ============================================

InferenceResult OnnxBackend::predict(
    const std::map<std::string, std::vector<float>>& inputs) {
    
    if (!loaded_) {
        return {false, {}, 0.0, "Model not loaded"};
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_name_ptrs;
        
        for (size_t i = 0; i < input_names_.size(); ++i) {
            const auto& name = input_names_[i];
            
            auto it = inputs.find(name);
            if (it == inputs.end()) {
                return {false, {}, 0.0, "Missing input: " + name};
            }
            
            const auto& data = it->second;
            std::string err;
            auto actual_shape = resolve_dynamic_shape(
                input_shapes_[i], data.size(), name, err);
            
            if (actual_shape.empty()) {
                return {false, {}, 0.0, err};
            }
            
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                mem_info,
                const_cast<float*>(data.data()),
                data.size(),
                actual_shape.data(),
                actual_shape.size()));
            
            input_name_ptrs.push_back(name.c_str());
        }
        
        // Output names
        std::vector<const char*> output_name_ptrs;
        output_name_ptrs.reserve(output_names_.size());
        for (const auto& n : output_names_) output_name_ptrs.push_back(n.c_str());
        
        // Run
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs.data(), input_tensors.data(), input_tensors.size(),
            output_name_ptrs.data(), output_name_ptrs.size());
        
        // Extract outputs
        std::map<std::string, std::vector<float>> result_outputs;
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& tensor = output_tensors[i];
            auto info = tensor.GetTensorTypeAndShapeInfo();
            size_t count = info.GetElementCount();
            
            float* ptr = tensor.GetTensorMutableData<float>();
            result_outputs[output_names_[i]] = std::vector<float>(ptr, ptr + count);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        touch();
        metrics_.record(ms, true);
        
        return {true, std::move(result_outputs), ms, ""};
        
    } catch (const Ort::Exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics_.record(ms, false);
        return {false, {}, ms, std::string("ONNX Runtime error: ") + e.what()};
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics_.record(ms, false);
        return {false, {}, ms, std::string("Error: ") + e.what()};
    }
}

// ============================================
// Introspection
// ============================================

ModelSchema OnnxBackend::get_schema() const {
    ModelSchema schema;
    
    for (size_t i = 0; i < input_names_.size(); ++i) {
        TensorSpecData spec;
        spec.name  = input_names_[i];
        spec.shape = input_shapes_[i];
        spec.dtype = ort_to_proto_dtype(input_dtypes_[i]);
        schema.inputs.push_back(std::move(spec));
    }
    
    for (size_t i = 0; i < output_names_.size(); ++i) {
        TensorSpecData spec;
        spec.name  = output_names_[i];
        spec.shape = output_shapes_[i];
        spec.dtype = ort_to_proto_dtype(output_dtypes_[i]);
        schema.outputs.push_back(std::move(spec));
    }
    
    // Try to read ONNX model metadata properties
    if (session_) {
        try {
            Ort::ModelMetadata meta = session_->GetModelMetadata();
            Ort::AllocatorWithDefaultOptions alloc;
            
            auto desc = meta.GetDescriptionAllocated(alloc);
            schema.description = desc.get();
            
            auto author = meta.GetProducerNameAllocated(alloc);
            schema.author = author.get();
        } catch (...) {
            // Metadata is optional; ignore failures.
        }
    }
    
    return schema;
}

int64_t OnnxBackend::memory_usage_bytes() const {
    if (!loaded_ || model_path_.empty()) return 0;
    
    // Rough estimate: file size ≈ memory footprint (conservative lower bound).
    std::ifstream f(model_path_, std::ios::ate | std::ios::binary);
    if (!f.is_open()) return 0;
    return static_cast<int64_t>(f.tellg());
}

std::string OnnxBackend::validate(const std::string& path) const {
    // Quick validation: try to open the file and parse headers.
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return "Cannot open file: " + path;
    
    // ONNX files start with protobuf magic bytes.
    // A lightweight check: try to create a session and immediately destroy it.
    try {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        opts.SetIntraOpNumThreads(1);
        
#ifdef _WIN32
        std::wstring wide(path.begin(), path.end());
        Ort::Session test_session(env_, wide.c_str(), opts);
#else
        Ort::Session test_session(const_cast<Ort::Env&>(env_), path.c_str(), opts);
#endif
        return "";  // Valid
    } catch (const Ort::Exception& e) {
        return std::string("ONNX validation failed: ") + e.what();
    }
}

// ============================================
// Private Helpers
// ============================================

void OnnxBackend::extract_metadata() {
    Ort::AllocatorWithDefaultOptions alloc;
    
    size_t n_in = session_->GetInputCount();
    input_names_.reserve(n_in);
    input_shapes_.reserve(n_in);
    input_dtypes_.reserve(n_in);
    
    for (size_t i = 0; i < n_in; ++i) {
        auto name_ptr = session_->GetInputNameAllocated(i, alloc);
        input_names_.push_back(name_ptr.get());
        
        auto type_info  = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shapes_.push_back(tensor_info.GetShape());
        input_dtypes_.push_back(tensor_info.GetElementType());
    }
    
    size_t n_out = session_->GetOutputCount();
    output_names_.reserve(n_out);
    output_shapes_.reserve(n_out);
    output_dtypes_.reserve(n_out);
    
    for (size_t i = 0; i < n_out; ++i) {
        auto name_ptr = session_->GetOutputNameAllocated(i, alloc);
        output_names_.push_back(name_ptr.get());
        
        auto type_info  = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_shapes_.push_back(tensor_info.GetShape());
        output_dtypes_.push_back(tensor_info.GetElementType());
    }
}

common::DataType OnnxBackend::ort_to_proto_dtype(ONNXTensorElementDataType ort_type) {
    switch (ort_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return common::FLOAT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return common::FLOAT64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return common::INT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return common::INT64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return common::UINT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return common::BOOL;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return common::FLOAT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  return common::STRING;
        default: return common::FLOAT32;  // Fallback
    }
}

std::vector<int64_t> OnnxBackend::resolve_dynamic_shape(
    const std::vector<int64_t>& shape,
    size_t data_size,
    const std::string& tensor_name,
    std::string& error_msg) {
    
    std::vector<int64_t> actual = shape;
    
    int64_t dynamic_count = 0;
    int64_t static_product = 1;
    
    for (auto dim : actual) {
        if (dim == -1) {
            dynamic_count++;
        } else {
            static_product *= dim;
        }
    }
    
    if (dynamic_count > 1) {
        error_msg = "Multiple dynamic dimensions not supported for " + tensor_name;
        return {};
    }
    
    if (dynamic_count == 1) {
        if (static_product == 0) {
            error_msg = "Invalid shape for " + tensor_name
                        + ": static dimensions product is 0";
            return {};
        }
        int64_t inferred = static_cast<int64_t>(data_size) / static_product;
        for (auto& dim : actual) {
            if (dim == -1) { dim = inferred; break; }
        }
    }
    
    // Final size check
    int64_t expected = std::accumulate(
        actual.begin(), actual.end(), 1LL, std::multiplies<int64_t>());
    
    if (static_cast<int64_t>(data_size) != expected) {
        std::string shape_str = "[";
        for (size_t j = 0; j < actual.size(); ++j) {
            if (j > 0) shape_str += ", ";
            shape_str += std::to_string(actual[j]);
        }
        shape_str += "]";
        
        error_msg = "Input size mismatch for " + tensor_name
                    + ": expected " + std::to_string(expected)
                    + " (shape: " + shape_str + "), got "
                    + std::to_string(data_size);
        return {};
    }
    
    return actual;
}

}  // namespace worker
}  // namespace mlinference
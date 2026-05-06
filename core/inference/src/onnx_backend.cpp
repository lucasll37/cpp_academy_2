// =============================================================================
// onnx_backend.cpp — ONNX Runtime backend implementation
// =============================================================================

#include "inference/onnx_backend.hpp"
#include "utils/logger.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace mlinference {
namespace inference {

// =============================================================================
// Constructor / Destructor
// =============================================================================

OnnxBackend::OnnxBackend(bool enable_gpu, uint32_t gpu_device, uint32_t num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "AsaMiia")
    , enable_gpu_(enable_gpu)
    , gpu_device_(gpu_device)
    , num_threads_(num_threads) {

    LOG_DEBUG("onnx_backend") << "[ctor] OnnxBackend construído; enable_gpu=" << enable_gpu_
         << " gpu_device=" << gpu_device_ << " num_threads=" << num_threads_;

    session_options_.SetIntraOpNumThreads(static_cast<int>(num_threads_));
    LOG_DEBUG("onnx_backend") << "[ctor] SetIntraOpNumThreads(" << num_threads_ << ")";

    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    LOG_DEBUG("onnx_backend") << "[ctor] SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED)";

    if (enable_gpu_) {
        LOG_DEBUG("onnx_backend") << "[ctor] tentando habilitar CUDA provider para device=" << gpu_device_;
        try {
            OrtCUDAProviderOptions cuda_opts;
            cuda_opts.device_id = static_cast<int>(gpu_device_);
            session_options_.AppendExecutionProvider_CUDA(cuda_opts);

            LOG_INFO("onnx_backend") << "[ctor] CUDA execution provider habilitado; device=" << gpu_device_;
        } catch (const Ort::Exception& e) {
            LOG_WARN("onnx_backend") << "[ctor] CUDA indisponível, fallback para CPU; erro=" << e.what();
        }
    } else {
        LOG_DEBUG("onnx_backend") << "[ctor] GPU desabilitado, usando CPU";
    }
}

OnnxBackend::~OnnxBackend() {
    LOG_DEBUG("onnx_backend") << "[dtor] OnnxBackend destruído; loaded_=" << loaded_ << " model_path_=" << model_path_;
    if (loaded_) unload();
}

// =============================================================================
// Lifecycle
// =============================================================================

bool OnnxBackend::load(const std::string& path,
                       const std::map<std::string, std::string>& /*config*/) {
    LOG_DEBUG("onnx_backend") << "[load] chamado; path=" << path << " loaded_=" << loaded_;

    if (loaded_) {
        LOG_WARN("onnx_backend") << "[load] já carregado, unload primeiro";
        return false;
    }

    try {
        LOG_DEBUG("onnx_backend") << "[load] criando Ort::Session para path=" << path;

#ifdef _WIN32
        std::wstring wide(path.begin(), path.end());
        LOG_DEBUG("onnx_backend") << "[load] plataforma Windows: usando wstring";
        session_ = std::make_unique<Ort::Session>(env_, wide.c_str(), session_options_);
#else
        LOG_DEBUG("onnx_backend") << "[load] plataforma não-Windows: usando string";
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), session_options_);
#endif

        LOG_DEBUG("onnx_backend") << "[load] Ort::Session criada; session_=" << (void*)session_.get();

        model_path_ = path;
        LOG_DEBUG("onnx_backend") << "[load] chamando extract_metadata()";
        extract_metadata();
        LOG_DEBUG("onnx_backend") << "[load] extract_metadata() concluído; n_inputs=" << input_names_.size()
             << " n_outputs=" << output_names_.size();

        loaded_    = true;
        load_time_ = std::chrono::steady_clock::now();
        last_used_ = load_time_;

        LOG_INFO("onnx_backend") << "[load] modelo carregado: path=" << path
             << " n_inputs=" << input_names_.size() << " n_outputs=" << output_names_.size();
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("onnx_backend") << "[load] Ort::Exception ao carregar '" << path << "': " << e.what();
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("onnx_backend") << "[load] exceção ao carregar '" << path << "': " << e.what();
        return false;
    }
}

void OnnxBackend::unload() {
    LOG_DEBUG("onnx_backend") << "[unload] chamado; loaded_=" << loaded_ << " model_path_=" << model_path_;

    session_.reset();
    LOG_DEBUG("onnx_backend") << "[unload] session_ resetada";

    LOG_DEBUG("onnx_backend") << "[unload] limpando input_names_ (" << input_names_.size() << " entradas)";
    input_names_.clear();
    LOG_DEBUG("onnx_backend") << "[unload] limpando output_names_ (" << output_names_.size() << " entradas)";
    output_names_.clear();
    input_shapes_.clear();
    output_shapes_.clear();
    input_dtypes_.clear();
    output_dtypes_.clear();
    model_path_.clear();
    loaded_ = false;

    LOG_DEBUG("onnx_backend") << "[unload] concluído; todos os campos limpos";
}

// =============================================================================
// Inference
// =============================================================================

InferenceResult OnnxBackend::predict(const client::Object& inputs) {
    LOG_DEBUG("onnx_backend") << "[predict] chamado; loaded_=" << loaded_
         << " n_input_tensors_esperados=" << input_names_.size();

    if (!loaded_) {
        LOG_ERROR("onnx_backend") << "[predict] FALHA PRÉ-CONDIÇÃO: modelo não carregado";
        return {false, {}, 0.0, "Model not loaded"};
    }

    for (const auto& [k, v] : inputs) {
        LOG_DEBUG("onnx_backend") << "[predict] input recebido key='" << k << "'";
    }

    auto start = std::chrono::high_resolution_clock::now();

    try {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        LOG_DEBUG("onnx_backend") << "[predict] MemoryInfo CPU criado";

        // input_data_buffers mantém os vetores de float vivos durante o
        // session_->Run(). Ort::Value::CreateTensor NÃO copia os dados —
        // mantém referência ao buffer externo. Se o vector<float> local fosse
        // destruído no fim de cada iteração do loop, o Ort::Value ficaria com
        // ponteiro dangling e o Run() leria memória liberada (UB → outputs
        // corrompidos como 1.88e+22).
        std::vector<std::vector<float>> input_data_buffers;
        input_data_buffers.reserve(input_names_.size());

        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_name_ptrs;

        for (size_t i = 0; i < input_names_.size(); ++i) {
            const auto& name = input_names_[i];
            LOG_DEBUG("onnx_backend") << "[predict] preparando tensor[" << i << "] name='" << name << "'";

            auto it = inputs.find(name);

            if (it == inputs.end()) {
                LOG_ERROR("onnx_backend") << "[predict] tensor '" << name << "' não encontrado nos inputs fornecidos";
                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                metrics_.record(ms, false);
                return {false, {}, ms, "Missing input tensor: " + name};
            }

            // Move o vector para o buffer persistente — vive até o fim de predict().
            input_data_buffers.push_back(value_to_floats(it->second));
            auto& data = input_data_buffers.back();

            LOG_DEBUG("onnx_backend") << "[predict] tensor[" << i << "] name='" << name << "' data.size()=" << data.size();

            if (data.empty()) {
                LOG_ERROR("onnx_backend") << "[predict] tensor '" << name << "' vazio ou tipo não suportado";
                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                metrics_.record(ms, false);
                return {false, {}, ms,
                    "Input '" + name + "' is empty or has unsupported type"};
            }

            // Preview dos primeiros valores
            {
                std::ostringstream vals_oss;
                size_t preview = std::min(data.size(), size_t(8));
                vals_oss << "[";
                for (size_t j = 0; j < preview; ++j) {
                    vals_oss << data[j];
                    if (j < preview - 1) vals_oss << ", ";
                }
                if (data.size() > 8) vals_oss << ", ...";
                vals_oss << "]";
                LOG_DEBUG("onnx_backend") << "[predict] tensor[" << i << "] name='" << name
                     << "' primeiros valores=" << vals_oss.str();
            }

            std::string err;
            LOG_DEBUG("onnx_backend") << "[predict] tensor[" << i << "] shape original=" << [&]() {
                std::ostringstream o; o << "[";
                for (size_t j = 0; j < input_shapes_[i].size(); ++j) {
                    o << input_shapes_[i][j];
                    if (j + 1 < input_shapes_[i].size()) o << ", ";
                }
                o << "]"; return o.str();
            }();

            auto actual_shape = resolve_dynamic_shape(
                input_shapes_[i], data.size(), name, err);

            if (actual_shape.empty()) {
                LOG_ERROR("onnx_backend") << "[predict] resolve_dynamic_shape falhou para '" << name << "': " << err;
                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                metrics_.record(ms, false);
                return {false, {}, ms, err};
            }

            {
                std::ostringstream shape_oss;
                shape_oss << "[";
                for (size_t j = 0; j < actual_shape.size(); ++j) {
                    shape_oss << actual_shape[j];
                    if (j + 1 < actual_shape.size()) shape_oss << ", ";
                }
                shape_oss << "]";
                LOG_DEBUG("onnx_backend") << "[predict] tensor[" << i << "] name='" << name
                     << "' shape resolvido=" << shape_oss.str();
            }

            // data.data() permanece válido porque data vive em input_data_buffers.
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                mem_info,
                data.data(),
                data.size(),
                actual_shape.data(),
                actual_shape.size()));

            input_name_ptrs.push_back(name.c_str());
            LOG_DEBUG("onnx_backend") << "[predict] tensor[" << i << "] name='" << name << "' adicionado";
        }

        LOG_DEBUG("onnx_backend") << "[predict] " << input_tensors.size() << " tensores de entrada preparados";

        std::vector<const char*> output_name_ptrs;
        output_name_ptrs.reserve(output_names_.size());
        for (const auto& n : output_names_) {
            output_name_ptrs.push_back(n.c_str());
            LOG_DEBUG("onnx_backend") << "[predict] output esperado: '" << n << "'";
        }

        LOG_DEBUG("onnx_backend") << "[predict] chamando session_->Run() com " << input_tensors.size()
             << " inputs e " << output_name_ptrs.size() << " outputs";

        // Neste ponto input_data_buffers ainda está vivo — todos os ponteiros
        // dentro de input_tensors são válidos.
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs.data(), input_tensors.data(), input_tensors.size(),
            output_name_ptrs.data(), output_name_ptrs.size());

        LOG_DEBUG("onnx_backend") << "[predict] session_->Run() concluído; n_output_tensors=" << output_tensors.size();

        client::Object result_outputs;
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& tensor = output_tensors[i];
            auto info    = tensor.GetTensorTypeAndShapeInfo();
            size_t count = info.GetElementCount();
            const float* ptr = tensor.GetTensorMutableData<float>();

            {
                auto shape = info.GetShape();
                std::ostringstream shape_oss;
                shape_oss << "[";
                for (size_t j = 0; j < shape.size(); ++j) {
                    shape_oss << shape[j];
                    if (j + 1 < shape.size()) shape_oss << ", ";
                }
                shape_oss << "]";
                LOG_DEBUG("onnx_backend") << "[predict] output[" << i << "] name='" << output_names_[i]
                     << "' shape=" << shape_oss.str() << " count=" << count;
            }

            {
                std::ostringstream vals_oss;
                size_t preview = std::min(count, size_t(8));
                vals_oss << "[";
                for (size_t j = 0; j < preview; ++j) {
                    vals_oss << ptr[j];
                    if (j < preview - 1) vals_oss << ", ";
                }
                if (count > 8) vals_oss << ", ...";
                vals_oss << "]";
                LOG_DEBUG("onnx_backend") << "[predict] output[" << i << "] name='" << output_names_[i]
                     << "' primeiros valores=" << vals_oss.str();
            }

            result_outputs[output_names_[i]] = floats_to_value(ptr, count);
            LOG_DEBUG("onnx_backend") << "[predict] output[" << i << "] name='" << output_names_[i] << "' inserido";
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        LOG_DEBUG("onnx_backend") << "[predict] SUCESSO; inference_time_ms=" << ms
             << " n_outputs=" << result_outputs.size();

        touch();
        metrics_.record(ms, true);

        return {true, std::move(result_outputs), ms, ""};

    } catch (const Ort::Exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        LOG_ERROR("onnx_backend") << "[predict] Ort::Exception: " << e.what() << " ms=" << ms;
        metrics_.record(ms, false);
        return {false, {}, ms, std::string("ONNX Runtime error: ") + e.what()};
    } catch (const std::exception& e) {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        LOG_ERROR("onnx_backend") << "[predict] exceção: " << e.what() << " ms=" << ms;
        metrics_.record(ms, false);
        return {false, {}, ms, std::string("Error: ") + e.what()};
    }
}

// =============================================================================
// Introspection
// =============================================================================

ModelSchema OnnxBackend::get_schema() const {
    LOG_DEBUG("onnx_backend") << "[get_schema] chamado; n_inputs=" << input_names_.size()
         << " n_outputs=" << output_names_.size();
    ModelSchema schema;

    for (size_t i = 0; i < input_names_.size(); ++i) {
        TensorSpecData spec;
        spec.name  = input_names_[i];
        spec.shape = input_shapes_[i];
        spec.dtype = ort_to_proto_dtype(input_dtypes_[i]);

        {
            std::ostringstream shape_oss;
            shape_oss << "[";
            for (size_t j = 0; j < spec.shape.size(); ++j) {
                shape_oss << spec.shape[j];
                if (j + 1 < spec.shape.size()) shape_oss << ", ";
            }
            shape_oss << "]";
            LOG_DEBUG("onnx_backend") << "[get_schema] input[" << i << "] name='" << spec.name
                 << "' shape=" << shape_oss.str() << " dtype=" << spec.dtype;
        }

        schema.inputs.push_back(std::move(spec));
    }

    for (size_t i = 0; i < output_names_.size(); ++i) {
        TensorSpecData spec;
        spec.name  = output_names_[i];
        spec.shape = output_shapes_[i];
        spec.dtype = ort_to_proto_dtype(output_dtypes_[i]);

        {
            std::ostringstream shape_oss;
            shape_oss << "[";
            for (size_t j = 0; j < spec.shape.size(); ++j) {
                shape_oss << spec.shape[j];
                if (j + 1 < spec.shape.size()) shape_oss << ", ";
            }
            shape_oss << "]";
            LOG_DEBUG("onnx_backend") << "[get_schema] output[" << i << "] name='" << spec.name
                 << "' shape=" << shape_oss.str() << " dtype=" << spec.dtype;
        }

        schema.outputs.push_back(std::move(spec));
    }

    if (session_) {
        LOG_DEBUG("onnx_backend") << "[get_schema] extraindo metadata da sessão ONNX";
        try {
            Ort::ModelMetadata meta = session_->GetModelMetadata();
            Ort::AllocatorWithDefaultOptions alloc;
            auto desc   = meta.GetDescriptionAllocated(alloc);
            auto author = meta.GetProducerNameAllocated(alloc);
            schema.description = desc.get();
            schema.author      = author.get();
            LOG_DEBUG("onnx_backend") << "[get_schema] metadata: description='" << schema.description
                 << "' author='" << schema.author << "'";
        } catch (...) {
            LOG_DEBUG("onnx_backend") << "[get_schema] metadata indisponível (ignorado)";
        }
    } else {
        LOG_DEBUG("onnx_backend") << "[get_schema] session_ é nullptr, metadata não extraída";
    }

    return schema;
}

int64_t OnnxBackend::memory_usage_bytes() const {
    LOG_DEBUG("onnx_backend") << "[memory_usage_bytes] loaded_=" << loaded_ << " model_path_='" << model_path_ << "'";
    if (!loaded_ || model_path_.empty()) {
        LOG_DEBUG("onnx_backend") << "[memory_usage_bytes] retornando 0 (não carregado ou path vazio)";
        return 0;
    }
    std::ifstream f(model_path_, std::ios::ate | std::ios::binary);
    if (!f.is_open()) {
        LOG_WARN("onnx_backend") << "[memory_usage_bytes] não foi possível abrir arquivo '" << model_path_ << "'";
        return 0;
    }
    int64_t size = static_cast<int64_t>(f.tellg());
    LOG_DEBUG("onnx_backend") << "[memory_usage_bytes] tamanho do arquivo=" << size << " bytes";
    return size;
}

std::string OnnxBackend::validate(const std::string& path) const {
    LOG_DEBUG("onnx_backend") << "[validate] chamado; path='" << path << "'";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        LOG_ERROR("onnx_backend") << "[validate] não foi possível abrir arquivo: '" << path << "'";
        return "Cannot open file: " + path;
    }

    LOG_DEBUG("onnx_backend") << "[validate] arquivo aberto; criando sessão temporária para validação";
    try {
        Ort::Env   tmp_env(ORT_LOGGING_LEVEL_ERROR, "validate");
        Ort::SessionOptions tmp_opts;
        tmp_opts.SetIntraOpNumThreads(1);
#ifdef _WIN32
        std::wstring wide(path.begin(), path.end());
        Ort::Session tmp_session(tmp_env, wide.c_str(), tmp_opts);
#else
        Ort::Session tmp_session(tmp_env, path.c_str(), tmp_opts);
#endif
        LOG_DEBUG("onnx_backend") << "[validate] sessão temporária criada com sucesso — modelo ONNX válido";
    } catch (const Ort::Exception& e) {
        LOG_WARN("onnx_backend") << "[validate] modelo ONNX inválido; path='" << path << "' erro=" << e.what();
        return std::string("Invalid ONNX model: ") + e.what();
    }
    LOG_DEBUG("onnx_backend") << "[validate] validação concluída com sucesso; path='" << path << "'";
    return "";
}

// =============================================================================
// Private helpers
// =============================================================================

void OnnxBackend::extract_metadata() {
    LOG_DEBUG("onnx_backend") << "[extract_metadata] chamado; session_=" << (void*)session_.get();
    Ort::AllocatorWithDefaultOptions alloc;

    size_t num_inputs  = session_->GetInputCount();
    size_t num_outputs = session_->GetOutputCount();
    LOG_DEBUG("onnx_backend") << "[extract_metadata] num_inputs=" << num_inputs << " num_outputs=" << num_outputs;

    input_names_.clear();
    input_shapes_.clear();
    input_dtypes_.clear();
    output_names_.clear();
    output_shapes_.clear();
    output_dtypes_.clear();

    for (size_t i = 0; i < num_inputs; ++i) {
        auto name_ptr = session_->GetInputNameAllocated(i, alloc);
        std::string name = name_ptr.get();
        input_names_.emplace_back(name);

        auto type_info   = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType dtype = tensor_info.GetElementType();
        input_dtypes_.push_back(dtype);

        std::vector<int64_t> shape;
        for (int64_t d : tensor_info.GetShape()) shape.push_back(d);
        input_shapes_.push_back(shape);

        std::ostringstream shape_oss;
        shape_oss << "[";
        for (size_t j = 0; j < shape.size(); ++j) {
            shape_oss << shape[j];
            if (j + 1 < shape.size()) shape_oss << ", ";
        }
        shape_oss << "]";
        LOG_DEBUG("onnx_backend") << "[extract_metadata] input[" << i << "] name='" << name
             << "' dtype=" << dtype << " shape=" << shape_oss.str();
    }

    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_->GetOutputNameAllocated(i, alloc);
        std::string name = name_ptr.get();
        output_names_.emplace_back(name);

        auto type_info   = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType dtype = tensor_info.GetElementType();
        output_dtypes_.push_back(dtype);

        std::vector<int64_t> shape;
        for (int64_t d : tensor_info.GetShape()) shape.push_back(d);
        output_shapes_.push_back(shape);

        std::ostringstream shape_oss;
        shape_oss << "[";
        for (size_t j = 0; j < shape.size(); ++j) {
            shape_oss << shape[j];
            if (j + 1 < shape.size()) shape_oss << ", ";
        }
        shape_oss << "]";
        LOG_DEBUG("onnx_backend") << "[extract_metadata] output[" << i << "] name='" << name
             << "' dtype=" << dtype << " shape=" << shape_oss.str();
    }

    LOG_DEBUG("onnx_backend") << "[extract_metadata] concluído; " << input_names_.size()
         << " inputs e " << output_names_.size() << " outputs registrados";
}

common::DataType OnnxBackend::ort_to_proto_dtype(ONNXTensorElementDataType t) {
    LOG_DEBUG("onnx_backend") << "[ort_to_proto_dtype] ONNXTensorElementDataType=" << t;
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return common::FLOAT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return common::FLOAT64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return common::INT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return common::INT64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return common::UINT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return common::BOOL;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  return common::STRING;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return common::FLOAT16;
        default:
            LOG_WARN("onnx_backend") << "[ort_to_proto_dtype] dtype desconhecido=" << t << ", fallback para FLOAT32";
            return common::FLOAT32;
    }
}

std::vector<int64_t> OnnxBackend::resolve_dynamic_shape(
    const std::vector<int64_t>& shape,
    size_t data_size,
    const std::string& tensor_name,
    std::string& error_msg) {

    LOG_DEBUG("onnx_backend") << "[resolve_dynamic_shape] tensor='" << tensor_name
         << "' data_size=" << data_size << " shape_size=" << shape.size();

    if (shape.empty()) {
        LOG_DEBUG("onnx_backend") << "[resolve_dynamic_shape] shape vazio → fallback flat 1-D [" << data_size << "]";
        return {static_cast<int64_t>(data_size)};
    }

    std::vector<int64_t> resolved = shape;
    int64_t static_product = 1;
    int dynamic_idx = -1;

    for (int i = 0; i < static_cast<int>(resolved.size()); ++i) {
        if (resolved[i] == -1) {
            if (dynamic_idx >= 0) {
                error_msg = "Tensor '" + tensor_name + "' has multiple dynamic dims";
                LOG_ERROR("onnx_backend") << "[resolve_dynamic_shape] múltiplas dimensões dinâmicas em '" << tensor_name << "'";
                return {};
            }
            dynamic_idx = i;
            LOG_DEBUG("onnx_backend") << "[resolve_dynamic_shape] dimensão dinâmica encontrada no índice " << i;
        } else {
            static_product *= resolved[i];
            LOG_DEBUG("onnx_backend") << "[resolve_dynamic_shape] dim[" << i << "]=" << resolved[i]
                 << " static_product acumulado=" << static_product;
        }
    }

    if (dynamic_idx >= 0) {
        LOG_DEBUG("onnx_backend") << "[resolve_dynamic_shape] resolvendo dim dinâmica[" << dynamic_idx << "]"
             << " data_size=" << data_size << " static_product=" << static_product;

        if (static_product == 0 || data_size % static_cast<size_t>(static_product) != 0) {
            error_msg = "Cannot resolve dynamic dim for '" + tensor_name
                      + "': data_size=" + std::to_string(data_size)
                      + " static_product=" + std::to_string(static_product);
            LOG_ERROR("onnx_backend") << "[resolve_dynamic_shape] " << error_msg;
            return {};
        }
        resolved[dynamic_idx] = static_cast<int64_t>(data_size) / static_product;
        LOG_DEBUG("onnx_backend") << "[resolve_dynamic_shape] dim[" << dynamic_idx << "] resolvida para "
             << resolved[dynamic_idx];
    }

    {
        std::ostringstream shape_oss;
        shape_oss << "[";
        for (size_t j = 0; j < resolved.size(); ++j) {
            shape_oss << resolved[j];
            if (j + 1 < resolved.size()) shape_oss << ", ";
        }
        shape_oss << "]";
        LOG_DEBUG("onnx_backend") << "[resolve_dynamic_shape] shape final resolvido=" << shape_oss.str()
             << " para tensor='" << tensor_name << "'";
    }

    return resolved;
}

std::vector<float> OnnxBackend::value_to_floats(const client::Value& v) {
    LOG_DEBUG("onnx_backend") << "[value_to_floats] is_number=" << v.is_number()
         << " is_array=" << v.is_array();

    if (v.is_number()) {
        float val = static_cast<float>(v.as_number());
        LOG_DEBUG("onnx_backend") << "[value_to_floats] escalar=" << val;
        return {val};
    }

    if (v.is_array()) {
        const auto& arr = v.as_array();
        LOG_DEBUG("onnx_backend") << "[value_to_floats] array size=" << arr.size();
        std::vector<float> out;
        out.reserve(arr.size());
        for (const auto& elem : arr) {
            if (elem.is_number())
                out.push_back(static_cast<float>(elem.as_number()));
            else {
                LOG_DEBUG("onnx_backend") << "[value_to_floats] elemento ignorado: não é número";
            }
        }
        LOG_DEBUG("onnx_backend") << "[value_to_floats] out.size()=" << out.size();
        return out;
    }

    LOG_WARN("onnx_backend") << "[value_to_floats] tipo não suportado (não é número nem array), retornando vazio";
    return {};
}

client::Value OnnxBackend::floats_to_value(const float* ptr, size_t count) {
    LOG_DEBUG("onnx_backend") << "[floats_to_value] count=" << count;

    if (count == 1) {
        LOG_DEBUG("onnx_backend") << "[floats_to_value] escalar=" << ptr[0];
        return client::Value{static_cast<double>(ptr[0])};
    }

    {
        std::ostringstream vals_oss;
        size_t preview = std::min(count, size_t(8));
        vals_oss << "[";
        for (size_t i = 0; i < preview; ++i) {
            vals_oss << ptr[i];
            if (i < preview - 1) vals_oss << ", ";
        }
        if (count > 8) vals_oss << ", ...";
        vals_oss << "]";
        LOG_DEBUG("onnx_backend") << "[floats_to_value] primeiros valores=" << vals_oss.str();
    }

    client::Array arr;
    arr.reserve(count);
    for (size_t i = 0; i < count; ++i)
        arr.push_back(client::Value{static_cast<double>(ptr[i])});

    return client::Value{std::move(arr)};
}

}  // namespace inference
}  // namespace mlinference
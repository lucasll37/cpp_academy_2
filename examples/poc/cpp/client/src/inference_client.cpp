// =============================================================================
// inference_client.cpp — Implementation of the AsaMiia client library
// =============================================================================

#include "client/inference_client.hpp"
#include <iostream>
#include <chrono>

namespace mlinference {
namespace client {

// ============================================
// Construction / Connection
// ============================================

InferenceClient::InferenceClient(const std::string& server_address)
    : server_address_(server_address) {}

InferenceClient::~InferenceClient() = default;

bool InferenceClient::connect() {
    channel_ = grpc::CreateChannel(server_address_, grpc::InsecureChannelCredentials());
    stub_ = worker::WorkerService::NewStub(channel_);

    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
    connected_ = channel_->WaitForConnected(deadline);

    if (connected_) {
        std::cout << "Connected to " << server_address_ << std::endl;
    } else {
        std::cerr << "Failed to connect to " << server_address_ << std::endl;
    }
    return connected_;
}

bool InferenceClient::is_connected() const {
    return connected_;
}

// ============================================
// Model Lifecycle
// ============================================

bool InferenceClient::load_model(const std::string& model_id,
                                  const std::string& model_path,
                                  const std::string& version) {
    if (!connected_) return false;

    grpc::ClientContext ctx;
    worker::LoadModelRequest req;
    worker::LoadModelResponse resp;

    req.set_model_id(model_id);
    req.set_model_path(model_path);
    req.set_version(version);

    auto status = stub_->LoadModel(&ctx, req, &resp);
    if (!status.ok()) {
        std::cerr << "LoadModel RPC failed: " << status.error_message() << std::endl;
        return false;
    }

    if (resp.success()) {
        std::cout << "Model loaded: " << model_id << std::endl;
    } else {
        std::cerr << "LoadModel failed: " << resp.error_message() << std::endl;
    }
    return resp.success();
}

bool InferenceClient::unload_model(const std::string& model_id) {
    if (!connected_) return false;

    grpc::ClientContext ctx;
    worker::UnloadModelRequest req;
    worker::UnloadModelResponse resp;

    req.set_model_id(model_id);

    auto status = stub_->UnloadModel(&ctx, req, &resp);
    if (!status.ok()) {
        std::cerr << "UnloadModel RPC failed: " << status.error_message() << std::endl;
        return false;
    }
    return resp.success();
}

// ============================================
// Inference
// ============================================

PredictionResult InferenceClient::predict(
    const std::string& model_id,
    const std::map<std::string, std::vector<float>>& inputs) {

    PredictionResult result;
    if (!connected_) {
        result.error_message = "Not connected";
        return result;
    }

    grpc::ClientContext ctx;
    worker::PredictRequest req;
    worker::PredictResponse resp;

    req.set_model_id(model_id);
    for (const auto& [name, data] : inputs) {
        set_tensor_data(&(*req.mutable_inputs())[name], data);
    }

    auto status = stub_->Predict(&ctx, req, &resp);
    if (!status.ok()) {
        result.error_message = "RPC failed: " + status.error_message();
        return result;
    }

    result.success           = resp.success();
    result.inference_time_ms = resp.inference_time_ms();
    result.error_message     = resp.error_message();

    if (resp.success()) {
        for (const auto& [name, tensor] : resp.outputs()) {
            result.outputs[name] = get_tensor_data(tensor);
        }
    }

    return result;
}

std::vector<PredictionResult> InferenceClient::batch_predict(
    const std::string& model_id,
    const std::vector<std::map<std::string, std::vector<float>>>& batch_inputs) {

    std::vector<PredictionResult> results;
    if (!connected_) return results;

    grpc::ClientContext ctx;
    worker::BatchPredictRequest req;
    worker::BatchPredictResponse resp;

    req.set_model_id(model_id);
    for (const auto& inputs : batch_inputs) {
        auto* single = req.add_requests();
        single->set_model_id(model_id);
        for (const auto& [name, data] : inputs) {
            set_tensor_data(&(*single->mutable_inputs())[name], data);
        }
    }

    auto status = stub_->BatchPredict(&ctx, req, &resp);
    if (!status.ok()) return results;

    for (const auto& single_resp : resp.responses()) {
        PredictionResult r;
        r.success            = single_resp.success();
        r.inference_time_ms  = single_resp.inference_time_ms();
        r.error_message      = single_resp.error_message();

        if (single_resp.success()) {
            for (const auto& [name, tensor] : single_resp.outputs()) {
                r.outputs[name] = get_tensor_data(tensor);
            }
        }
        results.push_back(std::move(r));
    }

    return results;
}

// ============================================
// Introspection
// ============================================

std::vector<ModelInfo> InferenceClient::list_models() {
    std::vector<ModelInfo> result;
    if (!connected_) return result;

    grpc::ClientContext ctx;
    worker::ListModelsRequest req;
    worker::ListModelsResponse resp;

    auto status = stub_->ListModels(&ctx, req, &resp);
    if (!status.ok()) return result;

    for (const auto& proto : resp.models()) {
        result.push_back(proto_to_model_info(proto));
    }
    return result;
}

ModelInfo InferenceClient::get_model_info(const std::string& model_id) {
    ModelInfo result;
    if (!connected_) return result;

    grpc::ClientContext ctx;
    worker::GetModelInfoRequest req;
    worker::GetModelInfoResponse resp;

    req.set_model_id(model_id);

    auto status = stub_->GetModelInfo(&ctx, req, &resp);
    if (!status.ok() || !resp.success()) return result;

    return proto_to_model_info(resp.model_info());
}

ValidationResult InferenceClient::validate_model(const std::string& model_path) {
    ValidationResult result;
    if (!connected_) return result;

    grpc::ClientContext ctx;
    worker::ValidateModelRequest req;
    worker::ValidateModelResponse resp;

    req.set_model_path(model_path);

    auto status = stub_->ValidateModel(&ctx, req, &resp);
    if (!status.ok()) {
        result.error_message = "RPC failed: " + status.error_message();
        return result;
    }

    result.valid         = resp.valid();
    result.backend       = backend_type_to_string(resp.backend());
    result.error_message = resp.error_message();

    for (const auto& w : resp.warnings()) {
        result.warnings.push_back(w);
    }
    for (const auto& ts : resp.inputs()) {
        ModelInfo::TensorSpec spec;
        spec.name  = ts.name();
        spec.dtype = dtype_to_string(ts.dtype());
        for (int64_t d : ts.shape()) spec.shape.push_back(d);
        result.inputs.push_back(std::move(spec));
    }
    for (const auto& ts : resp.outputs()) {
        ModelInfo::TensorSpec spec;
        spec.name  = ts.name();
        spec.dtype = dtype_to_string(ts.dtype());
        for (int64_t d : ts.shape()) spec.shape.push_back(d);
        result.outputs.push_back(std::move(spec));
    }

    return result;
}

WarmupResult InferenceClient::warmup_model(const std::string& model_id,
                                            uint32_t num_runs) {
    WarmupResult result;
    if (!connected_) return result;

    grpc::ClientContext ctx;
    worker::WarmupModelRequest req;
    worker::WarmupModelResponse resp;

    req.set_model_id(model_id);
    req.set_num_runs(num_runs);

    auto status = stub_->WarmupModel(&ctx, req, &resp);
    if (!status.ok()) {
        result.error_message = "RPC failed: " + status.error_message();
        return result;
    }

    result.success        = resp.success();
    result.runs_completed = resp.runs_completed();
    result.avg_time_ms    = resp.avg_time_ms();
    result.min_time_ms    = resp.min_time_ms();
    result.max_time_ms    = resp.max_time_ms();
    result.error_message  = resp.error_message();
    return result;
}

// ============================================
// Observability
// ============================================

bool InferenceClient::health_check() {
    if (!connected_) return false;

    grpc::ClientContext ctx;
    worker::HealthCheckRequest req;
    worker::HealthCheckResponse resp;

    auto status = stub_->HealthCheck(&ctx, req, &resp);
    return status.ok() && resp.healthy();
}

WorkerStatus InferenceClient::get_status() {
    WorkerStatus result;
    if (!connected_) return result;

    grpc::ClientContext ctx;
    worker::GetStatusRequest req;
    worker::GetStatusResponse resp;

    auto status = stub_->GetStatus(&ctx, req, &resp);
    if (!status.ok()) return result;

    result.worker_id           = resp.worker_id();
    result.total_requests      = resp.metrics().total_requests();
    result.successful_requests = resp.metrics().successful_requests();
    result.failed_requests     = resp.metrics().failed_requests();
    result.active_requests     = resp.metrics().active_requests();
    result.uptime_seconds      = resp.metrics().uptime_seconds();

    for (const auto& id : resp.loaded_model_ids()) {
        result.loaded_models.push_back(id);
    }
    for (const auto& b : resp.capabilities().supported_backends()) {
        result.supported_backends.push_back(b);
    }

    return result;
}

ServerMetrics InferenceClient::get_metrics() {
    ServerMetrics result;
    if (!connected_) return result;

    grpc::ClientContext ctx;
    worker::GetMetricsRequest req;
    worker::GetMetricsResponse resp;

    auto status = stub_->GetMetrics(&ctx, req, &resp);
    if (!status.ok()) return result;

    // Worker-level counters
    const auto& wm         = resp.worker_metrics();
    result.total_requests      = wm.total_requests();
    result.successful_requests = wm.successful_requests();
    result.failed_requests     = wm.failed_requests();
    result.active_requests     = wm.active_requests();
    result.uptime_seconds      = wm.uptime_seconds();

    // Per-model breakdown
    for (const auto& [id, m] : resp.per_model_metrics()) {
        ModelMetrics mm;
        mm.total_inferences  = m.total_inferences();
        mm.failed_inferences = m.failed_inferences();
        mm.avg_ms            = m.avg_inference_time_ms();
        mm.min_ms            = m.min_inference_time_ms();
        mm.max_ms            = m.max_inference_time_ms();
        mm.p95_ms            = m.p95_inference_time_ms();
        mm.p99_ms            = m.p99_inference_time_ms();
        mm.total_time_ms     = m.total_inference_time_ms();
        mm.last_used_at_unix = m.last_used_at_unix();
        mm.loaded_at_unix    = m.loaded_at_unix();
        result.per_model[id] = mm;
    }

    return result;
}

// ============================================
// File Discovery
// ============================================

std::vector<AvailableModel> InferenceClient::list_available_models(
    const std::string& directory) {

    std::vector<AvailableModel> result;
    if (!connected_) return result;

    grpc::ClientContext ctx;
    worker::ListAvailableModelsRequest req;
    worker::ListAvailableModelsResponse resp;

    if (!directory.empty()) req.set_directory(directory);

    auto status = stub_->ListAvailableModels(&ctx, req, &resp);
    if (!status.ok()) {
        std::cerr << "ListAvailableModels RPC failed: " << status.error_message() << std::endl;
        return result;
    }

    for (const auto& m : resp.models()) {
        AvailableModel am;
        am.filename        = m.filename();
        am.path            = m.path();
        am.extension       = m.extension();
        am.backend         = backend_type_to_string(m.backend());
        am.file_size_bytes = m.file_size_bytes();
        am.is_loaded       = m.is_loaded();
        am.loaded_as       = m.loaded_as();
        result.push_back(std::move(am));
    }
    return result;
}

// ============================================
// Helpers
// ============================================

void InferenceClient::set_tensor_data(common::Tensor* tensor,
                                       const std::vector<float>& data) {
    tensor->set_dtype(common::FLOAT32);
    tensor->add_shape(data.size());
    tensor->mutable_data()->assign(
        reinterpret_cast<const char*>(data.data()),
        data.size() * sizeof(float));
}

std::vector<float> InferenceClient::get_tensor_data(const common::Tensor& tensor) {
    const auto& bytes = tensor.data();
    if (tensor.dtype() != common::FLOAT32) return {};

    const float* ptr = reinterpret_cast<const float*>(bytes.data());
    size_t count     = bytes.size() / sizeof(float);
    return std::vector<float>(ptr, ptr + count);
}

ModelInfo InferenceClient::proto_to_model_info(const common::ModelInfo& proto) {
    ModelInfo info;
    info.model_id           = proto.model_id();
    info.version            = proto.version();
    info.backend            = backend_type_to_string(proto.backend());
    info.description        = proto.description();
    info.author             = proto.author();
    info.memory_usage_bytes = proto.memory_usage_bytes();
    info.is_warmed_up       = proto.is_warmed_up();

    for (const auto& ts : proto.inputs()) {
        ModelInfo::TensorSpec spec;
        spec.name        = ts.name();
        spec.dtype       = dtype_to_string(ts.dtype());
        spec.description = ts.description();
        for (int64_t d : ts.shape()) spec.shape.push_back(d);
        info.inputs.push_back(std::move(spec));
    }
    for (const auto& ts : proto.outputs()) {
        ModelInfo::TensorSpec spec;
        spec.name        = ts.name();
        spec.dtype       = dtype_to_string(ts.dtype());
        spec.description = ts.description();
        for (int64_t d : ts.shape()) spec.shape.push_back(d);
        info.outputs.push_back(std::move(spec));
    }
    for (const auto& [k, v] : proto.tags()) {
        info.tags[k] = v;
    }

    return info;
}

std::string InferenceClient::backend_type_to_string(common::BackendType type) {
    switch (type) {
        case common::BACKEND_ONNX:   return "onnx";
        case common::BACKEND_PYTHON: return "python";
        default:                      return "unknown";
    }
}

std::string InferenceClient::dtype_to_string(common::DataType type) {
    switch (type) {
        case common::FLOAT32: return "float32";
        case common::FLOAT64: return "float64";
        case common::INT32:   return "int32";
        case common::INT64:   return "int64";
        case common::UINT8:   return "uint8";
        case common::BOOL:    return "bool";
        case common::STRING:  return "string";
        case common::FLOAT16: return "float16";
        default:               return "unknown";
    }
}

}  // namespace client
}  // namespace mlinference
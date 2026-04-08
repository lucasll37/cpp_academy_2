// =============================================================================
// inference_client.cpp — Implementação do cliente AsaMiia
//
// Organização do ficheiro:
//
//   §1  IClientBackend  — interface interna pura
//   §2  GrpcClientBackend — implementação original via gRPC (comportamento
//                           anterior, apenas movido para dentro da classe)
//   §3  InProcessBackend  — nova implementação que instancia InferenceEngine
//                           directamente no mesmo processo
//   §4  InferenceClient   — classe pública; detecta o modo e delega
// =============================================================================

#include "client/inference_client.hpp"

// ── gRPC / Protobuf ──────────────────────────────────────────────────────────
#include <grpcpp/grpcpp.h>
#include "worker.grpc.pb.h"
#include "common.pb.h"

// ── Worker internals (apenas necessários para InProcessBackend) ───────────────
#include "worker/inference_engine.hpp"
#include "worker/backend_registry.hpp"

// ── STL ──────────────────────────────────────────────────────────────────────
#include <iostream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <thread>

namespace mlinference {
namespace client {

// =============================================================================
// §1  Interface interna
// =============================================================================

class IClientBackend {
public:
    virtual ~IClientBackend() = default;

    virtual bool connect()           = 0;
    virtual bool is_connected() const = 0;

    virtual bool load_model(const std::string& model_id,
                            const std::string& model_path,
                            const std::string& version)   = 0;

    virtual bool unload_model(const std::string& model_id) = 0;

    virtual PredictionResult predict(
        const std::string& model_id,
        const std::map<std::string, std::vector<float>>& inputs) = 0;

    virtual std::vector<PredictionResult> batch_predict(
        const std::string& model_id,
        const std::vector<std::map<std::string, std::vector<float>>>& batch_inputs) = 0;

    virtual std::vector<ModelInfo> list_models()                          = 0;
    virtual ModelInfo get_model_info(const std::string& model_id)        = 0;
    virtual ValidationResult validate_model(const std::string& path)     = 0;
    virtual WarmupResult warmup_model(const std::string& model_id,
                                      uint32_t num_runs)                  = 0;

    virtual bool health_check()      = 0;
    virtual WorkerStatus get_status() = 0;
    virtual ServerMetrics get_metrics() = 0;

    virtual std::vector<AvailableModel> list_available_models(
        const std::string& directory) = 0;
};

// =============================================================================
// §2  GrpcClientBackend  — comportamento original
// =============================================================================

class GrpcClientBackend final : public IClientBackend {
public:
    explicit GrpcClientBackend(const std::string& server_address)
        : server_address_(server_address) {}

    // ── helpers de serialização ───────────────────────────────────────────────

    static void set_tensor(common::Tensor* t, const std::vector<float>& data) {
        t->set_dtype(common::FLOAT32);
        t->add_shape(static_cast<int64_t>(data.size()));
        t->mutable_data()->assign(
            reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(float));
    }

    static std::vector<float> get_tensor(const common::Tensor& t) {
        if (t.dtype() != common::FLOAT32) return {};
        const float* ptr = reinterpret_cast<const float*>(t.data().data());
        return std::vector<float>(ptr, ptr + t.data().size() / sizeof(float));
    }

    static std::string backend_str(common::BackendType bt) {
        switch (bt) {
            case common::BACKEND_ONNX:   return "onnx";
            case common::BACKEND_PYTHON: return "python";
            default:                      return "unknown";
        }
    }

    static std::string dtype_str(common::DataType dt) {
        switch (dt) {
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

    static ModelInfo proto_to_model_info(const common::ModelInfo& p) {
        ModelInfo info;
        info.model_id           = p.model_id();
        info.version            = p.version();
        info.backend            = backend_str(p.backend());
        info.description        = p.description();
        info.author             = p.author();
        info.memory_usage_bytes = p.memory_usage_bytes();
        info.is_warmed_up       = p.is_warmed_up();
        info.loaded_at_unix     = p.loaded_at_unix();

        for (const auto& ts : p.inputs()) {
            ModelInfo::TensorSpec s;
            s.name        = ts.name();
            s.dtype       = dtype_str(ts.dtype());
            s.description = ts.description();
            for (int64_t d : ts.shape()) s.shape.push_back(d);
            info.inputs.push_back(std::move(s));
        }
        for (const auto& ts : p.outputs()) {
            ModelInfo::TensorSpec s;
            s.name        = ts.name();
            s.dtype       = dtype_str(ts.dtype());
            s.description = ts.description();
            for (int64_t d : ts.shape()) s.shape.push_back(d);
            info.outputs.push_back(std::move(s));
        }
        for (const auto& [k, v] : p.tags()) info.tags[k] = v;
        return info;
    }

    // ── IClientBackend ────────────────────────────────────────────────────────

    bool connect() override {
        channel_ = grpc::CreateChannel(server_address_,
                                        grpc::InsecureChannelCredentials());
        stub_     = worker::WorkerService::NewStub(channel_);

        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
        connected_    = channel_->WaitForConnected(deadline);

        if (connected_)
            std::cout << "[AsaMiia] Connected to " << server_address_ << std::endl;
        else
            std::cerr << "[AsaMiia] Failed to connect to " << server_address_ << std::endl;

        return connected_;
    }

    bool is_connected() const override { return connected_; }

    bool load_model(const std::string& model_id,
                    const std::string& model_path,
                    const std::string& version) override {
        if (!connected_) return false;

        grpc::ClientContext ctx;
        worker::LoadModelRequest  req;
        worker::LoadModelResponse resp;

        req.set_model_id(model_id);
        req.set_model_path(model_path);
        req.set_version(version);

        auto st = stub_->LoadModel(&ctx, req, &resp);
        if (!st.ok()) {
            std::cerr << "[AsaMiia] LoadModel RPC failed: " << st.error_message() << std::endl;
            return false;
        }
        if (!resp.success())
            std::cerr << "[AsaMiia] LoadModel: " << resp.error_message() << std::endl;
        return resp.success();
    }

    bool unload_model(const std::string& model_id) override {
        if (!connected_) return false;

        grpc::ClientContext ctx;
        worker::UnloadModelRequest  req;
        worker::UnloadModelResponse resp;

        req.set_model_id(model_id);
        auto st = stub_->UnloadModel(&ctx, req, &resp);
        if (!st.ok()) return false;
        return resp.success();
    }

    PredictionResult predict(
        const std::string& model_id,
        const std::map<std::string, std::vector<float>>& inputs) override {

        PredictionResult result;
        if (!connected_) { result.error_message = "Not connected"; return result; }

        grpc::ClientContext    ctx;
        worker::PredictRequest req;
        worker::PredictResponse resp;

        req.set_model_id(model_id);
        for (const auto& [name, data] : inputs)
            set_tensor(&(*req.mutable_inputs())[name], data);

        auto st = stub_->Predict(&ctx, req, &resp);
        if (!st.ok()) { result.error_message = "RPC failed: " + st.error_message(); return result; }

        result.success           = resp.success();
        result.inference_time_ms = resp.inference_time_ms();
        result.error_message     = resp.error_message();

        if (resp.success())
            for (const auto& [name, tensor] : resp.outputs())
                result.outputs[name] = get_tensor(tensor);

        return result;
    }

    std::vector<PredictionResult> batch_predict(
        const std::string& model_id,
        const std::vector<std::map<std::string, std::vector<float>>>& batch_inputs) override {

        std::vector<PredictionResult> results;
        if (!connected_) return results;

        grpc::ClientContext        ctx;
        worker::BatchPredictRequest  req;
        worker::BatchPredictResponse resp;

        req.set_model_id(model_id);
        for (const auto& inputs : batch_inputs) {
            auto* single = req.add_requests();
            single->set_model_id(model_id);
            for (const auto& [name, data] : inputs)
                set_tensor(&(*single->mutable_inputs())[name], data);
        }

        auto st = stub_->BatchPredict(&ctx, req, &resp);
        if (!st.ok()) return results;

        for (const auto& r : resp.responses()) {
            PredictionResult pr;
            pr.success           = r.success();
            pr.inference_time_ms = r.inference_time_ms();
            pr.error_message     = r.error_message();
            if (r.success())
                for (const auto& [name, tensor] : r.outputs())
                    pr.outputs[name] = get_tensor(tensor);
            results.push_back(std::move(pr));
        }
        return results;
    }

    std::vector<ModelInfo> list_models() override {
        std::vector<ModelInfo> result;
        if (!connected_) return result;

        grpc::ClientContext       ctx;
        worker::ListModelsRequest  req;
        worker::ListModelsResponse resp;

        if (!stub_->ListModels(&ctx, req, &resp).ok()) return result;
        for (const auto& p : resp.models())
            result.push_back(proto_to_model_info(p));
        return result;
    }

    ModelInfo get_model_info(const std::string& model_id) override {
        ModelInfo result;
        if (!connected_) return result;

        grpc::ClientContext          ctx;
        worker::GetModelInfoRequest  req;
        worker::GetModelInfoResponse resp;

        req.set_model_id(model_id);
        auto st = stub_->GetModelInfo(&ctx, req, &resp);
        if (!st.ok() || !resp.success()) return result;
        return proto_to_model_info(resp.model_info());
    }

    ValidationResult validate_model(const std::string& path) override {
        ValidationResult result;
        if (!connected_) return result;

        grpc::ClientContext           ctx;
        worker::ValidateModelRequest  req;
        worker::ValidateModelResponse resp;

        req.set_model_path(path);
        auto st = stub_->ValidateModel(&ctx, req, &resp);
        if (!st.ok()) return result;

        result.valid         = resp.valid();
        result.error_message = resp.error_message();
        result.backend       = GrpcClientBackend::backend_str(resp.backend());
        for (const auto& w : resp.warnings()) result.warnings.push_back(w);

        for (const auto& ts : resp.inputs()) {
            ModelInfo::TensorSpec s;
            s.name        = ts.name();
            s.dtype       = dtype_str(ts.dtype());
            s.description = ts.description();
            for (int64_t d : ts.shape()) s.shape.push_back(d);
            result.inputs.push_back(std::move(s));
        }
        for (const auto& ts : resp.outputs()) {
            ModelInfo::TensorSpec s;
            s.name        = ts.name();
            s.dtype       = dtype_str(ts.dtype());
            s.description = ts.description();
            for (int64_t d : ts.shape()) s.shape.push_back(d);
            result.outputs.push_back(std::move(s));
        }
        return result;
    }

    WarmupResult warmup_model(const std::string& model_id,
                               uint32_t num_runs) override {
        WarmupResult result;
        if (!connected_) return result;

        grpc::ClientContext         ctx;
        worker::WarmupModelRequest  req;
        worker::WarmupModelResponse resp;

        req.set_model_id(model_id);
        req.set_num_runs(num_runs);
        auto st = stub_->WarmupModel(&ctx, req, &resp);
        if (!st.ok()) return result;

        result.success        = resp.success();
        result.runs_completed = resp.runs_completed();
        result.avg_time_ms    = resp.avg_time_ms();
        result.min_time_ms    = resp.min_time_ms();
        result.max_time_ms    = resp.max_time_ms();
        result.error_message  = resp.error_message();
        return result;
    }

    bool health_check() override {
        if (!connected_) return false;

        grpc::ClientContext         ctx;
        worker::HealthCheckRequest  req;
        worker::HealthCheckResponse resp;

        auto st = stub_->HealthCheck(&ctx, req, &resp);
        return st.ok() && resp.healthy();
    }

    WorkerStatus get_status() override {
        WorkerStatus result;
        if (!connected_) return result;

        grpc::ClientContext       ctx;
        worker::GetStatusRequest  req;
        worker::GetStatusResponse resp;

        if (!stub_->GetStatus(&ctx, req, &resp).ok()) return result;

        result.worker_id          = resp.worker_id();
        result.total_requests     = resp.metrics().total_requests();
        result.successful_requests= resp.metrics().successful_requests();
        result.failed_requests    = resp.metrics().failed_requests();
        result.active_requests    = resp.metrics().active_requests();
        result.uptime_seconds     = resp.metrics().uptime_seconds();

        for (const auto& id : resp.loaded_model_ids())
            result.loaded_models.push_back(id);
        for (const auto& b : resp.capabilities().supported_backends())
            result.supported_backends.push_back(b);

        return result;
    }

    ServerMetrics get_metrics() override {
        ServerMetrics result;
        if (!connected_) return result;

        grpc::ClientContext        ctx;
        worker::GetMetricsRequest  req;
        worker::GetMetricsResponse resp;

        if (!stub_->GetMetrics(&ctx, req, &resp).ok()) return result;

        const auto& wm           = resp.worker_metrics();
        result.total_requests     = wm.total_requests();
        result.successful_requests= wm.successful_requests();
        result.failed_requests    = wm.failed_requests();
        result.active_requests    = wm.active_requests();
        result.uptime_seconds     = wm.uptime_seconds();

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

    std::vector<AvailableModel> list_available_models(
        const std::string& directory) override {

        std::vector<AvailableModel> result;
        if (!connected_) return result;

        grpc::ClientContext                  ctx;
        worker::ListAvailableModelsRequest   req;
        worker::ListAvailableModelsResponse  resp;

        if (!directory.empty()) req.set_directory(directory);
        if (!stub_->ListAvailableModels(&ctx, req, &resp).ok()) return result;

        for (const auto& m : resp.models()) {
            AvailableModel am;
            am.filename        = m.filename();
            am.path            = m.path();
            am.extension       = m.extension();
            am.backend         = backend_str(m.backend());
            am.file_size_bytes = m.file_size_bytes();
            am.is_loaded       = m.is_loaded();
            am.loaded_as       = m.loaded_as();
            result.push_back(std::move(am));
        }
        return result;
    }

private:
    std::string server_address_;
    bool connected_ = false;

    std::shared_ptr<grpc::Channel>                     channel_;
    std::unique_ptr<worker::WorkerService::Stub>       stub_;
};

// =============================================================================
// §3  InProcessBackend  — engine embarcada no mesmo processo
// =============================================================================

class InProcessBackend final : public IClientBackend {
public:
    InProcessBackend() = default;

    // ── helpers de conversão ─────────────────────────────────────────────────

    static std::string backend_str(common::BackendType bt) {
        switch (bt) {
            case common::BACKEND_ONNX:   return "onnx";
            case common::BACKEND_PYTHON: return "python";
            default:                      return "unknown";
        }
    }

    static std::string dtype_str(common::DataType dt) {
        switch (dt) {
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

    /// Converte common::ModelInfo (protobuf) → ModelInfo (struct do cliente).
    static ModelInfo proto_to_model_info(const common::ModelInfo& p) {
        ModelInfo info;
        info.model_id           = p.model_id();
        info.version            = p.version();
        info.backend            = backend_str(p.backend());
        info.description        = p.description();
        info.author             = p.author();
        info.memory_usage_bytes = p.memory_usage_bytes();
        info.is_warmed_up       = p.is_warmed_up();
        info.loaded_at_unix     = p.loaded_at_unix();

        for (const auto& ts : p.inputs()) {
            ModelInfo::TensorSpec s;
            s.name        = ts.name();
            s.dtype       = dtype_str(ts.dtype());
            s.description = ts.description();
            for (int64_t d : ts.shape()) s.shape.push_back(d);
            info.inputs.push_back(std::move(s));
        }
        for (const auto& ts : p.outputs()) {
            ModelInfo::TensorSpec s;
            s.name        = ts.name();
            s.dtype       = dtype_str(ts.dtype());
            s.description = ts.description();
            for (int64_t d : ts.shape()) s.shape.push_back(d);
            info.outputs.push_back(std::move(s));
        }
        for (const auto& [k, v] : p.tags()) info.tags[k] = v;
        return info;
    }

    // ── IClientBackend ────────────────────────────────────────────────────────

    bool connect() override {
        // A "conexão" in-process consiste apenas em instanciar o engine.
        // Usa os mesmos defaults do WorkerServer: CPU, 4 threads.
        engine_ = std::make_unique<worker::InferenceEngine>(
            /*enable_gpu=*/false,
            /*gpu_device_id=*/0,
            /*num_threads=*/4);

        connected_  = true;
        start_time_ = std::chrono::steady_clock::now();

        std::cout << "[AsaMiia] In-process engine initialized." << std::endl;
        return true;
    }

    bool is_connected() const override { return connected_; }

    // ── Ciclo de vida ─────────────────────────────────────────────────────────

    bool load_model(const std::string& model_id,
                    const std::string& model_path,
                    const std::string& version) override {
        if (!connected_) return false;

        bool ok = engine_->load_model(model_id, model_path);
        if (ok) {
            // Guarda a versão separadamente — InferenceEngine não expõe setter.
            versions_[model_id] = version;
            std::cout << "[AsaMiia] In-process model loaded: " << model_id << std::endl;
        } else {
            std::cerr << "[AsaMiia] Failed to load model: " << model_id << std::endl;
        }
        return ok;
    }

    bool unload_model(const std::string& model_id) override {
        if (!connected_) return false;
        versions_.erase(model_id);
        return engine_->unload_model(model_id);
    }

    // ── Inferência ────────────────────────────────────────────────────────────

    PredictionResult predict(
        const std::string& model_id,
        const std::map<std::string, std::vector<float>>& inputs) override {

        PredictionResult result;
        if (!connected_) { result.error_message = "Not connected"; return result; }

        total_requests_++;

        auto ir = engine_->predict(model_id, inputs);

        result.success           = ir.success;
        result.inference_time_ms = ir.inference_time_ms;
        result.error_message     = ir.error_message;
        result.outputs           = ir.outputs;

        if (ir.success) successful_requests_++;
        else            failed_requests_++;

        return result;
    }

    std::vector<PredictionResult> batch_predict(
        const std::string& model_id,
        const std::vector<std::map<std::string, std::vector<float>>>& batch_inputs) override {

        std::vector<PredictionResult> results;
        results.reserve(batch_inputs.size());
        for (const auto& inp : batch_inputs)
            results.push_back(predict(model_id, inp));
        return results;
    }

    // ── Introspecção ──────────────────────────────────────────────────────────

    std::vector<ModelInfo> list_models() override {
        std::vector<ModelInfo> result;
        if (!connected_) return result;

        for (const auto& id : engine_->get_loaded_model_ids()) {
            auto proto = engine_->get_model_info(id);
            auto info  = proto_to_model_info(proto);

            // Sobrepõe a versão armazenada localmente, se existir.
            auto it = versions_.find(id);
            if (it != versions_.end()) info.version = it->second;

            result.push_back(std::move(info));
        }
        return result;
    }

    ModelInfo get_model_info(const std::string& model_id) override {
        ModelInfo result;
        if (!connected_) return result;

        auto proto = engine_->get_model_info(model_id);
        result     = proto_to_model_info(proto);

        auto it = versions_.find(model_id);
        if (it != versions_.end()) result.version = it->second;

        return result;
    }

    ValidationResult validate_model(const std::string& path) override {
        ValidationResult result;
        if (!connected_) return result;

        auto vr              = engine_->validate_model(path);
        result.valid         = vr.valid;
        result.error_message = vr.error_message;
        result.backend       = backend_str(vr.backend);
        result.warnings      = vr.warnings;

        // Popula schema preview a partir do ModelSchema retornado pelo engine.
        for (const auto& spec : vr.schema.inputs) {
            ModelInfo::TensorSpec s;
            s.name        = spec.name;
            s.dtype       = dtype_str(spec.dtype);
            s.description = spec.description;
            s.shape       = spec.shape;
            result.inputs.push_back(std::move(s));
        }
        for (const auto& spec : vr.schema.outputs) {
            ModelInfo::TensorSpec s;
            s.name        = spec.name;
            s.dtype       = dtype_str(spec.dtype);
            s.description = spec.description;
            s.shape       = spec.shape;
            result.outputs.push_back(std::move(s));
        }
        return result;
    }

    WarmupResult warmup_model(const std::string& model_id,
                               uint32_t num_runs) override {
        WarmupResult result;
        if (!connected_) return result;

        auto wr           = engine_->warmup_model(model_id, num_runs);
        result.success        = wr.success;
        result.runs_completed = wr.runs_completed;
        result.avg_time_ms    = wr.avg_time_ms;
        result.min_time_ms    = wr.min_time_ms;
        result.max_time_ms    = wr.max_time_ms;
        result.error_message  = wr.error_message;
        return result;
    }

    // ── Observabilidade ───────────────────────────────────────────────────────

    bool health_check() override { return connected_; }

    WorkerStatus get_status() override {
        WorkerStatus s;
        if (!connected_) return s;

        s.worker_id           = "inprocess";
        s.total_requests      = total_requests_;
        s.successful_requests = successful_requests_;
        s.failed_requests     = failed_requests_;
        s.active_requests     = 0;  // chamadas são síncronas — nunca há sobreposição

        auto now   = std::chrono::steady_clock::now();
        s.uptime_seconds = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count());

        s.loaded_models = engine_->get_loaded_model_ids();

        auto info = engine_->get_engine_info();
        s.supported_backends = info.supported_backends;

        return s;
    }

    ServerMetrics get_metrics() override {
        ServerMetrics result;
        if (!connected_) return result;

        auto now = std::chrono::steady_clock::now();
        result.total_requests      = total_requests_;
        result.successful_requests = successful_requests_;
        result.failed_requests     = failed_requests_;
        result.active_requests     = 0;
        result.uptime_seconds      = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count());

        for (const auto& id : engine_->get_loaded_model_ids()) {
            const auto* rm = engine_->get_model_metrics(id);
            if (!rm) continue;

            ModelMetrics mm;
            mm.total_inferences  = rm->total_inferences;
            mm.failed_inferences = rm->failed_inferences;
            mm.avg_ms            = rm->avg_time_ms();
            mm.min_ms            = rm->min_time_ms;
            mm.max_ms            = rm->max_time_ms;
            mm.p95_ms            = rm->p95_time_ms();
            mm.p99_ms            = rm->p99_time_ms();
            mm.total_time_ms     = rm->total_time_ms;

            // loaded_at_unix via proto
            auto proto        = engine_->get_model_info(id);
            mm.loaded_at_unix = proto.loaded_at_unix();

            result.per_model[id] = mm;
        }
        return result;
    }

    // ── Descoberta de ficheiros ───────────────────────────────────────────────

    std::vector<AvailableModel> list_available_models(
        const std::string& directory) override {

        std::vector<AvailableModel> result;
        if (!connected_) return result;

        namespace fs = std::filesystem;

        std::string dir = directory.empty() ? "./models" : directory;
        if (!fs::exists(dir) || !fs::is_directory(dir)) return result;

        auto& registry = worker::BackendRegistry::instance();
        auto  exts     = registry.registered_extensions();

        // Constrói mapa path → model_id para modelos já carregados.
        std::map<std::string, std::string> loaded_paths;
        for (const auto& id : engine_->get_loaded_model_ids()) {
            auto proto = engine_->get_model_info(id);
            loaded_paths[proto.model_path()] = id;
        }

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;

            AvailableModel am;
            am.filename        = entry.path().filename().string();
            am.path            = entry.path().string();
            am.extension       = ext;
            am.backend         = backend_str(registry.detect_backend(am.path));
            am.file_size_bytes = static_cast<int64_t>(entry.file_size());

            auto it = loaded_paths.find(am.path);
            if (it != loaded_paths.end()) {
                am.is_loaded = true;
                am.loaded_as = it->second;
            }

            result.push_back(std::move(am));
        }
        return result;
    }

private:
    std::unique_ptr<worker::InferenceEngine> engine_;
    bool connected_ = false;

    // Contadores locais (o engine já mantém os seus próprios por modelo,
    // mas não expõe um total agregado — mantemos aqui para get_status()).
    uint64_t total_requests_      = 0;
    uint64_t successful_requests_ = 0;
    uint64_t failed_requests_     = 0;
    std::chrono::steady_clock::time_point start_time_;

    // Versões guardadas no load_model (o InferenceEngine não as expõe).
    std::map<std::string, std::string> versions_;
};

// =============================================================================
// §4  InferenceClient  — detecção de modo e delegação
// =============================================================================

static bool is_inprocess_target(const std::string& target) {
    return target == "inprocess"
        || target == "in_process"
        || target == "local";
}

InferenceClient::InferenceClient(const std::string& target) {
    if (is_inprocess_target(target))
        backend_ = std::make_unique<InProcessBackend>();
    else
        backend_ = std::make_unique<GrpcClientBackend>(target);
}

// O destrutor deve ser definido no .cpp porque IClientBackend é incompleto
// no header. O compilador precisa do destrutor completo de IClientBackend
// para gerar o delete correcto em unique_ptr.
InferenceClient::~InferenceClient() = default;

bool InferenceClient::connect()            { return backend_->connect();       }
bool InferenceClient::is_connected() const { return backend_->is_connected();  }

bool InferenceClient::load_model(const std::string& id,
                                  const std::string& path,
                                  const std::string& version) {
    return backend_->load_model(id, path, version);
}

bool InferenceClient::unload_model(const std::string& id) {
    return backend_->unload_model(id);
}

PredictionResult InferenceClient::predict(
    const std::string& id,
    const std::map<std::string, std::vector<float>>& inputs) {
    return backend_->predict(id, inputs);
}

std::vector<PredictionResult> InferenceClient::batch_predict(
    const std::string& id,
    const std::vector<std::map<std::string, std::vector<float>>>& batch) {
    return backend_->batch_predict(id, batch);
}

std::vector<ModelInfo> InferenceClient::list_models() {
    return backend_->list_models();
}

ModelInfo InferenceClient::get_model_info(const std::string& id) {
    return backend_->get_model_info(id);
}

ValidationResult InferenceClient::validate_model(const std::string& path) {
    return backend_->validate_model(path);
}

WarmupResult InferenceClient::warmup_model(const std::string& id, uint32_t n) {
    return backend_->warmup_model(id, n);
}

bool InferenceClient::health_check()    { return backend_->health_check();  }
WorkerStatus InferenceClient::get_status() { return backend_->get_status(); }
ServerMetrics InferenceClient::get_metrics() { return backend_->get_metrics(); }

std::vector<AvailableModel> InferenceClient::list_available_models(
    const std::string& directory) {
    return backend_->list_available_models(directory);
}

}  // namespace client
}  // namespace mlinference

// // =============================================================================
// // inference_client.cpp — Implementation of the AsaMiia client library
// // =============================================================================

// #include "client/inference_client.hpp"
// #include <iostream>
// #include <chrono>

// namespace mlinference {
// namespace client {

// // ============================================
// // Construction / Connection
// // ============================================

// InferenceClient::InferenceClient(const std::string& server_address)
//     : server_address_(server_address) {}

// InferenceClient::~InferenceClient() = default;

// bool InferenceClient::connect() {
//     channel_ = grpc::CreateChannel(server_address_, grpc::InsecureChannelCredentials());
//     stub_ = worker::WorkerService::NewStub(channel_);

//     auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
//     connected_ = channel_->WaitForConnected(deadline);

//     if (connected_) {
//         std::cout << "Connected to " << server_address_ << std::endl;
//     } else {
//         std::cerr << "Failed to connect to " << server_address_ << std::endl;
//     }
//     return connected_;
// }

// bool InferenceClient::is_connected() const {
//     return connected_;
// }

// // ============================================
// // Model Lifecycle
// // ============================================

// bool InferenceClient::load_model(const std::string& model_id,
//                                   const std::string& model_path,
//                                   const std::string& version) {
//     if (!connected_) return false;

//     grpc::ClientContext ctx;
//     worker::LoadModelRequest req;
//     worker::LoadModelResponse resp;

//     req.set_model_id(model_id);
//     req.set_model_path(model_path);
//     req.set_version(version);

//     auto status = stub_->LoadModel(&ctx, req, &resp);
//     if (!status.ok()) {
//         std::cerr << "LoadModel RPC failed: " << status.error_message() << std::endl;
//         return false;
//     }

//     if (resp.success()) {
//         std::cout << "Model loaded: " << model_id << std::endl;
//     } else {
//         std::cerr << "LoadModel failed: " << resp.error_message() << std::endl;
//     }
//     return resp.success();
// }

// bool InferenceClient::unload_model(const std::string& model_id) {
//     if (!connected_) return false;

//     grpc::ClientContext ctx;
//     worker::UnloadModelRequest req;
//     worker::UnloadModelResponse resp;

//     req.set_model_id(model_id);

//     auto status = stub_->UnloadModel(&ctx, req, &resp);
//     if (!status.ok()) {
//         std::cerr << "UnloadModel RPC failed: " << status.error_message() << std::endl;
//         return false;
//     }
//     return resp.success();
// }

// // ============================================
// // Inference
// // ============================================

// PredictionResult InferenceClient::predict(
//     const std::string& model_id,
//     const std::map<std::string, std::vector<float>>& inputs) {

//     PredictionResult result;
//     if (!connected_) {
//         result.error_message = "Not connected";
//         return result;
//     }

//     grpc::ClientContext ctx;
//     worker::PredictRequest req;
//     worker::PredictResponse resp;

//     req.set_model_id(model_id);
//     for (const auto& [name, data] : inputs) {
//         set_tensor_data(&(*req.mutable_inputs())[name], data);
//     }

//     auto status = stub_->Predict(&ctx, req, &resp);
//     if (!status.ok()) {
//         result.error_message = "RPC failed: " + status.error_message();
//         return result;
//     }

//     result.success           = resp.success();
//     result.inference_time_ms = resp.inference_time_ms();
//     result.error_message     = resp.error_message();

//     if (resp.success()) {
//         for (const auto& [name, tensor] : resp.outputs()) {
//             result.outputs[name] = get_tensor_data(tensor);
//         }
//     }

//     return result;
// }

// std::vector<PredictionResult> InferenceClient::batch_predict(
//     const std::string& model_id,
//     const std::vector<std::map<std::string, std::vector<float>>>& batch_inputs) {

//     std::vector<PredictionResult> results;
//     if (!connected_) return results;

//     grpc::ClientContext ctx;
//     worker::BatchPredictRequest req;
//     worker::BatchPredictResponse resp;

//     req.set_model_id(model_id);
//     for (const auto& inputs : batch_inputs) {
//         auto* single = req.add_requests();
//         single->set_model_id(model_id);
//         for (const auto& [name, data] : inputs) {
//             set_tensor_data(&(*single->mutable_inputs())[name], data);
//         }
//     }

//     auto status = stub_->BatchPredict(&ctx, req, &resp);
//     if (!status.ok()) return results;

//     for (const auto& single_resp : resp.responses()) {
//         PredictionResult r;
//         r.success            = single_resp.success();
//         r.inference_time_ms  = single_resp.inference_time_ms();
//         r.error_message      = single_resp.error_message();

//         if (single_resp.success()) {
//             for (const auto& [name, tensor] : single_resp.outputs()) {
//                 r.outputs[name] = get_tensor_data(tensor);
//             }
//         }
//         results.push_back(std::move(r));
//     }

//     return results;
// }

// // ============================================
// // Introspection
// // ============================================

// std::vector<ModelInfo> InferenceClient::list_models() {
//     std::vector<ModelInfo> result;
//     if (!connected_) return result;

//     grpc::ClientContext ctx;
//     worker::ListModelsRequest req;
//     worker::ListModelsResponse resp;

//     auto status = stub_->ListModels(&ctx, req, &resp);
//     if (!status.ok()) return result;

//     for (const auto& proto : resp.models()) {
//         result.push_back(proto_to_model_info(proto));
//     }
//     return result;
// }

// ModelInfo InferenceClient::get_model_info(const std::string& model_id) {
//     ModelInfo result;
//     if (!connected_) return result;

//     grpc::ClientContext ctx;
//     worker::GetModelInfoRequest req;
//     worker::GetModelInfoResponse resp;

//     req.set_model_id(model_id);

//     auto status = stub_->GetModelInfo(&ctx, req, &resp);
//     if (!status.ok() || !resp.success()) return result;

//     return proto_to_model_info(resp.model_info());
// }

// ValidationResult InferenceClient::validate_model(const std::string& model_path) {
//     ValidationResult result;
//     if (!connected_) return result;

//     grpc::ClientContext ctx;
//     worker::ValidateModelRequest req;
//     worker::ValidateModelResponse resp;

//     req.set_model_path(model_path);

//     auto status = stub_->ValidateModel(&ctx, req, &resp);
//     if (!status.ok()) {
//         result.error_message = "RPC failed: " + status.error_message();
//         return result;
//     }

//     result.valid         = resp.valid();
//     result.backend       = backend_type_to_string(resp.backend());
//     result.error_message = resp.error_message();

//     for (const auto& w : resp.warnings()) {
//         result.warnings.push_back(w);
//     }
//     for (const auto& ts : resp.inputs()) {
//         ModelInfo::TensorSpec spec;
//         spec.name  = ts.name();
//         spec.dtype = dtype_to_string(ts.dtype());
//         for (int64_t d : ts.shape()) spec.shape.push_back(d);
//         result.inputs.push_back(std::move(spec));
//     }
//     for (const auto& ts : resp.outputs()) {
//         ModelInfo::TensorSpec spec;
//         spec.name  = ts.name();
//         spec.dtype = dtype_to_string(ts.dtype());
//         for (int64_t d : ts.shape()) spec.shape.push_back(d);
//         result.outputs.push_back(std::move(spec));
//     }

//     return result;
// }

// WarmupResult InferenceClient::warmup_model(const std::string& model_id,
//                                             uint32_t num_runs) {
//     WarmupResult result;
//     if (!connected_) return result;

//     grpc::ClientContext ctx;
//     worker::WarmupModelRequest req;
//     worker::WarmupModelResponse resp;

//     req.set_model_id(model_id);
//     req.set_num_runs(num_runs);

//     auto status = stub_->WarmupModel(&ctx, req, &resp);
//     if (!status.ok()) {
//         result.error_message = "RPC failed: " + status.error_message();
//         return result;
//     }

//     result.success        = resp.success();
//     result.runs_completed = resp.runs_completed();
//     result.avg_time_ms    = resp.avg_time_ms();
//     result.min_time_ms    = resp.min_time_ms();
//     result.max_time_ms    = resp.max_time_ms();
//     result.error_message  = resp.error_message();
//     return result;
// }

// // ============================================
// // Observability
// // ============================================

// bool InferenceClient::health_check() {
//     if (!connected_) return false;

//     grpc::ClientContext ctx;
//     worker::HealthCheckRequest req;
//     worker::HealthCheckResponse resp;

//     auto status = stub_->HealthCheck(&ctx, req, &resp);
//     return status.ok() && resp.healthy();
// }

// WorkerStatus InferenceClient::get_status() {
//     WorkerStatus result;
//     if (!connected_) return result;

//     grpc::ClientContext ctx;
//     worker::GetStatusRequest req;
//     worker::GetStatusResponse resp;

//     auto status = stub_->GetStatus(&ctx, req, &resp);
//     if (!status.ok()) return result;

//     result.worker_id           = resp.worker_id();
//     result.total_requests      = resp.metrics().total_requests();
//     result.successful_requests = resp.metrics().successful_requests();
//     result.failed_requests     = resp.metrics().failed_requests();
//     result.active_requests     = resp.metrics().active_requests();
//     result.uptime_seconds      = resp.metrics().uptime_seconds();

//     for (const auto& id : resp.loaded_model_ids()) {
//         result.loaded_models.push_back(id);
//     }
//     for (const auto& b : resp.capabilities().supported_backends()) {
//         result.supported_backends.push_back(b);
//     }

//     return result;
// }

// ServerMetrics InferenceClient::get_metrics() {
//     ServerMetrics result;
//     if (!connected_) return result;

//     grpc::ClientContext ctx;
//     worker::GetMetricsRequest req;
//     worker::GetMetricsResponse resp;

//     auto status = stub_->GetMetrics(&ctx, req, &resp);
//     if (!status.ok()) return result;

//     // Worker-level counters
//     const auto& wm         = resp.worker_metrics();
//     result.total_requests      = wm.total_requests();
//     result.successful_requests = wm.successful_requests();
//     result.failed_requests     = wm.failed_requests();
//     result.active_requests     = wm.active_requests();
//     result.uptime_seconds      = wm.uptime_seconds();

//     // Per-model breakdown
//     for (const auto& [id, m] : resp.per_model_metrics()) {
//         ModelMetrics mm;
//         mm.total_inferences  = m.total_inferences();
//         mm.failed_inferences = m.failed_inferences();
//         mm.avg_ms            = m.avg_inference_time_ms();
//         mm.min_ms            = m.min_inference_time_ms();
//         mm.max_ms            = m.max_inference_time_ms();
//         mm.p95_ms            = m.p95_inference_time_ms();
//         mm.p99_ms            = m.p99_inference_time_ms();
//         mm.total_time_ms     = m.total_inference_time_ms();
//         mm.last_used_at_unix = m.last_used_at_unix();
//         mm.loaded_at_unix    = m.loaded_at_unix();
//         result.per_model[id] = mm;
//     }

//     return result;
// }

// // ============================================
// // File Discovery
// // ============================================

// std::vector<AvailableModel> InferenceClient::list_available_models(
//     const std::string& directory) {

//     std::vector<AvailableModel> result;
//     if (!connected_) return result;

//     grpc::ClientContext ctx;
//     worker::ListAvailableModelsRequest req;
//     worker::ListAvailableModelsResponse resp;

//     if (!directory.empty()) req.set_directory(directory);

//     auto status = stub_->ListAvailableModels(&ctx, req, &resp);
//     if (!status.ok()) {
//         std::cerr << "ListAvailableModels RPC failed: " << status.error_message() << std::endl;
//         return result;
//     }

//     for (const auto& m : resp.models()) {
//         AvailableModel am;
//         am.filename        = m.filename();
//         am.path            = m.path();
//         am.extension       = m.extension();
//         am.backend         = backend_type_to_string(m.backend());
//         am.file_size_bytes = m.file_size_bytes();
//         am.is_loaded       = m.is_loaded();
//         am.loaded_as       = m.loaded_as();
//         result.push_back(std::move(am));
//     }
//     return result;
// }

// // ============================================
// // Helpers
// // ============================================

// void InferenceClient::set_tensor_data(common::Tensor* tensor,
//                                        const std::vector<float>& data) {
//     tensor->set_dtype(common::FLOAT32);
//     tensor->add_shape(data.size());
//     tensor->mutable_data()->assign(
//         reinterpret_cast<const char*>(data.data()),
//         data.size() * sizeof(float));
// }

// std::vector<float> InferenceClient::get_tensor_data(const common::Tensor& tensor) {
//     const auto& bytes = tensor.data();
//     if (tensor.dtype() != common::FLOAT32) return {};

//     const float* ptr = reinterpret_cast<const float*>(bytes.data());
//     size_t count     = bytes.size() / sizeof(float);
//     return std::vector<float>(ptr, ptr + count);
// }

// ModelInfo InferenceClient::proto_to_model_info(const common::ModelInfo& proto) {
//     ModelInfo info;
//     info.model_id           = proto.model_id();
//     info.version            = proto.version();
//     info.backend            = backend_type_to_string(proto.backend());
//     info.description        = proto.description();
//     info.author             = proto.author();
//     info.memory_usage_bytes = proto.memory_usage_bytes();
//     info.is_warmed_up       = proto.is_warmed_up();

//     for (const auto& ts : proto.inputs()) {
//         ModelInfo::TensorSpec spec;
//         spec.name        = ts.name();
//         spec.dtype       = dtype_to_string(ts.dtype());
//         spec.description = ts.description();
//         for (int64_t d : ts.shape()) spec.shape.push_back(d);
//         info.inputs.push_back(std::move(spec));
//     }
//     for (const auto& ts : proto.outputs()) {
//         ModelInfo::TensorSpec spec;
//         spec.name        = ts.name();
//         spec.dtype       = dtype_to_string(ts.dtype());
//         spec.description = ts.description();
//         for (int64_t d : ts.shape()) spec.shape.push_back(d);
//         info.outputs.push_back(std::move(spec));
//     }
//     for (const auto& [k, v] : proto.tags()) {
//         info.tags[k] = v;
//     }

//     return info;
// }

// std::string InferenceClient::backend_type_to_string(common::BackendType type) {
//     switch (type) {
//         case common::BACKEND_ONNX:   return "onnx";
//         case common::BACKEND_PYTHON: return "python";
//         default:                      return "unknown";
//     }
// }

// std::string InferenceClient::dtype_to_string(common::DataType type) {
//     switch (type) {
//         case common::FLOAT32: return "float32";
//         case common::FLOAT64: return "float64";
//         case common::INT32:   return "int32";
//         case common::INT64:   return "int64";
//         case common::UINT8:   return "uint8";
//         case common::BOOL:    return "bool";
//         case common::STRING:  return "string";
//         case common::FLOAT16: return "float16";
//         default:               return "unknown";
//     }
// }

// }  // namespace client
// }  // namespace mlinference
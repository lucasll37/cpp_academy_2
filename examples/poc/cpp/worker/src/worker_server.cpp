// =============================================================================
// worker_server.cpp — gRPC service implementation (com suporte a S3)
// =============================================================================

#include "worker/worker_server.hpp"
#include "worker/backend_registry.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

namespace fs = std::filesystem;

namespace mlinference {
namespace worker {

// =============================================================================
// WorkerServiceImpl — construtor / destrutor
// =============================================================================

WorkerServiceImpl::WorkerServiceImpl(
    const std::string& worker_id,
    bool               enable_gpu,
    uint32_t           num_threads,
    const std::string& models_dir,
    std::shared_ptr<S3ModelSource> s3_source)
    : worker_id_(worker_id)
    , models_dir_(models_dir)
    , inference_engine_(std::make_shared<InferenceEngine>(enable_gpu, 0, num_threads))
    , s3_source_(std::move(s3_source))
    , start_time_(std::chrono::steady_clock::now()) {

    std::cout << "[WorkerService] Initialized: " << worker_id_
              << " (models: " << models_dir_ << ")" << std::endl;

    if (s3_source_ && s3_source_->enabled()) {
        std::cout << "[WorkerService] S3 habilitado: s3://"
                  << s3_source_->config().bucket << "/"
                  << s3_source_->config().prefix << std::endl;
    }
}

WorkerServiceImpl::~WorkerServiceImpl() {
    std::cout << "[WorkerService] Destroyed: " << worker_id_ << std::endl;
}

// =============================================================================
// Inference RPCs
// =============================================================================

grpc::Status WorkerServiceImpl::Predict(
    grpc::ServerContext* /*context*/,
    const PredictRequest* request,
    PredictResponse* response) {

    active_requests_++;
    total_requests_++;

    std::map<std::string, std::vector<float>> inputs;
    for (const auto& [name, tensor] : request->inputs()) {
        const auto& bytes = tensor.data();
        const float* ptr  = reinterpret_cast<const float*>(bytes.data());
        size_t count      = bytes.size() / sizeof(float);
        inputs[name]      = std::vector<float>(ptr, ptr + count);
    }

    auto result = inference_engine_->predict(request->model_id(), inputs);

    response->set_success(result.success);
    response->set_inference_time_ms(result.inference_time_ms);
    response->set_error_message(result.error_message);

    if (result.success) {
        successful_requests_++;
        for (const auto& [name, data] : result.outputs) {
            auto& tensor = (*response->mutable_outputs())[name];
            tensor.set_data(std::string(
                reinterpret_cast<const char*>(data.data()),
                data.size() * sizeof(float)));
        }
    } else {
        failed_requests_++;
    }

    active_requests_--;
    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::PredictStream(
    grpc::ServerContext* /*context*/,
    grpc::ServerReaderWriter<PredictResponse, PredictRequest>* stream) {

    PredictRequest request;
    while (stream->Read(&request)) {
        PredictResponse response;
        Predict(nullptr, &request, &response);
        stream->Write(response);
    }
    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::BatchPredict(
    grpc::ServerContext* /*context*/,
    const BatchPredictRequest* request,
    BatchPredictResponse* response) {

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& single_req : request->requests()) {
        auto* single_resp = response->add_responses();
        grpc::Status s = Predict(nullptr, &single_req, single_resp);
        if (!s.ok()) {
            single_resp->set_success(false);
            single_resp->set_error_message(s.error_message());
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    response->set_total_time_ms(
        std::chrono::duration<double, std::milli>(end - start).count());

    return grpc::Status::OK;
}

// =============================================================================
// Model Lifecycle RPCs
// =============================================================================

grpc::Status WorkerServiceImpl::LoadModel(
    grpc::ServerContext* /*context*/,
    const LoadModelRequest* request,
    LoadModelResponse* response) {

    std::string model_id   = request->model_id();
    std::string model_path = request->model_path();

    std::cout << "[WorkerService] LoadModel: " << model_id
              << " from " << model_path << std::endl;

    // ------------------------------------------------------------------
    // S3: se o path for uma URI s3://, fazer download on-demand para cache
    // ------------------------------------------------------------------
    S3ModelEntry s3_entry_used;
    bool         from_s3 = false;

    if (s3_source_ && s3_source_->enabled() &&
        s3_source_->is_s3_uri(model_path)) {

        // Localiza a entrada S3 correspondente à URI
        std::string requested_key = S3ModelSource::key_from_uri(model_path);
        std::vector<S3ModelEntry> entries;
        try {
            entries = s3_source_->list();
        } catch (const std::exception& ex) {
            response->set_success(false);
            response->set_error_message(
                std::string("Falha ao listar S3 para localizar modelo: ") + ex.what());
            return grpc::Status::OK;
        }

        auto it = std::find_if(entries.begin(), entries.end(),
                               [&](const S3ModelEntry& e) {
                                   return e.key == requested_key;
                               });

        if (it == entries.end()) {
            response->set_success(false);
            response->set_error_message(
                "Modelo não encontrado no bucket: " + model_path);
            return grpc::Status::OK;
        }

        try {
            model_path    = s3_source_->download(*it);
            s3_entry_used = *it;
            from_s3       = true;
        } catch (const std::exception& ex) {
            response->set_success(false);
            response->set_error_message(
                std::string("Falha ao baixar modelo do S3: ") + ex.what());
            return grpc::Status::OK;
        }
    }

    // ------------------------------------------------------------------
    // Carrega no InferenceEngine (path local — disco ou cache S3)
    // ------------------------------------------------------------------
    std::map<std::string, std::string> config(
        request->backend_config().begin(),
        request->backend_config().end());

    bool ok = inference_engine_->load_model(
        model_id, model_path, request->force_backend(), config);

    response->set_success(ok);

    if (ok) {
        auto info = inference_engine_->get_model_info(model_id);
        if (!request->version().empty())
            info.set_version(request->version());
        *response->mutable_model_info() = std::move(info);

        // Registra origem S3 para que evict() possa ser chamado no UnloadModel
        if (from_s3)
            loaded_s3_models_[model_id] = s3_entry_used;

    } else {
        response->set_error_message(
            "Failed to load model: " + model_id);

        // Falha no carregamento — libera cache S3 imediatamente
        if (from_s3)
            s3_source_->evict(s3_entry_used);
    }

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::UnloadModel(
    grpc::ServerContext* /*context*/,
    const UnloadModelRequest* request,
    UnloadModelResponse* response) {

    std::cout << "[WorkerService] UnloadModel: " << request->model_id() << std::endl;

    bool ok = inference_engine_->unload_model(request->model_id());
    response->set_success(ok);
    response->set_message(ok ? "Model unloaded" : "Model not found");

    // ------------------------------------------------------------------
    // S3: remover arquivo de cache após descarregar do engine
    // ------------------------------------------------------------------
    if (ok && s3_source_) {
        auto it = loaded_s3_models_.find(request->model_id());
        if (it != loaded_s3_models_.end()) {
            s3_source_->evict(it->second);
            loaded_s3_models_.erase(it);
        }
    }

    return grpc::Status::OK;
}

// =============================================================================
// Model Introspection RPCs
// =============================================================================

grpc::Status WorkerServiceImpl::ListModels(
    grpc::ServerContext* /*context*/,
    const ListModelsRequest* /*request*/,
    ListModelsResponse* response) {

    auto ids = inference_engine_->get_loaded_model_ids();
    for (const auto& id : ids) {
        auto info = inference_engine_->get_model_info(id);
        *response->add_models() = std::move(info);
    }
    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::GetModelInfo(
    grpc::ServerContext* /*context*/,
    const GetModelInfoRequest* request,
    GetModelInfoResponse* response) {

    if (!inference_engine_->is_model_loaded(request->model_id())) {
        response->set_success(false);
        response->set_error_message("Model not loaded: " + request->model_id());
        return grpc::Status::OK;
    }

    response->set_success(true);
    auto info = inference_engine_->get_model_info(request->model_id());
    *response->mutable_model_info() = std::move(info);

    auto* m = inference_engine_->get_model_metrics(request->model_id());
    if (m)
        fill_runtime_metrics(*m, response->mutable_metrics());

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::ValidateModel(
    grpc::ServerContext* /*context*/,
    const ValidateModelRequest* request,
    ValidateModelResponse* response) {

    auto result = inference_engine_->validate_model(
        request->model_path(), request->force_backend());

    response->set_valid(result.valid);
    response->set_backend(result.backend);
    response->set_error_message(result.error_message);
    for (const auto& w : result.warnings) response->add_warnings(w);

    for (const auto& spec : result.schema.inputs) {
        auto* ts = response->add_inputs();
        ts->set_name(spec.name);
        ts->set_dtype(spec.dtype);
        for (int64_t dim : spec.shape) ts->add_shape(dim);
    }
    for (const auto& spec : result.schema.outputs) {
        auto* ts = response->add_outputs();
        ts->set_name(spec.name);
        ts->set_dtype(spec.dtype);
        for (int64_t dim : spec.shape) ts->add_shape(dim);
    }

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::WarmupModel(
    grpc::ServerContext* /*context*/,
    const WarmupModelRequest* request,
    WarmupModelResponse* response) {

    uint32_t runs = request->num_runs() > 0 ? request->num_runs() : 5;
    auto result   = inference_engine_->warmup_model(request->model_id(), runs);

    response->set_success(result.success);
    response->set_runs_completed(result.runs_completed);
    response->set_avg_time_ms(result.avg_time_ms);
    response->set_min_time_ms(result.min_time_ms);
    response->set_max_time_ms(result.max_time_ms);
    response->set_error_message(result.error_message);

    return grpc::Status::OK;
}

// =============================================================================
// Worker Observability RPCs
// =============================================================================

grpc::Status WorkerServiceImpl::GetStatus(
    grpc::ServerContext* /*context*/,
    const GetStatusRequest* /*request*/,
    GetStatusResponse* response) {

    response->set_worker_id(worker_id_);
    response->set_status(common::WORKER_READY);

    auto* caps        = response->mutable_capabilities();
    auto  engine_info = inference_engine_->get_engine_info();
    caps->set_supports_gpu(engine_info.gpu_enabled);
    caps->set_num_cpu_cores(std::thread::hardware_concurrency());
    for (const auto& b : engine_info.supported_backends)
        caps->add_supported_backends(b);

    *response->mutable_metrics() = build_worker_metrics();

    for (const auto& id : inference_engine_->get_loaded_model_ids())
        response->add_loaded_model_ids(id);

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::GetMetrics(
    grpc::ServerContext* /*context*/,
    const GetMetricsRequest* /*request*/,
    GetMetricsResponse* response) {

    *response->mutable_worker_metrics() = build_worker_metrics();

    for (const auto& id : inference_engine_->get_loaded_model_ids()) {
        auto* m = inference_engine_->get_model_metrics(id);
        if (m) {
            auto* dst = &(*response->mutable_per_model_metrics())[id];
            fill_runtime_metrics(*m, dst);
            auto info = inference_engine_->get_model_info(id);
            dst->set_loaded_at_unix(info.loaded_at_unix());
        }
    }

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::HealthCheck(
    grpc::ServerContext* /*context*/,
    const HealthCheckRequest* /*request*/,
    HealthCheckResponse* response) {

    response->set_healthy(true);
    response->set_message("Worker " + worker_id_ + " is healthy");
    return grpc::Status::OK;
}

// =============================================================================
// File Discovery RPCs
// =============================================================================

grpc::Status WorkerServiceImpl::ListAvailableModels(
    grpc::ServerContext* /*context*/,
    const ListAvailableModelsRequest* request,
    ListAvailableModelsResponse* response) {

    std::string dir = request->directory().empty() ? models_dir_ : request->directory();
    response->set_directory(dir);

    auto& registry   = BackendRegistry::instance();
    auto  extensions = registry.registered_extensions();

    // ------------------------------------------------------------------
    // 1. Modelos locais (comportamento original)
    // ------------------------------------------------------------------
    if (fs::exists(dir) && fs::is_directory(dir)) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            if (std::find(extensions.begin(), extensions.end(), ext) ==
                extensions.end()) continue;

            auto* m = response->add_models();
            m->set_filename(entry.path().filename().string());
            m->set_path(entry.path().string());
            m->set_extension(ext);
            m->set_backend(registry.detect_backend(entry.path().string()));
            m->set_file_size_bytes(static_cast<int64_t>(entry.file_size()));

            // Verifica se já está carregado
            for (const auto& id : inference_engine_->get_loaded_model_ids()) {
                auto info = inference_engine_->get_model_info(id);
                if (info.model_path() == entry.path().string()) {
                    m->set_is_loaded(true);
                    m->set_loaded_as(id);
                    break;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Modelos S3 (apenas metadados — sem download)
    // ------------------------------------------------------------------
    if (s3_source_ && s3_source_->enabled()) {
        std::vector<S3ModelEntry> s3_entries;
        try {
            s3_entries = s3_source_->list();
        } catch (const std::exception& ex) {
            // Não falha o RPC por causa do S3 — loga e continua
            std::cerr << "[WorkerService] Aviso: falha ao listar S3: "
                      << ex.what() << std::endl;
        }

        for (const auto& entry : s3_entries) {
            // Evita duplicatas: se o arquivo já está no cache local e
            // foi listado acima (por ter sido carregado antes), não
            // adiciona novamente. Usa a URI s3:// como path identificador.
            auto* m = response->add_models();
            m->set_filename(entry.filename);
            m->set_path(entry.s3_uri);           // path = URI s3:// — usado no LoadModel
            m->set_extension(entry.extension);
            m->set_backend(registry.detect_backend(entry.filename));
            m->set_file_size_bytes(entry.size_bytes);

            // Verifica se já está carregado (via mapa de modelos S3 ativos)
            auto it = std::find_if(
                loaded_s3_models_.begin(), loaded_s3_models_.end(),
                [&](const auto& kv) { return kv.second.s3_uri == entry.s3_uri; });
            if (it != loaded_s3_models_.end()) {
                m->set_is_loaded(true);
                m->set_loaded_as(it->first);
            }
        }
    }

    return grpc::Status::OK;
}

// =============================================================================
// Helpers
// =============================================================================

void WorkerServiceImpl::fill_model_info(
    const common::ModelInfo& src,
    common::ModelInfo* dst) const {
    *dst = src;
}

void WorkerServiceImpl::fill_runtime_metrics(
    const RuntimeMetrics& src,
    ModelRuntimeMetrics*  dst) const {

    uint64_t ok = src.total_inferences - src.failed_inferences;

    dst->set_total_inferences(src.total_inferences);
    dst->set_failed_inferences(src.failed_inferences);
    dst->set_avg_inference_time_ms(src.avg_time_ms());
    dst->set_min_inference_time_ms((ok > 0) ? src.min_time_ms : 0.0);
    dst->set_max_inference_time_ms(src.max_time_ms);
    dst->set_p95_inference_time_ms(src.p95_time_ms());
    dst->set_p99_inference_time_ms(src.p99_time_ms());
    dst->set_total_inference_time_ms(src.total_time_ms);
}

common::WorkerMetrics WorkerServiceImpl::build_worker_metrics() const {
    common::WorkerMetrics m;
    m.set_total_requests(total_requests_.load());
    m.set_successful_requests(successful_requests_.load());
    m.set_failed_requests(failed_requests_.load());
    m.set_active_requests(active_requests_.load());

    auto now    = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        now - start_time_);
    m.set_uptime_seconds(uptime.count());

    uint64_t ok = successful_requests_.load();
    if (ok > 0) {
        double   sum   = 0.0;
        uint64_t count = 0;
        for (const auto& id : inference_engine_->get_loaded_model_ids()) {
            auto* mm = inference_engine_->get_model_metrics(id);
            if (mm && mm->total_inferences > mm->failed_inferences) {
                sum   += mm->total_time_ms;
                count += mm->total_inferences - mm->failed_inferences;
            }
        }
        if (count > 0)
            m.set_avg_response_time_ms(sum / static_cast<double>(count));
    }

    return m;
}

// =============================================================================
// WorkerServer
// =============================================================================

WorkerServer::WorkerServer(
    const std::string& worker_id,
    const std::string& server_address,
    bool               enable_gpu,
    uint32_t           num_threads,
    const std::string& models_dir,
    std::shared_ptr<S3ModelSource> s3_source)
    : worker_id_(worker_id)
    , server_address_(server_address) {

    service_ = std::make_unique<WorkerServiceImpl>(
        worker_id, enable_gpu, num_threads, models_dir, std::move(s3_source));
}

WorkerServer::~WorkerServer() {
    stop();
}

void WorkerServer::run() {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());

    server_ = builder.BuildAndStart();
    std::cout << "[WorkerServer] Listening on " << server_address_ << std::endl;
    server_->Wait();
}

void WorkerServer::stop() {
    if (server_)
        server_->Shutdown();
}

}  // namespace worker
}  // namespace mlinference



// // =============================================================================
// // worker_server.cpp — gRPC service implementation
// // =============================================================================

// #include "worker/worker_server.hpp"
// #include "worker/backend_registry.hpp"
// #include <iostream>
// #include <thread>
// #include <filesystem>

// namespace mlinference {
// namespace worker {

// // ============================================
// // WorkerServiceImpl
// // ============================================

// WorkerServiceImpl::WorkerServiceImpl(
//     const std::string& worker_id,
//     bool enable_gpu,
//     uint32_t num_threads,
//     const std::string& models_dir)
//     : worker_id_(worker_id)
//     , models_dir_(models_dir)
//     , inference_engine_(std::make_shared<InferenceEngine>(enable_gpu, 0, num_threads))
//     , start_time_(std::chrono::steady_clock::now()) {

//     std::cout << "[WorkerService] Initialized: " << worker_id_
//               << " (models: " << models_dir_ << ")" << std::endl;
// }

// WorkerServiceImpl::~WorkerServiceImpl() {
//     std::cout << "[WorkerService] Destroyed: " << worker_id_ << std::endl;
// }

// // ============================================
// // Inference RPCs
// // ============================================

// grpc::Status WorkerServiceImpl::Predict(
//     grpc::ServerContext* /*context*/,
//     const PredictRequest* request,
//     PredictResponse* response) {

//     active_requests_++;
//     total_requests_++;

//     // Deserialise protobuf tensors → flat float vectors
//     std::map<std::string, std::vector<float>> inputs;
//     for (const auto& [name, tensor] : request->inputs()) {
//         const auto& bytes = tensor.data();
//         const float* ptr = reinterpret_cast<const float*>(bytes.data());
//         size_t count = bytes.size() / sizeof(float);
//         inputs[name] = std::vector<float>(ptr, ptr + count);
//     }

//     auto result = inference_engine_->predict(request->model_id(), inputs);

//     response->set_success(result.success);
//     response->set_inference_time_ms(result.inference_time_ms);

//     if (result.success) {
//         for (const auto& [name, data] : result.outputs) {
//             auto& tensor = (*response->mutable_outputs())[name];
//             tensor.set_data(
//                 reinterpret_cast<const char*>(data.data()),
//                 data.size() * sizeof(float));
//             tensor.add_shape(data.size());
//             tensor.set_dtype(common::FLOAT32);
//         }
//         successful_requests_++;
//     } else {
//         response->set_error_message(result.error_message);
//         failed_requests_++;
//     }

//     active_requests_--;
//     return grpc::Status::OK;
// }

// grpc::Status WorkerServiceImpl::PredictStream(
//     grpc::ServerContext* context,
//     grpc::ServerReaderWriter<PredictResponse, PredictRequest>* stream) {

//     PredictRequest request;
//     while (stream->Read(&request)) {
//         PredictResponse response;
//         Predict(context, &request, &response);
//         if (!stream->Write(response)) break;
//     }
//     return grpc::Status::OK;
// }

// grpc::Status WorkerServiceImpl::BatchPredict(
//     grpc::ServerContext* context,
//     const BatchPredictRequest* request,
//     BatchPredictResponse* response) {

//     auto start = std::chrono::high_resolution_clock::now();
//     response->set_success(true);

//     for (const auto& single : request->requests()) {
//         PredictResponse single_resp;
//         Predict(context, &single, &single_resp);
//         *response->add_responses() = single_resp;

//         if (!single_resp.success()) {
//             response->set_success(false);
//             response->set_error_message(single_resp.error_message());
//             break;
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     response->set_total_time_ms(
//         std::chrono::duration<double, std::milli>(end - start).count());

//     return grpc::Status::OK;
// }

// // ============================================
// // Model Lifecycle RPCs
// // ============================================

// grpc::Status WorkerServiceImpl::LoadModel(
//     grpc::ServerContext* /*context*/,
//     const LoadModelRequest* request,
//     LoadModelResponse* response) {

//     std::cout << "[WorkerService] LoadModel: " << request->model_id()
//               << " from " << request->model_path() << std::endl;

//     std::map<std::string, std::string> config(
//         request->backend_config().begin(),
//         request->backend_config().end());

//     bool ok = inference_engine_->load_model(
//         request->model_id(),
//         request->model_path(),
//         request->force_backend(),
//         config);

//     response->set_success(ok);

//     if (ok) {
//         auto info = inference_engine_->get_model_info(request->model_id());
//         if (!request->version().empty()) {
//             info.set_version(request->version());
//         }
//         *response->mutable_model_info() = std::move(info);
//     } else {
//         response->set_error_message("Failed to load model: " + request->model_id());
//     }

//     return grpc::Status::OK;
// }

// grpc::Status WorkerServiceImpl::UnloadModel(
//     grpc::ServerContext* /*context*/,
//     const UnloadModelRequest* request,
//     UnloadModelResponse* response) {

//     std::cout << "[WorkerService] UnloadModel: " << request->model_id() << std::endl;

//     bool ok = inference_engine_->unload_model(request->model_id());
//     response->set_success(ok);
//     response->set_message(ok ? "Model unloaded" : "Model not found");

//     return grpc::Status::OK;
// }

// // ============================================
// // Model Introspection RPCs
// // ============================================

// grpc::Status WorkerServiceImpl::ListModels(
//     grpc::ServerContext* /*context*/,
//     const ListModelsRequest* /*request*/,
//     ListModelsResponse* response) {

//     auto ids = inference_engine_->get_loaded_model_ids();
//     for (const auto& id : ids) {
//         auto info = inference_engine_->get_model_info(id);
//         *response->add_models() = std::move(info);
//     }

//     return grpc::Status::OK;
// }

// grpc::Status WorkerServiceImpl::GetModelInfo(
//     grpc::ServerContext* /*context*/,
//     const GetModelInfoRequest* request,
//     GetModelInfoResponse* response) {

//     if (!inference_engine_->is_model_loaded(request->model_id())) {
//         response->set_success(false);
//         response->set_error_message("Model not loaded: " + request->model_id());
//         return grpc::Status::OK;
//     }

//     response->set_success(true);
//     auto info = inference_engine_->get_model_info(request->model_id());
//     *response->mutable_model_info() = std::move(info);

//     auto* m = inference_engine_->get_model_metrics(request->model_id());
//     if (m) {
//         fill_runtime_metrics(*m, response->mutable_metrics());
//     }

//     return grpc::Status::OK;
// }

// grpc::Status WorkerServiceImpl::ValidateModel(
//     grpc::ServerContext* /*context*/,
//     const ValidateModelRequest* request,
//     ValidateModelResponse* response) {

//     auto result = inference_engine_->validate_model(
//         request->model_path(), request->force_backend());

//     response->set_valid(result.valid);
//     response->set_backend(result.backend);
//     response->set_error_message(result.error_message);

//     for (const auto& w : result.warnings) {
//         response->add_warnings(w);
//     }

//     for (const auto& spec : result.schema.inputs) {
//         auto* ts = response->add_inputs();
//         ts->set_name(spec.name);
//         ts->set_dtype(spec.dtype);
//         for (int64_t dim : spec.shape) ts->add_shape(dim);
//     }
//     for (const auto& spec : result.schema.outputs) {
//         auto* ts = response->add_outputs();
//         ts->set_name(spec.name);
//         ts->set_dtype(spec.dtype);
//         for (int64_t dim : spec.shape) ts->add_shape(dim);
//     }

//     return grpc::Status::OK;
// }

// grpc::Status WorkerServiceImpl::WarmupModel(
//     grpc::ServerContext* /*context*/,
//     const WarmupModelRequest* request,
//     WarmupModelResponse* response) {

//     uint32_t runs = request->num_runs() > 0 ? request->num_runs() : 5;
//     auto result = inference_engine_->warmup_model(request->model_id(), runs);

//     response->set_success(result.success);
//     response->set_runs_completed(result.runs_completed);
//     response->set_avg_time_ms(result.avg_time_ms);
//     response->set_min_time_ms(result.min_time_ms);
//     response->set_max_time_ms(result.max_time_ms);
//     response->set_error_message(result.error_message);

//     return grpc::Status::OK;
// }

// // ============================================
// // Worker Observability RPCs
// // ============================================

// grpc::Status WorkerServiceImpl::GetStatus(
//     grpc::ServerContext* /*context*/,
//     const GetStatusRequest* /*request*/,
//     GetStatusResponse* response) {

//     response->set_worker_id(worker_id_);
//     response->set_status(common::WORKER_READY);

//     auto* caps = response->mutable_capabilities();
//     auto engine_info = inference_engine_->get_engine_info();
//     caps->set_supports_gpu(engine_info.gpu_enabled);
//     caps->set_num_cpu_cores(std::thread::hardware_concurrency());
//     for (const auto& b : engine_info.supported_backends) {
//         caps->add_supported_backends(b);
//     }

//     *response->mutable_metrics() = build_worker_metrics();

//     for (const auto& id : inference_engine_->get_loaded_model_ids()) {
//         response->add_loaded_model_ids(id);
//     }

//     return grpc::Status::OK;
// }

// grpc::Status WorkerServiceImpl::GetMetrics(
//     grpc::ServerContext* /*context*/,
//     const GetMetricsRequest* /*request*/,
//     GetMetricsResponse* response) {

//     *response->mutable_worker_metrics() = build_worker_metrics();

//     for (const auto& id : inference_engine_->get_loaded_model_ids()) {
//         auto* m = inference_engine_->get_model_metrics(id);
//         if (m) {
//             auto* dst = &(*response->mutable_per_model_metrics())[id];
//             fill_runtime_metrics(*m, dst);
//             // Timestamps live on ModelBackend; fetch via ModelInfo.
//             auto info = inference_engine_->get_model_info(id);
//             dst->set_loaded_at_unix(info.loaded_at_unix());
//             // last_used_at_unix: not exposed in ModelInfo proto; left at 0.
//         }
//     }

//     return grpc::Status::OK;
// }

// grpc::Status WorkerServiceImpl::HealthCheck(
//     grpc::ServerContext* /*context*/,
//     const HealthCheckRequest* /*request*/,
//     HealthCheckResponse* response) {

//     response->set_healthy(true);
//     response->set_message("Worker " + worker_id_ + " is healthy");
//     return grpc::Status::OK;
// }

// // ============================================
// // File Discovery RPCs
// // ============================================

// grpc::Status WorkerServiceImpl::ListAvailableModels(
//     grpc::ServerContext* /*context*/,
//     const ListAvailableModelsRequest* request,
//     ListAvailableModelsResponse* response) {

//     namespace fs = std::filesystem;

//     std::string dir = request->directory().empty() ? models_dir_ : request->directory();
//     response->set_directory(dir);

//     if (!fs::exists(dir) || !fs::is_directory(dir)) {
//         return grpc::Status::OK;  // empty list — not an error
//     }

//     auto& registry = BackendRegistry::instance();
//     auto extensions = registry.registered_extensions();

//     for (const auto& entry : fs::directory_iterator(dir)) {
//         if (!entry.is_regular_file()) continue;

//         std::string ext = entry.path().extension().string();
//         auto it = std::find(extensions.begin(), extensions.end(), ext);
//         if (it == extensions.end()) continue;

//         auto* m = response->add_models();
//         m->set_filename(entry.path().filename().string());
//         m->set_path(entry.path().string());
//         m->set_extension(ext);
//         m->set_backend(registry.detect_backend(entry.path().string()));
//         m->set_file_size_bytes(static_cast<int64_t>(entry.file_size()));

//         // Check if already loaded
//         for (const auto& id : inference_engine_->get_loaded_model_ids()) {
//             auto info = inference_engine_->get_model_info(id);
//             if (info.model_path() == entry.path().string()) {
//                 m->set_is_loaded(true);
//                 m->set_loaded_as(id);
//                 break;
//             }
//         }
//     }

//     return grpc::Status::OK;
// }

// // ============================================
// // Helpers
// // ============================================

// void WorkerServiceImpl::fill_model_info(
//     const common::ModelInfo& src,
//     common::ModelInfo* dst) const {
//     *dst = src;
// }

// void WorkerServiceImpl::fill_runtime_metrics(
//     const RuntimeMetrics& src,
//     ModelRuntimeMetrics* dst) const {

//     uint64_t ok = src.total_inferences - src.failed_inferences;

//     dst->set_total_inferences(src.total_inferences);
//     dst->set_failed_inferences(src.failed_inferences);
//     dst->set_avg_inference_time_ms(src.avg_time_ms());

//     // Guard: min is only meaningful when at least one successful sample
//     // exists.  Before the first successful call, min_time_ms == 0.0
//     // (the new safe default), so this is safe to send as-is.
//     // The explicit ok > 0 check ensures we never send a stale value if
//     // this worker was built against an older model_backend.hpp.
//     dst->set_min_inference_time_ms((ok > 0) ? src.min_time_ms : 0.0);
//     dst->set_max_inference_time_ms(src.max_time_ms);
//     dst->set_p95_inference_time_ms(src.p95_time_ms());
//     dst->set_p99_inference_time_ms(src.p99_time_ms());
//     dst->set_total_inference_time_ms(src.total_time_ms);
//     // Note: last_used_at_unix and loaded_at_unix are timestamps that live on
//     // ModelBackend, not RuntimeMetrics.  They are populated by the caller
//     // (GetMetrics / GetModelInfo) via get_model_info(), which already writes
//     // them into ModelInfo.  Nothing to set here.
// }

// common::WorkerMetrics WorkerServiceImpl::build_worker_metrics() const {
//     common::WorkerMetrics m;
//     m.set_total_requests(total_requests_.load());
//     m.set_successful_requests(successful_requests_.load());
//     m.set_failed_requests(failed_requests_.load());
//     m.set_active_requests(active_requests_.load());

//     auto now    = std::chrono::steady_clock::now();
//     auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
//     m.set_uptime_seconds(uptime.count());

//     uint64_t ok = successful_requests_.load();
//     if (ok > 0) {
//         double sum = 0.0;
//         uint64_t count = 0;
//         for (const auto& id : inference_engine_->get_loaded_model_ids()) {
//             auto* mm = inference_engine_->get_model_metrics(id);
//             if (mm && mm->total_inferences > mm->failed_inferences) {
//                 sum   += mm->total_time_ms;
//                 count += mm->total_inferences - mm->failed_inferences;
//             }
//         }
//         if (count > 0) {
//             m.set_avg_response_time_ms(sum / static_cast<double>(count));
//         }
//     }

//     return m;
// }

// // ============================================
// // WorkerServer
// // ============================================

// WorkerServer::WorkerServer(const std::string& worker_id,
//                            const std::string& server_address,
//                            bool enable_gpu,
//                            uint32_t num_threads,
//                            const std::string& models_dir)
//     : worker_id_(worker_id)
//     , server_address_(server_address) {

//     service_ = std::make_unique<WorkerServiceImpl>(
//         worker_id, enable_gpu, num_threads, models_dir);
// }

// WorkerServer::~WorkerServer() {
//     stop();
// }

// void WorkerServer::run() {
//     grpc::ServerBuilder builder;
//     builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
//     builder.RegisterService(service_.get());

//     server_ = builder.BuildAndStart();
//     std::cout << "[WorkerServer] Listening on " << server_address_ << std::endl;
//     server_->Wait();
// }

// void WorkerServer::stop() {
//     if (server_) {
//         server_->Shutdown();
//     }
// }

// }  // namespace worker
// }  // namespace mlinference
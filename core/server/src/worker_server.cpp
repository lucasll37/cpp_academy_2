// =============================================================================
// worker_server.cpp — gRPC service implementation
// =============================================================================

#include "server/worker_server.hpp"
#include "inference/backend_registry.hpp"
#include "client/value_convert.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

namespace fs = std::filesystem;

namespace miia {
namespace server {

using inference::InferenceEngine;

// =============================================================================
// WorkerServiceImpl — construtor / destrutor
// =============================================================================

WorkerServiceImpl::WorkerServiceImpl(
    const std::string& worker_id,
    bool               enable_gpu,
    uint32_t           num_threads,
    const std::string& models_dir)
    : worker_id_(worker_id)
    , models_dir_(models_dir)
    , inference_engine_(std::make_shared<InferenceEngine>(enable_gpu, 0, num_threads))
    , start_time_(std::chrono::steady_clock::now()) {

    std::cout << "[WorkerService] Initialized: " << worker_id_
              << " (models: " << models_dir_ << ")" << std::endl;

    LOG_DEBUG("worker") << "WorkerServiceImpl ctor | worker_id=" << worker_id_
                        << " models_dir=" << models_dir_
                        << " gpu=" << enable_gpu
                        << " threads=" << num_threads;
}

WorkerServiceImpl::~WorkerServiceImpl() {
    std::cout << "[WorkerService] Destroyed: " << worker_id_ << std::endl;

    LOG_DEBUG("worker") << "WorkerServiceImpl dtor | worker_id=" << worker_id_;
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

    LOG_DEBUG("worker") << "Predict | model=" << request->model_id()
                        << " active=" << active_requests_.load();

    // Deserialise google.protobuf.Struct → client::Object
    client::Object inputs = client::from_proto_struct(request->inputs());

    auto result = inference_engine_->predict(request->model_id(), inputs);

    response->set_success(result.success);
    response->set_inference_time_ms(result.inference_time_ms);
    response->set_error_message(result.error_message);

    if (result.success) {
        successful_requests_++;
        *response->mutable_outputs() = client::to_proto_struct(result.outputs);
        LOG_DEBUG("worker") << "Predict OK | model=" << request->model_id()
                            << " time_ms=" << result.inference_time_ms;
    } else {
        failed_requests_++;
        LOG_WARN("worker") << "Predict FAILED | model=" << request->model_id()
                           << " error=" << result.error_message;
    }

    active_requests_--;
    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::PredictStream(
    grpc::ServerContext* /*context*/,
    grpc::ServerReaderWriter<PredictResponse, PredictRequest>* stream) {

    LOG_DEBUG("worker") << "PredictStream | started";

    PredictRequest request;
    uint32_t count = 0;
    while (stream->Read(&request)) {
        PredictResponse response;
        Predict(nullptr, &request, &response);
        stream->Write(response);
        ++count;
    }

    LOG_DEBUG("worker") << "PredictStream | finished | items=" << count;

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::BatchPredict(
    grpc::ServerContext* /*context*/,
    const BatchPredictRequest* request,
    BatchPredictResponse* response) {

    LOG_DEBUG("worker") << "BatchPredict | count=" << request->requests_size();

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& single_req : request->requests()) {
        auto* single_resp = response->add_responses();

        // Chama o engine diretamente em vez de this->Predict() para evitar
        // double-counting nos contadores atômicos e possível reentrância
        // no mutex do InferenceEngine após o fix de serialização do predict().
        active_requests_++;
        total_requests_++;

        client::Object inputs = client::from_proto_struct(single_req.inputs());
        auto result = inference_engine_->predict(single_req.model_id(), inputs);

        single_resp->set_success(result.success);
        single_resp->set_inference_time_ms(result.inference_time_ms);
        single_resp->set_error_message(result.error_message);

        if (result.success) {
            successful_requests_++;
            *single_resp->mutable_outputs() = client::to_proto_struct(result.outputs);
        } else {
            failed_requests_++;
            LOG_WARN("worker") << "BatchPredict item FAILED | model=" << single_req.model_id()
                               << " error=" << result.error_message;
        }

        active_requests_--;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    bool all_ok = true;
    for (const auto& r : response->responses())
        if (!r.success()) { all_ok = false; break; }

    response->set_success(all_ok);
    response->set_total_time_ms(total_ms);
    if (!all_ok) {
        response->set_error_message("One or more predictions failed");
        LOG_WARN("worker") << "BatchPredict partial failure | total_ms=" << total_ms
                           << " count=" << request->requests_size();
    } else {
        LOG_DEBUG("worker") << "BatchPredict OK | total_ms=" << total_ms
                            << " count=" << request->requests_size();
    }

    return grpc::Status::OK;
}

// =============================================================================
// Model Lifecycle RPCs
// =============================================================================

grpc::Status WorkerServiceImpl::LoadModel(
    grpc::ServerContext* /*context*/,
    const LoadModelRequest* request,
    LoadModelResponse* response) {

    std::string model_path = request->model_path();

    LOG_DEBUG("worker") << "LoadModel | model=" << request->model_id()
                        << " path=" << model_path
                        << " backend=" << request->force_backend();

    // protobuf::Map iterators não são compatíveis com std::map range-constructor
    // no GCC 13 — construir manualmente.
    std::map<std::string, std::string> config;
    for (auto it = request->backend_config().begin();
         it != request->backend_config().end(); ++it) {
        config[it->first] = it->second;
    }

    bool ok = inference_engine_->load_model(
        request->model_id(), model_path, request->force_backend(), config);

    response->set_success(ok);
    if (!ok) {
        response->set_error_message("Failed to load model: " + model_path);
        LOG_ERROR("worker") << "LoadModel FAILED | model=" << request->model_id()
                            << " path=" << model_path;
        return grpc::Status::OK;
    }

    LOG_INFO("worker") << "LoadModel OK | model=" << request->model_id()
                       << " path=" << model_path
                       << " backend=" << request->force_backend();

    *response->mutable_model_info() = inference_engine_->get_model_info(request->model_id());
    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::UnloadModel(
    grpc::ServerContext* /*context*/,
    const UnloadModelRequest* request,
    UnloadModelResponse* response) {

    LOG_DEBUG("worker") << "UnloadModel | model=" << request->model_id();

    bool ok = inference_engine_->unload_model(request->model_id());
    response->set_success(ok);
    response->set_message(ok ? "Model unloaded" : "Model not found: " + request->model_id());

    if (ok)
        LOG_INFO("worker") << "UnloadModel OK | model=" << request->model_id();
    else
        LOG_WARN("worker") << "UnloadModel | model not found: " << request->model_id();

    return grpc::Status::OK;
}

// =============================================================================
// Model Introspection RPCs
// =============================================================================

grpc::Status WorkerServiceImpl::ListModels(
    grpc::ServerContext* /*context*/,
    const ListModelsRequest* /*request*/,
    ListModelsResponse* response) {

    for (const auto& id : inference_engine_->get_loaded_model_ids()) {
        auto info = inference_engine_->get_model_info(id);
        *response->add_models() = info;
    }

    LOG_DEBUG("worker") << "ListModels | loaded=" << response->models_size();

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::GetModelInfo(
    grpc::ServerContext* /*context*/,
    const GetModelInfoRequest* request,
    GetModelInfoResponse* response) {

    LOG_DEBUG("worker") << "GetModelInfo | model=" << request->model_id();

    if (!inference_engine_->is_model_loaded(request->model_id())) {
        response->set_success(false);
        response->set_error_message("Model not loaded: " + request->model_id());
        LOG_WARN("worker") << "GetModelInfo | model not loaded: " << request->model_id();
        return grpc::Status::OK;
    }

    *response->mutable_model_info() = inference_engine_->get_model_info(request->model_id());
    response->set_success(true);
    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::ValidateModel(
    grpc::ServerContext* /*context*/,
    const ValidateModelRequest* request,
    ValidateModelResponse* response) {

    LOG_DEBUG("worker") << "ValidateModel | path=" << request->model_path();

    auto result = inference_engine_->validate_model(request->model_path());
    response->set_valid(result.valid);
    response->set_error_message(result.error_message);
    response->set_backend(result.backend);

    for (const auto& spec : result.inputs) {
        auto* ts = response->add_inputs();
        ts->set_name(spec.name);
        ts->set_dtype(spec.dtype);
        ts->set_description(spec.description);
        for (int64_t d : spec.shape) ts->add_shape(d);
    }
    for (const auto& spec : result.outputs) {
        auto* ts = response->add_outputs();
        ts->set_name(spec.name);
        ts->set_dtype(spec.dtype);
        ts->set_description(spec.description);
        for (int64_t d : spec.shape) ts->add_shape(d);
    }
    for (const auto& w : result.warnings) {
        response->add_warnings(w);
        LOG_WARN("worker") << "ValidateModel warning | path=" << request->model_path()
                           << " msg=" << w;
    }

    if (result.valid)
        LOG_INFO("worker") << "ValidateModel OK | path=" << request->model_path()
                           << " backend=" << result.backend
                           << " inputs=" << result.inputs.size()
                           << " outputs=" << result.outputs.size();
    else
        LOG_WARN("worker") << "ValidateModel INVALID | path=" << request->model_path()
                           << " error=" << result.error_message;

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::WarmupModel(
    grpc::ServerContext* /*context*/,
    const WarmupModelRequest* request,
    WarmupModelResponse* response) {

    uint32_t runs = request->num_runs() > 0 ? request->num_runs() : 5;

    LOG_DEBUG("worker") << "WarmupModel | model=" << request->model_id()
                        << " runs=" << runs;

    auto result = inference_engine_->warmup_model(request->model_id(), runs);

    response->set_success(result.success);
    response->set_runs_completed(result.runs_completed);
    response->set_avg_time_ms(result.avg_time_ms);
    response->set_min_time_ms(result.min_time_ms);
    response->set_max_time_ms(result.max_time_ms);
    response->set_error_message(result.error_message);

    if (result.success)
        LOG_INFO("worker") << "WarmupModel OK | model=" << request->model_id()
                           << " runs=" << result.runs_completed
                           << " avg_ms=" << result.avg_time_ms
                           << " min_ms=" << result.min_time_ms
                           << " max_ms=" << result.max_time_ms;
    else
        LOG_WARN("worker") << "WarmupModel FAILED | model=" << request->model_id()
                           << " error=" << result.error_message;

    return grpc::Status::OK;
}

// =============================================================================
// Observability RPCs
// =============================================================================

grpc::Status WorkerServiceImpl::GetStatus(
    grpc::ServerContext* /*context*/,
    const GetStatusRequest* /*request*/,
    GetStatusResponse* response) {

    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_).count();

    response->set_worker_id(worker_id_);

    auto* wm = response->mutable_metrics();
    wm->set_total_requests(total_requests_.load());
    wm->set_successful_requests(successful_requests_.load());
    wm->set_failed_requests(failed_requests_.load());
    wm->set_active_requests(active_requests_.load());
    wm->set_uptime_seconds(static_cast<int64_t>(uptime));

    for (const auto& id : inference_engine_->get_loaded_model_ids())
        response->add_loaded_model_ids(id);

    // Preenche capabilities com os backends suportados pelo engine.
    auto* cap = response->mutable_capabilities();
    for (const auto& b : inference_engine_->get_engine_info().supported_backends)
        cap->add_supported_backends(b);

    LOG_DEBUG("worker") << "GetStatus | uptime_s=" << uptime
                        << " total=" << total_requests_.load()
                        << " ok=" << successful_requests_.load()
                        << " fail=" << failed_requests_.load()
                        << " active=" << active_requests_.load();

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::GetMetrics(
    grpc::ServerContext* /*context*/,
    const GetMetricsRequest* /*request*/,
    GetMetricsResponse* response) {

    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time_).count();

    auto* wm = response->mutable_worker_metrics();
    wm->set_total_requests(total_requests_.load());
    wm->set_successful_requests(successful_requests_.load());
    wm->set_failed_requests(failed_requests_.load());
    wm->set_active_requests(active_requests_.load());
    wm->set_uptime_seconds(static_cast<uint64_t>(uptime));

    for (const auto& id : inference_engine_->get_loaded_model_ids()) {
        const auto* m = inference_engine_->get_model_metrics(id);
        if (!m) continue;

        auto& pm = (*response->mutable_per_model_metrics())[id];
        pm.set_total_inferences(m->total_inferences);
        pm.set_failed_inferences(m->failed_inferences);
        pm.set_avg_inference_time_ms(m->avg_time_ms());
        pm.set_min_inference_time_ms(m->min_time_ms);
        pm.set_max_inference_time_ms(m->max_time_ms);
        pm.set_p95_inference_time_ms(m->p95_time_ms());
        pm.set_p99_inference_time_ms(m->p99_time_ms());
        pm.set_total_inference_time_ms(m->total_time_ms);
    }

    LOG_DEBUG("worker") << "GetMetrics | uptime_s=" << uptime
                        << " models=" << response->per_model_metrics_size();

    return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::HealthCheck(
    grpc::ServerContext* /*context*/,
    const HealthCheckRequest* /*request*/,
    HealthCheckResponse* response) {

    response->set_healthy(true);
    response->set_message("OK");

    LOG_DEBUG("worker") << "HealthCheck | OK";

    return grpc::Status::OK;
}

// =============================================================================
// File Discovery RPC
// =============================================================================

grpc::Status WorkerServiceImpl::ListAvailableModels(
    grpc::ServerContext* /*context*/,
    const ListAvailableModelsRequest* request,
    ListAvailableModelsResponse* response) {

    std::string dir = request->directory().empty() ? models_dir_ : request->directory();

    LOG_DEBUG("worker") << "ListAvailableModels | dir=" << dir;

    std::error_code ec;
    if (!fs::exists(dir, ec)) {
        LOG_ERROR("worker") << "ListAvailableModels | directory not found: " << dir;
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "Directory not found: " + dir);
    }

    auto loaded_ids = inference_engine_->get_loaded_model_ids();

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;

        auto* m = response->add_models();
        m->set_filename(entry.path().filename().string());
        m->set_path(entry.path().string());
        m->set_extension(entry.path().extension().string());
        m->set_file_size_bytes(static_cast<int64_t>(entry.file_size()));

        const auto& ext = entry.path().extension().string();
        if (ext == ".onnx")
            m->set_backend(common::BACKEND_ONNX);
        else if (ext == ".py")
            m->set_backend(common::BACKEND_PYTHON);
        else
            m->set_backend(common::BACKEND_UNKNOWN);

        // Check if loaded
        const std::string entry_path = entry.path().string();
        bool is_loaded = false;
        std::string loaded_as;
        for (const auto& id : loaded_ids) {
            const std::string model_path = inference_engine_->get_model_info(id).model_path();
            if (model_path == entry_path) {
                is_loaded = true;
                loaded_as = id;
                break;
            }
        }
        m->set_is_loaded(is_loaded);
        if (is_loaded) m->set_loaded_as(loaded_as);
    }

    LOG_DEBUG("worker") << "ListAvailableModels | dir=" << dir
                        << " count=" << response->models_size();

    return grpc::Status::OK;
}

// =============================================================================
// WorkerServer — dono do service + servidor gRPC
// =============================================================================

WorkerServer::WorkerServer(
    const std::string& worker_id,
    const std::string& server_address,
    bool               enable_gpu,
    uint32_t           num_threads,
    const std::string& models_dir)
    : worker_id_(worker_id)
    , server_address_(server_address)
    , service_(std::make_unique<WorkerServiceImpl>(
        worker_id, enable_gpu, num_threads, models_dir)) {

    LOG_DEBUG("worker") << "WorkerServer ctor | worker_id=" << worker_id
                        << " address=" << server_address;
}

WorkerServer::~WorkerServer() {
    LOG_DEBUG("worker") << "WorkerServer dtor | worker_id=" << worker_id_;
    stop();
}

void WorkerServer::run() {
    // Verifica se a porta já está em uso antes de tentar o bind gRPC,
    // evitando o erro verboso interno do gRPC como única indicação de falha.
    {
        int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
        if (fd < 0) fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            int opt = 1;
            ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            struct sockaddr_in6 addr{};
            addr.sin6_family = AF_INET6;
            addr.sin6_addr   = in6addr_any;
            // extrai porta do server_address_ (formato "host:port")
            auto colon = server_address_.rfind(':');
            int port = (colon != std::string::npos)
                ? std::stoi(server_address_.substr(colon + 1)) : 50052;
            addr.sin6_port = htons(static_cast<uint16_t>(port));
            if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0
                && errno == EADDRINUSE) {
                ::close(fd);
                std::cerr << "[WorkerServer] Port already in use: "
                          << server_address_ << std::endl;
                LOG_ERROR("worker") << "Port already in use: " << server_address_
                                    << " — encerre o processo que ocupa a porta e tente novamente";
                return;
            }
            ::close(fd);
        }
    }

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());

    LOG_DEBUG("worker") << "WorkerServer building | address=" << server_address_;

    server_ = builder.BuildAndStart();

    if (!server_) {
        std::cerr << "[WorkerServer] Failed to start on " << server_address_ << std::endl;
        LOG_ERROR("worker") << "Failed to bind on " << server_address_;
        return;
    }

    std::cout << "[WorkerServer] Listening on " << server_address_ << std::endl;
    LOG_INFO("worker") << "Listening on " << server_address_;

    server_->Wait();  // Bloqueia até Shutdown() ser chamado

    LOG_INFO("worker") << "Server stopped | address=" << server_address_;
}

void WorkerServer::stop() {
    if (server_) {
        LOG_INFO("worker") << "Shutting down | address=" << server_address_;
        server_->Shutdown();
        server_.reset();
    }
}

}  // namespace server
}  // namespace miia
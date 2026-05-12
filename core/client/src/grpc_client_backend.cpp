// =============================================================================
// grpc_client_backend.cpp — Backend gRPC do cliente Miia
//
// Implementa IClientBackend via chamadas gRPC ao WorkerService.
// Converte client::Object ↔ google.protobuf.Struct na fronteira.
// =============================================================================

#include "client/grpc_client_backend.hpp"
#include "client/value_convert.hpp"
#include "utils/logger.hpp"

#include <grpcpp/grpcpp.h>
#include "server.grpc.pb.h"
#include "common.pb.h"

#include <chrono>

namespace mlinference {
namespace client {

// =============================================================================
// Helper privado — estado do canal
// =============================================================================

bool GrpcClientBackend::is_channel_ready() const {
    if (!channel_ || !ever_connected_) return false;
    auto state = channel_->GetState(/*try_to_connect=*/false);
    bool ready = (state == GRPC_CHANNEL_READY || state == GRPC_CHANNEL_IDLE);
    LOG_DEBUG("grpc_backend") << "[is_channel_ready] state=" << static_cast<int>(state)
                              << " ready=" << ready;
    return ready;
}

// =============================================================================
// Conexão
// =============================================================================

bool GrpcClientBackend::connect() {
    LOG_DEBUG("grpc_backend") << "[connect] chamado; server_address_='" << server_address_ << "'";

    channel_ = grpc::CreateChannel(server_address_, grpc::InsecureChannelCredentials());
    LOG_DEBUG("grpc_backend") << "[connect] channel_ criado para '" << server_address_ << "'";

    stub_ = server::WorkerService::NewStub(channel_);
    LOG_DEBUG("grpc_backend") << "[connect] stub_ criado; stub_=" << (void*)stub_.get();

    grpc::ClientContext ctx;
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
    ctx.set_deadline(deadline);
    LOG_DEBUG("grpc_backend") << "[connect] deadline de 5s configurado; chamando HealthCheck()";

    server::HealthCheckRequest  req;
    server::HealthCheckResponse resp;
    auto st = stub_->HealthCheck(&ctx, req, &resp);

    LOG_DEBUG("grpc_backend") << "[connect] HealthCheck retornou ok=" << st.ok()
                              << " healthy=" << resp.healthy()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";

    bool ok = st.ok() && resp.healthy();
    if (ok) {
        LOG_INFO("grpc_backend") << "[connect] conectado com sucesso a '" << server_address_ << "'"
                                 << " channel_state=" << static_cast<int>(channel_->GetState(false));
    } else {
        LOG_ERROR("grpc_backend") << "[connect] falha ao conectar; ok=" << st.ok()
                                  << " healthy=" << resp.healthy()
                                  << " error_message='" << st.error_message() << "'";
    }

    ever_connected_ = ok;
    return ok;
}

// =============================================================================
// Ciclo de vida dos modelos
// =============================================================================

bool GrpcClientBackend::load_model(const std::string& model_id,
                                    const std::string& model_path,
                                    const std::string& /*version*/) {
    LOG_DEBUG("grpc_backend") << "[load_model] chamado; model_id='" << model_id
                              << "' model_path='" << model_path
                              << "' channel_ready=" << is_channel_ready();
    if (!is_channel_ready()) {
        LOG_ERROR("grpc_backend") << "[load_model] FALHA PRÉ-CONDIÇÃO: canal não disponível"
                                  << " state=" << static_cast<int>(channel_ ? channel_->GetState(false) : GRPC_CHANNEL_SHUTDOWN);
        return false;
    }

    grpc::ClientContext       ctx;
    server::LoadModelRequest  req;
    server::LoadModelResponse resp;

    req.set_model_id(model_id);
    req.set_model_path(model_path);

    LOG_DEBUG("grpc_backend") << "[load_model] chamando stub_->LoadModel()";
    auto st = stub_->LoadModel(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[load_model] LoadModel retornou ok=" << st.ok()
                              << " success=" << resp.success()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";

    if (!st.ok()) {
        LOG_ERROR("grpc_backend") << "[load_model] FALHA RPC: " << st.error_message();
        return false;
    }
    if (!resp.success()) {
        LOG_WARN("grpc_backend") << "[load_model] servidor recusou o modelo; error='" << resp.error_message() << "'";
        return false;
    }

    LOG_INFO("grpc_backend") << "[load_model] modelo carregado com sucesso; model_id='" << model_id
                             << "' path='" << model_path << "'";
    return true;
}

bool GrpcClientBackend::unload_model(const std::string& model_id) {
    LOG_DEBUG("grpc_backend") << "[unload_model] chamado; model_id='" << model_id
                              << "' channel_ready=" << is_channel_ready();
    if (!is_channel_ready()) {
        LOG_ERROR("grpc_backend") << "[unload_model] FALHA PRÉ-CONDIÇÃO: canal não disponível";
        return false;
    }

    grpc::ClientContext         ctx;
    server::UnloadModelRequest  req;
    server::UnloadModelResponse resp;

    req.set_model_id(model_id);

    LOG_DEBUG("grpc_backend") << "[unload_model] chamando stub_->UnloadModel()";
    auto st = stub_->UnloadModel(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[unload_model] UnloadModel retornou ok=" << st.ok()
                              << " success=" << resp.success()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";

    if (!st.ok()) {
        LOG_ERROR("grpc_backend") << "[unload_model] FALHA RPC: " << st.error_message();
        return false;
    }
    if (!resp.success()) {
        LOG_WARN("grpc_backend") << "[unload_model] servidor não encontrou o modelo; msg='" << resp.message() << "'";
        return false;
    }

    LOG_INFO("grpc_backend") << "[unload_model] modelo descarregado; model_id='" << model_id << "'";
    return true;
}

// =============================================================================
// Inferência
// =============================================================================

PredictionResult GrpcClientBackend::predict(const std::string& model_id,
                                             const Object& inputs) {
    LOG_DEBUG("grpc_backend") << "[predict] chamado; model_id='" << model_id
                              << "' channel_ready=" << is_channel_ready();
    PredictionResult result;
    if (!is_channel_ready()) {
        LOG_ERROR("grpc_backend") << "[predict] FALHA PRÉ-CONDIÇÃO: canal não disponível"
                                  << " state=" << static_cast<int>(channel_ ? channel_->GetState(false) : GRPC_CHANNEL_SHUTDOWN);
        result.success       = false;
        result.error_message = "gRPC channel not available";
        return result;
    }

    grpc::ClientContext      ctx;
    server::PredictRequest   req;
    server::PredictResponse  resp;

    req.set_model_id(model_id);
    *req.mutable_inputs() = to_proto_struct(inputs);

    LOG_DEBUG("grpc_backend") << "[predict] chamando stub_->Predict()";
    auto st = stub_->Predict(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[predict] Predict retornou ok=" << st.ok()
                              << " success=" << resp.success()
                              << " inference_time_ms=" << resp.inference_time_ms()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";

    if (!st.ok()) {
        LOG_ERROR("grpc_backend") << "[predict] FALHA RPC: " << st.error_message();
        result.success       = false;
        result.error_message = st.error_message();
        return result;
    }

    result.success           = resp.success();
    result.inference_time_ms = resp.inference_time_ms();
    result.error_message     = resp.error_message();

    if (result.success) {
        result.outputs = from_proto_struct(resp.outputs());
        LOG_DEBUG("grpc_backend") << "[predict] OK; model_id='" << model_id
                                  << "' time_ms=" << result.inference_time_ms;
    } else {
        LOG_WARN("grpc_backend") << "[predict] servidor reportou falha; model_id='" << model_id
                                 << "' error='" << result.error_message << "'";
    }
    return result;
}

std::vector<PredictionResult> GrpcClientBackend::batch_predict(
    const std::string& model_id,
    const std::vector<Object>& batch_inputs) {

    LOG_DEBUG("grpc_backend") << "[batch_predict] chamado; model_id='" << model_id
                              << "' batch_size=" << batch_inputs.size()
                              << "' channel_ready=" << is_channel_ready();

    std::vector<PredictionResult> results;
    if (!is_channel_ready()) {
        LOG_ERROR("grpc_backend") << "[batch_predict] FALHA PRÉ-CONDIÇÃO: canal não disponível";
        results.resize(batch_inputs.size());
        for (auto& r : results) { r.success = false; r.error_message = "gRPC channel not available"; }
        return results;
    }

    grpc::ClientContext           ctx;
    server::BatchPredictRequest   req;
    server::BatchPredictResponse  resp;

    for (const auto& inputs : batch_inputs) {
        auto* single = req.add_requests();
        single->set_model_id(model_id);
        *single->mutable_inputs() = to_proto_struct(inputs);
    }

    LOG_DEBUG("grpc_backend") << "[batch_predict] chamando stub_->BatchPredict()";
    auto st = stub_->BatchPredict(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[batch_predict] BatchPredict retornou ok=" << st.ok()
                              << " success=" << resp.success()
                              << " total_time_ms=" << resp.total_time_ms()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";

    if (!st.ok()) {
        LOG_ERROR("grpc_backend") << "[batch_predict] FALHA RPC: " << st.error_message();
        results.resize(batch_inputs.size());
        for (auto& r : results) { r.success = false; r.error_message = st.error_message(); }
        return results;
    }

    results.reserve(resp.responses_size());
    for (const auto& r : resp.responses()) {
        PredictionResult pr;
        pr.success           = r.success();
        pr.inference_time_ms = r.inference_time_ms();
        pr.error_message     = r.error_message();
        if (pr.success)
            pr.outputs = from_proto_struct(r.outputs());
        else
            LOG_WARN("grpc_backend") << "[batch_predict] item falhou; error='" << pr.error_message << "'";
        results.push_back(std::move(pr));
    }

    LOG_DEBUG("grpc_backend") << "[batch_predict] concluído; n_results=" << results.size()
                              << " total_time_ms=" << resp.total_time_ms();
    return results;
}

WarmupResult GrpcClientBackend::warmup_model(const std::string& model_id,
                                              uint32_t           num_runs) {
    LOG_DEBUG("grpc_backend") << "[warmup_model] chamado; model_id='" << model_id
                              << "' num_runs=" << num_runs
                              << " channel_ready=" << is_channel_ready();
    WarmupResult result;
    if (!is_channel_ready()) {
        LOG_ERROR("grpc_backend") << "[warmup_model] FALHA PRÉ-CONDIÇÃO: canal não disponível";
        result.success       = false;
        result.error_message = "gRPC channel not available";
        return result;
    }

    grpc::ClientContext          ctx;
    server::WarmupModelRequest   req;
    server::WarmupModelResponse  resp;

    req.set_model_id(model_id);
    req.set_num_runs(num_runs);

    LOG_DEBUG("grpc_backend") << "[warmup_model] chamando stub_->WarmupModel()";
    auto st = stub_->WarmupModel(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[warmup_model] WarmupModel retornou ok=" << st.ok()
                              << " success=" << resp.success()
                              << " runs_completed=" << resp.runs_completed()
                              << " avg_ms=" << resp.avg_time_ms()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";

    if (!st.ok()) {
        LOG_ERROR("grpc_backend") << "[warmup_model] FALHA RPC: " << st.error_message();
        result.success       = false;
        result.error_message = st.error_message();
        return result;
    }

    result.success        = resp.success();
    result.runs_completed = resp.runs_completed();
    result.avg_time_ms    = resp.avg_time_ms();
    result.min_time_ms    = resp.min_time_ms();
    result.max_time_ms    = resp.max_time_ms();
    result.error_message  = resp.error_message();

    if (result.success)
        LOG_INFO("grpc_backend") << "[warmup_model] OK; model_id='" << model_id
                                 << "' runs=" << result.runs_completed
                                 << " avg_ms=" << result.avg_time_ms
                                 << " min_ms=" << result.min_time_ms
                                 << " max_ms=" << result.max_time_ms;
    else
        LOG_WARN("grpc_backend") << "[warmup_model] warmup sem sucesso; error_message='" << result.error_message << "'";

    return result;
}

// =============================================================================
// Observabilidade
// =============================================================================

bool GrpcClientBackend::health_check() {
    LOG_DEBUG("grpc_backend") << "[health_check] chamado; channel_ready=" << is_channel_ready();
    if (!is_channel_ready()) {
        LOG_WARN("grpc_backend") << "[health_check] canal não disponível"
                                 << " state=" << static_cast<int>(channel_ ? channel_->GetState(false) : GRPC_CHANNEL_SHUTDOWN);
        return false;
    }

    grpc::ClientContext         ctx;
    server::HealthCheckRequest  req;
    server::HealthCheckResponse resp;

    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(3);
    ctx.set_deadline(deadline);
    LOG_DEBUG("grpc_backend") << "[health_check] deadline de 3s configurado; chamando stub_->HealthCheck()";

    auto st = stub_->HealthCheck(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[health_check] HealthCheck retornou ok=" << st.ok()
                              << " healthy=" << resp.healthy()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";

    bool result = st.ok() && resp.healthy();
    if (!result)
        LOG_WARN("grpc_backend") << "[health_check] serviço não saudável; ok=" << st.ok()
                                 << " healthy=" << resp.healthy();
    return result;
}

WorkerStatus GrpcClientBackend::get_status() {
    LOG_DEBUG("grpc_backend") << "[get_status] chamado; channel_ready=" << is_channel_ready();
    WorkerStatus result;
    if (!is_channel_ready()) {
        LOG_WARN("grpc_backend") << "[get_status] canal não disponível, retornando WorkerStatus vazio";
        return result;
    }

    grpc::ClientContext        ctx;
    server::GetStatusRequest   req;
    server::GetStatusResponse  resp;

    LOG_DEBUG("grpc_backend") << "[get_status] chamando stub_->GetStatus()";
    auto st = stub_->GetStatus(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[get_status] GetStatus retornou ok=" << st.ok()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";
    if (!st.ok()) {
        LOG_WARN("grpc_backend") << "[get_status] FALHA RPC: " << st.error_message();
        return result;
    }

    result.worker_id = resp.worker_id();
    const auto& metrics = resp.metrics();
    result.total_requests      = metrics.total_requests();
    result.successful_requests = metrics.successful_requests();
    result.failed_requests     = metrics.failed_requests();
    result.active_requests     = metrics.active_requests();
    result.uptime_seconds      = metrics.uptime_seconds();

    LOG_DEBUG("grpc_backend") << "[get_status] worker_id='" << result.worker_id
                              << "' metrics: total=" << metrics.total_requests()
                              << " successful=" << metrics.successful_requests()
                              << " failed=" << metrics.failed_requests()
                              << " active=" << metrics.active_requests()
                              << " uptime_seconds=" << metrics.uptime_seconds();

    for (const auto& id : resp.loaded_model_ids()) {
        LOG_DEBUG("grpc_backend") << "[get_status] loaded model='" << id << "'";
        result.loaded_models.push_back(id);
    }
    for (const auto& b : resp.capabilities().supported_backends()) {
        LOG_DEBUG("grpc_backend") << "[get_status] supported backend='" << b << "'";
        result.supported_backends.push_back(b);
    }

    return result;
}

ServerMetrics GrpcClientBackend::get_metrics() {
    LOG_DEBUG("grpc_backend") << "[get_metrics] chamado; channel_ready=" << is_channel_ready();
    ServerMetrics result;
    if (!is_channel_ready()) {
        LOG_WARN("grpc_backend") << "[get_metrics] canal não disponível, retornando ServerMetrics vazio";
        return result;
    }

    grpc::ClientContext        ctx;
    server::GetMetricsRequest  req;
    server::GetMetricsResponse resp;

    LOG_DEBUG("grpc_backend") << "[get_metrics] chamando stub_->GetMetrics()";
    auto st = stub_->GetMetrics(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[get_metrics] GetMetrics retornou ok=" << st.ok()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";
    if (!st.ok()) {
        LOG_WARN("grpc_backend") << "[get_metrics] FALHA RPC: " << st.error_message();
        return result;
    }

    const auto& wm = resp.worker_metrics();
    result.total_requests      = wm.total_requests();
    result.successful_requests = wm.successful_requests();
    result.failed_requests     = wm.failed_requests();
    result.active_requests     = wm.active_requests();
    result.uptime_seconds      = wm.uptime_seconds();

    LOG_DEBUG("grpc_backend") << "[get_metrics] worker_metrics: total=" << wm.total_requests()
                              << " successful=" << wm.successful_requests()
                              << " failed=" << wm.failed_requests()
                              << " active=" << wm.active_requests()
                              << " uptime_seconds=" << wm.uptime_seconds();

    // após os campos escalares do worker_metrics, antes do return
    for (const auto& [id, pm] : resp.per_model_metrics()) {
        ModelMetrics mm;
        mm.total_inferences    = pm.total_inferences();
        mm.failed_inferences   = pm.failed_inferences();
        mm.avg_ms              = pm.avg_inference_time_ms();
        mm.min_ms              = pm.min_inference_time_ms();
        mm.max_ms              = pm.max_inference_time_ms();
        mm.p95_ms              = pm.p95_inference_time_ms();
        mm.p99_ms              = pm.p99_inference_time_ms();
        mm.total_time_ms       = pm.total_inference_time_ms();
        result.per_model[id]   = std::move(mm);
        LOG_DEBUG("grpc_backend") << "[get_metrics] per_model['" << id << "']"
                                << " total=" << mm.total_inferences
                                << " avg_ms=" << mm.avg_ms;
    }

    return result;
}

// =============================================================================
// Introspecção
// =============================================================================

std::vector<ModelInfo> GrpcClientBackend::list_models() {
    LOG_DEBUG("grpc_backend") << "[list_models] chamado; channel_ready=" << is_channel_ready();
    std::vector<ModelInfo> result;
    if (!is_channel_ready()) {
        LOG_WARN("grpc_backend") << "[list_models] canal não disponível, retornando lista vazia";
        return result;
    }

    grpc::ClientContext        ctx;
    server::ListModelsRequest  req;
    server::ListModelsResponse resp;

    LOG_DEBUG("grpc_backend") << "[list_models] chamando stub_->ListModels()";
    auto st = stub_->ListModels(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[list_models] ListModels retornou ok=" << st.ok()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";
    if (!st.ok()) {
        LOG_WARN("grpc_backend") << "[list_models] FALHA RPC: " << st.error_message();
        return result;
    }

    LOG_DEBUG("grpc_backend") << "[list_models] n_modelos=" << resp.models_size();
    for (const auto& p : resp.models()) {
        LOG_DEBUG("grpc_backend") << "[list_models] convertendo model_id='" << p.model_id() << "'";
        result.push_back(proto_to_model_info(p));
    }

    LOG_DEBUG("grpc_backend") << "[list_models] retornando " << result.size() << " modelos";
    return result;
}

ModelInfo GrpcClientBackend::get_model_info(const std::string& model_id) {
    LOG_DEBUG("grpc_backend") << "[get_model_info] chamado; model_id='" << model_id
                              << "' channel_ready=" << is_channel_ready();
    if (!is_channel_ready()) {
        LOG_WARN("grpc_backend") << "[get_model_info] canal não disponível, retornando ModelInfo vazio";
        return {};
    }

    grpc::ClientContext          ctx;
    server::GetModelInfoRequest  req;
    server::GetModelInfoResponse resp;

    req.set_model_id(model_id);
    LOG_DEBUG("grpc_backend") << "[get_model_info] chamando stub_->GetModelInfo(model_id='" << model_id << "')";
    auto st = stub_->GetModelInfo(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[get_model_info] GetModelInfo retornou ok=" << st.ok()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";
    if (!st.ok()) {
        LOG_WARN("grpc_backend") << "[get_model_info] FALHA RPC: " << st.error_message();
        return {};
    }

    LOG_DEBUG("grpc_backend") << "[get_model_info] convertendo proto para ModelInfo";
    return proto_to_model_info(resp.model_info());
}

ValidationResult GrpcClientBackend::validate_model(const std::string& path) {
    LOG_DEBUG("grpc_backend") << "[validate_model] chamado; path='" << path
                              << "' channel_ready=" << is_channel_ready();
    ValidationResult result;
    if (!is_channel_ready()) {
        LOG_ERROR("grpc_backend") << "[validate_model] FALHA PRÉ-CONDIÇÃO: canal não disponível";
        return result;
    }

    grpc::ClientContext            ctx;
    server::ValidateModelRequest   req;
    server::ValidateModelResponse  resp;

    req.set_model_path(path);
    LOG_DEBUG("grpc_backend") << "[validate_model] chamando stub_->ValidateModel(path='" << path << "')";
    auto st = stub_->ValidateModel(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[validate_model] ValidateModel retornou ok=" << st.ok()
                              << " valid=" << resp.valid()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";
    if (!st.ok()) {
        LOG_ERROR("grpc_backend") << "[validate_model] FALHA RPC: " << st.error_message();
        return result;
    }

    result.valid         = resp.valid();
    result.error_message = resp.error_message();

    switch (resp.backend()) {
        case common::BACKEND_ONNX:   result.backend = "onnx";    break;
        case common::BACKEND_PYTHON: result.backend = "python";  break;
        default:                      result.backend = "unknown"; break;
    }

    LOG_DEBUG("grpc_backend") << "[validate_model] valid=" << resp.valid()
                              << " backend='" << result.backend << "'"
                              << " n_inputs=" << resp.inputs_size()
                              << " n_outputs=" << resp.outputs_size()
                              << " n_warnings=" << resp.warnings_size();

    for (const auto& ts : resp.inputs()) {
        ModelInfo::TensorSpec s;
        s.name        = ts.name();
        s.dtype       = dtype_str(ts.dtype());
        s.description = ts.description();
        s.structured  = ts.structured();
        for (int64_t d : ts.shape()) s.shape.push_back(d);
        result.inputs.push_back(std::move(s));
    }
    for (const auto& ts : resp.outputs()) {
        ModelInfo::TensorSpec s;
        s.name        = ts.name();
        s.dtype       = dtype_str(ts.dtype());
        s.description = ts.description();
        s.structured  = ts.structured();
        for (int64_t d : ts.shape()) s.shape.push_back(d);
        result.outputs.push_back(std::move(s));
    }
    for (const auto& w : resp.warnings()) {
        LOG_WARN("grpc_backend") << "[validate_model] warning='" << w << "'";
        result.warnings.push_back(w);
    }

    if (!result.valid)
        LOG_WARN("grpc_backend") << "[validate_model] modelo inválido; error='" << result.error_message << "'";

    return result;
}

std::vector<AvailableModel> GrpcClientBackend::list_available_models(
    const std::string& directory) {

    LOG_DEBUG("grpc_backend") << "[list_available_models] chamado; directory='" << directory
                              << "' channel_ready=" << is_channel_ready();
    std::vector<AvailableModel> result;
    if (!is_channel_ready()) {
        LOG_WARN("grpc_backend") << "[list_available_models] canal não disponível, retornando lista vazia";
        return result;
    }

    grpc::ClientContext                  ctx;
    server::ListAvailableModelsRequest   req;
    server::ListAvailableModelsResponse  resp;

    req.set_directory(directory);
    LOG_DEBUG("grpc_backend") << "[list_available_models] chamando stub_->ListAvailableModels(directory='" << directory << "')";
    auto st = stub_->ListAvailableModels(&ctx, req, &resp);
    LOG_DEBUG("grpc_backend") << "[list_available_models] retornou ok=" << st.ok()
                              << " error_code=" << static_cast<int>(st.error_code())
                              << " error_message='" << st.error_message() << "'";
    if (!st.ok()) {
        LOG_WARN("grpc_backend") << "[list_available_models] FALHA RPC: " << st.error_message();
        return result;
    }

    LOG_DEBUG("grpc_backend") << "[list_available_models] n_modelos=" << resp.models_size();
    for (const auto& m : resp.models()) {
        AvailableModel am;
        am.filename        = m.filename();
        am.path            = m.path();
        am.extension       = m.extension();
        am.file_size_bytes = m.file_size_bytes();
        am.is_loaded       = m.is_loaded();
        am.loaded_as       = m.loaded_as();

        switch (m.backend()) {
            case common::BACKEND_ONNX:   am.backend = "onnx";    break;
            case common::BACKEND_PYTHON: am.backend = "python";  break;
            default:                      am.backend = "unknown"; break;
        }

        LOG_DEBUG("grpc_backend") << "[list_available_models] modelo: filename='" << am.filename
                                  << "' backend='" << am.backend
                                  << "' is_loaded=" << am.is_loaded
                                  << " loaded_as='" << am.loaded_as
                                  << "' file_size_bytes=" << am.file_size_bytes;
        result.push_back(std::move(am));
    }

    LOG_DEBUG("grpc_backend") << "[list_available_models] concluído; n_modelos=" << result.size();
    return result;
}

// =============================================================================
// Helpers privados
// =============================================================================

std::string GrpcClientBackend::dtype_str(common::DataType dt) {
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

ModelInfo GrpcClientBackend::proto_to_model_info(const common::ModelInfo& p) {
    LOG_DEBUG("grpc_backend") << "[proto_to_model_info] model_id='" << p.model_id()
                              << "' version='" << p.version()
                              << "' backend_enum=" << static_cast<int>(p.backend());
    ModelInfo info;
    info.model_id           = p.model_id();
    info.version            = p.version();
    info.description        = p.description();
    info.author             = p.author();
    info.memory_usage_bytes = p.memory_usage_bytes();
    info.is_warmed_up       = p.is_warmed_up();
    info.loaded_at_unix     = p.loaded_at_unix();

    switch (p.backend()) {
        case common::BACKEND_ONNX:   info.backend = "onnx";    break;
        case common::BACKEND_PYTHON: info.backend = "python";  break;
        default:                      info.backend = "unknown"; break;
    }

    LOG_DEBUG("grpc_backend") << "[proto_to_model_info] backend='" << info.backend
                              << "' n_inputs=" << p.inputs_size()
                              << " n_outputs=" << p.outputs_size()
                              << " n_tags=" << p.tags_size();

    for (const auto& ts : p.inputs()) {
        ModelInfo::TensorSpec s;
        s.name        = ts.name();
        s.dtype       = dtype_str(ts.dtype());
        s.description = ts.description();
        s.structured  = ts.structured();
        for (int64_t d : ts.shape()) s.shape.push_back(d);
        LOG_DEBUG("grpc_backend") << "[proto_to_model_info] input name='" << s.name
                                  << "' dtype='" << s.dtype
                                  << "' structured=" << s.structured
                                  << " shape_size=" << s.shape.size();
        info.inputs.push_back(std::move(s));
    }
    for (const auto& ts : p.outputs()) {
        ModelInfo::TensorSpec s;
        s.name        = ts.name();
        s.dtype       = dtype_str(ts.dtype());
        s.description = ts.description();
        s.structured  = ts.structured();
        for (int64_t d : ts.shape()) s.shape.push_back(d);
        LOG_DEBUG("grpc_backend") << "[proto_to_model_info] output name='" << s.name
                                  << "' dtype='" << s.dtype
                                  << "' structured=" << s.structured
                                  << " shape_size=" << s.shape.size();
        info.outputs.push_back(std::move(s));
    }
    for (const auto& [k, v] : p.tags()) {
        LOG_DEBUG("grpc_backend") << "[proto_to_model_info] tag '" << k << "'='" << v << "'";
        info.tags[k] = v;
    }

    LOG_DEBUG("grpc_backend") << "[proto_to_model_info] concluído para model_id='" << info.model_id << "'";
    return info;
}

}  // namespace client
}  // namespace mlinference

// // =============================================================================
// // grpc_client_backend.cpp — Backend gRPC do cliente Miia
// //
// // Implementa IClientBackend via chamadas gRPC ao WorkerService.
// // Converte client::Object ↔ google.protobuf.Struct na fronteira.
// // =============================================================================

// #include "client/grpc_client_backend.hpp"
// #include "client/value_convert.hpp"
// #include "utils/logger.hpp"

// #include <grpcpp/grpcpp.h>
// #include "server.grpc.pb.h"
// #include "common.pb.h"

// #include <chrono>

// namespace mlinference {
// namespace client {

// // =============================================================================
// // Conexão
// // =============================================================================

// bool GrpcClientBackend::connect() {
//     LOG_DEBUG("grpc_backend") << "[connect] chamado; server_address_='" << server_address_ << "'";
//     auto channel = grpc::CreateChannel(
//         server_address_, grpc::InsecureChannelCredentials());
//     LOG_DEBUG("grpc_backend") << "[connect] channel criado para '" << server_address_ << "'";
//     stub_ = server::WorkerService::NewStub(channel);
//     LOG_DEBUG("grpc_backend") << "[connect] stub_ criado; stub_=" << (void*)stub_.get();
//     grpc::ClientContext ctx;
//     auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
//     ctx.set_deadline(deadline);
//     LOG_DEBUG("grpc_backend") << "[connect] deadline de 5s configurado; chamando HealthCheck()";
//     server::HealthCheckRequest  req;
//     server::HealthCheckResponse resp;
//     auto st = stub_->HealthCheck(&ctx, req, &resp);

//     LOG_DEBUG("grpc_backend") << "[connect] HealthCheck retornou ok=" << st.ok()
//          << " healthy=" << resp.healthy()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     connected_ = st.ok() && resp.healthy();
//     if (connected_) {
//         LOG_INFO("grpc_backend") << "[connect] conectado com sucesso a '" << server_address_ << "'";
//     } else {
//         LOG_ERROR("grpc_backend") << "[connect] falha ao conectar; ok=" << st.ok()
//              << " healthy=" << resp.healthy()
//              << " error_message='" << st.error_message() << "'";
//     }
//     return connected_;
// }

// // =============================================================================
// // Ciclo de vida dos modelos
// // =============================================================================

// bool GrpcClientBackend::load_model(const std::string& model_id,
//                                     const std::string& model_path,
//                                     const std::string& /*version*/) {
//     LOG_DEBUG("grpc_backend") << "[load_model] chamado; model_id='" << model_id
//          << "' model_path='" << model_path << "' connected_=" << connected_;
//     if (!connected_) {
//         LOG_ERROR("grpc_backend") << "[load_model] FALHA PRÉ-CONDIÇÃO: não conectado";
//         return false;
//     }

//     grpc::ClientContext       ctx;
//     server::LoadModelRequest  req;
//     server::LoadModelResponse resp;

//     req.set_model_id(model_id);
//     req.set_model_path(model_path);
//     LOG_DEBUG("grpc_backend") << "[load_model] chamando stub_->LoadModel(model_id='" << model_id
//          << "' model_path='" << model_path << "')";
//     auto st = stub_->LoadModel(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[load_model] LoadModel retornou ok=" << st.ok()
//          << " success=" << resp.success()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     bool result = st.ok() && resp.success();
//     if (result) {
//         LOG_INFO("grpc_backend") << "[load_model] modelo carregado: model_id='" << model_id << "'";
//     } else {
//         LOG_ERROR("grpc_backend") << "[load_model] falha ao carregar model_id='" << model_id
//              << "'; ok=" << st.ok() << " success=" << resp.success()
//              << " error_message='" << st.error_message() << "'";
//     }
//     return result;
// }

// bool GrpcClientBackend::unload_model(const std::string& model_id) {
//     LOG_DEBUG("grpc_backend") << "[unload_model] chamado; model_id='" << model_id
//          << "' connected_=" << connected_;
//     if (!connected_) {
//         LOG_ERROR("grpc_backend") << "[unload_model] FALHA PRÉ-CONDIÇÃO: não conectado";
//         return false;
//     }

//     grpc::ClientContext         ctx;
//     server::UnloadModelRequest  req;
//     server::UnloadModelResponse resp;

//     req.set_model_id(model_id);
//     LOG_DEBUG("grpc_backend") << "[unload_model] chamando stub_->UnloadModel(model_id='" << model_id << "')";
//     auto st = stub_->UnloadModel(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[unload_model] UnloadModel retornou ok=" << st.ok()
//          << " success=" << resp.success()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     bool result = st.ok() && resp.success();
//     if (result) {
//         LOG_INFO("grpc_backend") << "[unload_model] modelo descarregado: model_id='" << model_id << "'";
//     } else {
//         LOG_ERROR("grpc_backend") << "[unload_model] falha ao descarregar model_id='" << model_id
//              << "'; ok=" << st.ok() << " success=" << resp.success()
//              << " error_message='" << st.error_message() << "'";
//     }
//     return result;
// }

// // =============================================================================
// // Inferência
// // =============================================================================

// PredictionResult GrpcClientBackend::predict(const std::string& model_id,
//                                              const Object& inputs) {
//     LOG_DEBUG("grpc_backend") << "[predict] chamado; model_id='" << model_id
//          << "' n_inputs=" << inputs.size() << " connected_=" << connected_;
//     PredictionResult result;
//     if (!connected_) {
//         LOG_ERROR("grpc_backend") << "[predict] FALHA PRÉ-CONDIÇÃO: não conectado";
//         result.error_message = "Not connected";
//         return result;
//     }

//     for (const auto& [k, v] : inputs) {
//         LOG_DEBUG("grpc_backend") << "[predict] input key='" << k << "'";
//     }

//     grpc::ClientContext     ctx;
//     server::PredictRequest  req;
//     server::PredictResponse resp;

//     req.set_model_id(model_id);
//     *req.mutable_inputs() = to_proto_struct(inputs);
//     LOG_DEBUG("grpc_backend") << "[predict] chamando stub_->Predict()";
//     auto st = stub_->Predict(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[predict] Predict retornou ok=" << st.ok()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     if (!st.ok()) {
//         result.error_message = "RPC failed: " + st.error_message();
//         LOG_ERROR("grpc_backend") << "[predict] FALHA RPC: " << result.error_message;
//         return result;
//     }

//     result.success           = resp.success();
//     result.inference_time_ms = resp.inference_time_ms();
//     result.error_message     = resp.error_message();

//     LOG_DEBUG("grpc_backend") << "[predict] resp.success()=" << resp.success()
//          << " inference_time_ms=" << resp.inference_time_ms()
//          << " resp.error_message='" << resp.error_message() << "'";
//     if (resp.success()) {
//         result.outputs = from_proto_struct(resp.outputs());
//         LOG_DEBUG("grpc_backend") << "[predict] n_outputs=" << result.outputs.size();
//         for (const auto& [k, v] : result.outputs) {
//             LOG_DEBUG("grpc_backend") << "[predict] output key='" << k << "'";
//         }
//     } else {
//         LOG_WARN("grpc_backend") << "[predict] resp.success()=false; error_message='" << resp.error_message() << "'";
//     }

//     LOG_DEBUG("grpc_backend") << "[predict] concluído; success=" << result.success
//          << " inference_time_ms=" << result.inference_time_ms;
//     return result;
// }

// std::vector<PredictionResult> GrpcClientBackend::batch_predict(
//     const std::string& model_id,
//     const std::vector<Object>& batch_inputs) {

//     LOG_DEBUG("grpc_backend") << "[batch_predict] chamado; model_id='" << model_id
//          << "' batch_size=" << batch_inputs.size() << " connected_=" << connected_;
//     std::vector<PredictionResult> results;
//     if (!connected_) {
//         LOG_ERROR("grpc_backend") << "[batch_predict] FALHA PRÉ-CONDIÇÃO: não conectado";
//         return results;
//     }

//     grpc::ClientContext          ctx;
//     server::BatchPredictRequest  req;
//     server::BatchPredictResponse resp;

//     req.set_model_id(model_id);
//     for (size_t i = 0; i < batch_inputs.size(); ++i) {
//         auto* single = req.add_requests();
//         single->set_model_id(model_id);
//         *single->mutable_inputs() = to_proto_struct(batch_inputs[i]);
//         LOG_DEBUG("grpc_backend") << "[batch_predict] item[" << i << "] n_keys=" << batch_inputs[i].size();
//     }

//     LOG_DEBUG("grpc_backend") << "[batch_predict] chamando stub_->BatchPredict() com " << batch_inputs.size() << " itens";
//     auto st = stub_->BatchPredict(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[batch_predict] BatchPredict retornou ok=" << st.ok()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     if (!st.ok()) {
//         LOG_ERROR("grpc_backend") << "[batch_predict] FALHA RPC: " << st.error_message();
//         return results;
//     }

//     LOG_DEBUG("grpc_backend") << "[batch_predict] n_respostas=" << resp.responses_size();
//     for (int i = 0; i < resp.responses_size(); ++i) {
//         const auto& r = resp.responses(i);
//         PredictionResult pr;
//         pr.success           = r.success();
//         pr.inference_time_ms = r.inference_time_ms();
//         pr.error_message     = r.error_message();

//         LOG_DEBUG("grpc_backend") << "[batch_predict] response[" << i << "] success=" << r.success()
//              << " inference_time_ms=" << r.inference_time_ms()
//              << " error_message='" << r.error_message() << "'";
//         if (r.success()) {
//             pr.outputs = from_proto_struct(r.outputs());
//             LOG_DEBUG("grpc_backend") << "[batch_predict] response[" << i << "] n_outputs=" << pr.outputs.size();
//         } else {
//             LOG_WARN("grpc_backend") << "[batch_predict] response[" << i << "] success=false; error_message='" << r.error_message() << "'";
//         }
//         results.push_back(std::move(pr));
//     }

//     LOG_DEBUG("grpc_backend") << "[batch_predict] concluído; n_results=" << results.size();
//     return results;
// }

// // =============================================================================
// // Introspecção
// // =============================================================================

// std::vector<ModelInfo> GrpcClientBackend::list_models() {
//     LOG_DEBUG("grpc_backend") << "[list_models] chamado; connected_=" << connected_;
//     std::vector<ModelInfo> result;
//     if (!connected_) {
//         LOG_WARN("grpc_backend") << "[list_models] não conectado, retornando lista vazia";
//         return result;
//     }

//     grpc::ClientContext        ctx;
//     server::ListModelsRequest  req;
//     server::ListModelsResponse resp;

//     LOG_DEBUG("grpc_backend") << "[list_models] chamando stub_->ListModels()";
//     auto st = stub_->ListModels(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[list_models] ListModels retornou ok=" << st.ok()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     if (!st.ok()) {
//         LOG_WARN("grpc_backend") << "[list_models] FALHA RPC: " << st.error_message();
//         return result;
//     }

//     LOG_DEBUG("grpc_backend") << "[list_models] n_modelos=" << resp.models_size();
//     for (const auto& p : resp.models()) {
//         LOG_DEBUG("grpc_backend") << "[list_models] convertendo model_id='" << p.model_id() << "'";
//         result.push_back(proto_to_model_info(p));
//     }

//     LOG_DEBUG("grpc_backend") << "[list_models] retornando " << result.size() << " modelos";
//     return result;
// }

// ModelInfo GrpcClientBackend::get_model_info(const std::string& model_id) {
//     LOG_DEBUG("grpc_backend") << "[get_model_info] chamado; model_id='" << model_id
//          << "' connected_=" << connected_;
//     if (!connected_) {
//         LOG_WARN("grpc_backend") << "[get_model_info] não conectado, retornando ModelInfo vazio";
//         return {};
//     }

//     grpc::ClientContext          ctx;
//     server::GetModelInfoRequest  req;
//     server::GetModelInfoResponse resp;

//     req.set_model_id(model_id);
//     LOG_DEBUG("grpc_backend") << "[get_model_info] chamando stub_->GetModelInfo(model_id='" << model_id << "')";
//     auto st = stub_->GetModelInfo(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[get_model_info] GetModelInfo retornou ok=" << st.ok()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     if (!st.ok()) {
//         LOG_WARN("grpc_backend") << "[get_model_info] FALHA RPC: " << st.error_message();
//         return {};
//     }

//     LOG_DEBUG("grpc_backend") << "[get_model_info] convertendo proto para ModelInfo";
//     return proto_to_model_info(resp.model_info());
// }

// ValidationResult GrpcClientBackend::validate_model(const std::string& path) {
//     LOG_DEBUG("grpc_backend") << "[validate_model] chamado; path='" << path
//          << "' connected_=" << connected_;
//     ValidationResult result;
//     if (!connected_) {
//         LOG_ERROR("grpc_backend") << "[validate_model] FALHA PRÉ-CONDIÇÃO: não conectado";
//         return result;
//     }

//     grpc::ClientContext            ctx;
//     server::ValidateModelRequest   req;
//     server::ValidateModelResponse  resp;

//     req.set_model_path(path);
//     LOG_DEBUG("grpc_backend") << "[validate_model] chamando stub_->ValidateModel(path='" << path << "')";
//     auto st = stub_->ValidateModel(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[validate_model] ValidateModel retornou ok=" << st.ok()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     if (!st.ok()) {
//         LOG_ERROR("grpc_backend") << "[validate_model] FALHA RPC: " << st.error_message();
//         return result;
//     }

//     result.valid         = resp.valid();
//     result.error_message = resp.error_message();
//     LOG_DEBUG("grpc_backend") << "[validate_model] valid=" << resp.valid()
//          << " error_message='" << resp.error_message() << "'"
//          << " backend_enum=" << static_cast<int>(resp.backend())
//          << " n_inputs=" << resp.inputs_size()
//          << " n_outputs=" << resp.outputs_size()
//          << " n_warnings=" << resp.warnings_size();
//     switch (resp.backend()) {
//         case common::BACKEND_ONNX:   result.backend = "onnx";    break;
//         case common::BACKEND_PYTHON: result.backend = "python";  break;
//         default:                      result.backend = "unknown"; break;
//     }
//     LOG_DEBUG("grpc_backend") << "[validate_model] backend='" << result.backend << "'";
//     for (const auto& ts : resp.inputs()) {
//         ModelInfo::TensorSpec s;
//         s.name        = ts.name();
//         s.dtype       = dtype_str(ts.dtype());
//         s.description = ts.description();
//         s.structured  = ts.structured();
//         for (int64_t d : ts.shape()) s.shape.push_back(d);
//         LOG_DEBUG("grpc_backend") << "[validate_model] input spec name='" << s.name
//              << "' dtype='" << s.dtype << "' structured=" << s.structured
//              << " shape_size=" << s.shape.size();
//         result.inputs.push_back(std::move(s));
//     }

//     for (const auto& ts : resp.outputs()) {
//         ModelInfo::TensorSpec s;
//         s.name        = ts.name();
//         s.dtype       = dtype_str(ts.dtype());
//         s.description = ts.description();
//         s.structured  = ts.structured();
//         for (int64_t d : ts.shape()) s.shape.push_back(d);
//         LOG_DEBUG("grpc_backend") << "[validate_model] output spec name='" << s.name
//              << "' dtype='" << s.dtype << "' structured=" << s.structured
//              << " shape_size=" << s.shape.size();
//         result.outputs.push_back(std::move(s));
//     }

//     for (const auto& w : resp.warnings()) {
//         LOG_WARN("grpc_backend") << "[validate_model] warning='" << w << "'";
//         result.warnings.push_back(w);
//     }

//     if (!result.valid) {
//         LOG_WARN("grpc_backend") << "[validate_model] modelo inválido; error_message='" << result.error_message << "'";
//     }
//     LOG_DEBUG("grpc_backend") << "[validate_model] concluído; valid=" << result.valid;
//     return result;
// }

// WarmupResult GrpcClientBackend::warmup_model(const std::string& model_id,
//                                               uint32_t num_runs) {
//     LOG_DEBUG("grpc_backend") << "[warmup_model] chamado; model_id='" << model_id
//          << "' num_runs=" << num_runs << " connected_=" << connected_;
//     WarmupResult result;
//     if (!connected_) {
//         LOG_ERROR("grpc_backend") << "[warmup_model] FALHA PRÉ-CONDIÇÃO: não conectado";
//         return result;
//     }

//     grpc::ClientContext          ctx;
//     server::WarmupModelRequest   req;
//     server::WarmupModelResponse  resp;

//     req.set_model_id(model_id);
//     req.set_num_runs(num_runs);
//     LOG_DEBUG("grpc_backend") << "[warmup_model] chamando stub_->WarmupModel(model_id='" << model_id
//          << "' num_runs=" << num_runs << ")";
//     auto st = stub_->WarmupModel(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[warmup_model] WarmupModel retornou ok=" << st.ok()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     if (!st.ok()) {
//         LOG_ERROR("grpc_backend") << "[warmup_model] FALHA RPC: " << st.error_message();
//         return result;
//     }

//     result.success        = resp.success();
//     result.runs_completed = resp.runs_completed();
//     result.avg_time_ms    = resp.avg_time_ms();
//     result.min_time_ms    = resp.min_time_ms();
//     result.max_time_ms    = resp.max_time_ms();
//     result.error_message  = resp.error_message();

//     if (result.success) {
//         LOG_INFO("grpc_backend") << "[warmup_model] warmup concluído: model_id='" << model_id
//              << "' runs_completed=" << result.runs_completed
//              << " avg_time_ms=" << result.avg_time_ms
//              << " min_time_ms=" << result.min_time_ms
//              << " max_time_ms=" << result.max_time_ms;
//     } else {
//         LOG_WARN("grpc_backend") << "[warmup_model] warmup sem sucesso; error_message='" << result.error_message << "'";
//     }
//     return result;
// }

// // =============================================================================
// // Observabilidade
// // =============================================================================

// bool GrpcClientBackend::health_check() {
//     LOG_DEBUG("grpc_backend") << "[health_check] chamado; connected_=" << connected_;
//     if (!connected_) {
//         LOG_WARN("grpc_backend") << "[health_check] não conectado, retornando false";
//         return false;
//     }

//     grpc::ClientContext         ctx;
//     server::HealthCheckRequest  req;
//     server::HealthCheckResponse resp;

//     auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(3);
//     ctx.set_deadline(deadline);
//     LOG_DEBUG("grpc_backend") << "[health_check] deadline de 3s configurado; chamando stub_->HealthCheck()";
//     auto st = stub_->HealthCheck(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[health_check] HealthCheck retornou ok=" << st.ok()
//          << " healthy=" << resp.healthy()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     bool result = st.ok() && resp.healthy();
//     if (!result) {
//         LOG_WARN("grpc_backend") << "[health_check] serviço não saudável; ok=" << st.ok()
//              << " healthy=" << resp.healthy();
//     }
//     return result;
// }

// WorkerStatus GrpcClientBackend::get_status() {
//     LOG_DEBUG("grpc_backend") << "[get_status] chamado; connected_=" << connected_;
//     WorkerStatus result;
//     if (!connected_) {
//         LOG_WARN("grpc_backend") << "[get_status] não conectado, retornando WorkerStatus vazio";
//         return result;
//     }

//     grpc::ClientContext        ctx;
//     server::GetStatusRequest   req;
//     server::GetStatusResponse  resp;

//     LOG_DEBUG("grpc_backend") << "[get_status] chamando stub_->GetStatus()";
//     auto st = stub_->GetStatus(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[get_status] GetStatus retornou ok=" << st.ok()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     if (!st.ok()) {
//         LOG_WARN("grpc_backend") << "[get_status] FALHA RPC: " << st.error_message();
//         return result;
//     }

//     result.worker_id = resp.worker_id();
//     LOG_DEBUG("grpc_backend") << "[get_status] worker_id='" << resp.worker_id() << "'";
//     const auto& metrics = resp.metrics();
//     result.total_requests      = metrics.total_requests();
//     result.successful_requests = metrics.successful_requests();
//     result.failed_requests     = metrics.failed_requests();
//     result.active_requests     = metrics.active_requests();
//     result.uptime_seconds      = metrics.uptime_seconds();

//     LOG_DEBUG("grpc_backend") << "[get_status] metrics: total=" << metrics.total_requests()
//          << " successful=" << metrics.successful_requests()
//          << " failed=" << metrics.failed_requests()
//          << " active=" << metrics.active_requests()
//          << " uptime_seconds=" << metrics.uptime_seconds();
//     LOG_DEBUG("grpc_backend") << "[get_status] n_loaded_model_ids=" << resp.loaded_model_ids_size();
//     for (const auto& id : resp.loaded_model_ids()) {
//         LOG_DEBUG("grpc_backend") << "[get_status] loaded model='" << id << "'";
//         result.loaded_models.push_back(id);
//     }

//     LOG_DEBUG("grpc_backend") << "[get_status] n_supported_backends=" << resp.capabilities().supported_backends_size();
//     for (const auto& b : resp.capabilities().supported_backends()) {
//         LOG_DEBUG("grpc_backend") << "[get_status] supported backend='" << b << "'";
//         result.supported_backends.push_back(b);
//     }

//     return result;
// }

// ServerMetrics GrpcClientBackend::get_metrics() {
//     LOG_DEBUG("grpc_backend") << "[get_metrics] chamado; connected_=" << connected_;
//     ServerMetrics result;
//     if (!connected_) {
//         LOG_WARN("grpc_backend") << "[get_metrics] não conectado, retornando ServerMetrics vazio";
//         return result;
//     }

//     grpc::ClientContext        ctx;
//     server::GetMetricsRequest  req;
//     server::GetMetricsResponse resp;

//     LOG_DEBUG("grpc_backend") << "[get_metrics] chamando stub_->GetMetrics()";
//     auto st = stub_->GetMetrics(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[get_metrics] GetMetrics retornou ok=" << st.ok()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     if (!st.ok()) {
//         LOG_WARN("grpc_backend") << "[get_metrics] FALHA RPC: " << st.error_message();
//         return result;
//     }

//     const auto& wm = resp.worker_metrics();
//     result.total_requests      = wm.total_requests();
//     result.successful_requests = wm.successful_requests();
//     result.failed_requests     = wm.failed_requests();
//     result.active_requests     = wm.active_requests();
//     result.uptime_seconds      = wm.uptime_seconds();

//     LOG_DEBUG("grpc_backend") << "[get_metrics] worker_metrics: total=" << wm.total_requests()
//          << " successful=" << wm.successful_requests()
//          << " failed=" << wm.failed_requests()
//          << " active=" << wm.active_requests()
//          << " uptime_seconds=" << wm.uptime_seconds();
//     LOG_DEBUG("grpc_backend") << "[get_metrics] n_per_model_metrics=" << resp.per_model_metrics().size();
//     for (auto it = resp.per_model_metrics().begin();
//          it != resp.per_model_metrics().end(); ++it) {

//         const auto& mm = it->second;
//         ModelMetrics m;
//         m.total_inferences  = mm.total_inferences();
//         m.failed_inferences = mm.failed_inferences();
//         m.avg_ms            = mm.avg_inference_time_ms();
//         m.min_ms            = mm.min_inference_time_ms();
//         m.max_ms            = mm.max_inference_time_ms();
//         m.p95_ms            = mm.p95_inference_time_ms();
//         m.p99_ms            = mm.p99_inference_time_ms();
//         m.total_time_ms     = mm.total_inference_time_ms();
//         m.last_used_at_unix = mm.last_used_at_unix();
//         m.loaded_at_unix    = mm.loaded_at_unix();

//         LOG_DEBUG("grpc_backend") << "[get_metrics] model='" << it->first
//              << "' total_inferences=" << m.total_inferences
//              << " failed_inferences=" << m.failed_inferences
//              << " avg_ms=" << m.avg_ms
//              << " min_ms=" << m.min_ms
//              << " max_ms=" << m.max_ms
//              << " p95_ms=" << m.p95_ms
//              << " p99_ms=" << m.p99_ms
//              << " total_time_ms=" << m.total_time_ms;
//         result.per_model[it->first] = std::move(m);
//     }

//     LOG_DEBUG("grpc_backend") << "[get_metrics] concluído; n_per_model=" << result.per_model.size();
//     return result;
// }

// // =============================================================================
// // Descoberta de arquivos
// // =============================================================================

// std::vector<AvailableModel> GrpcClientBackend::list_available_models(
//     const std::string& directory) {

//     LOG_DEBUG("grpc_backend") << "[list_available_models] chamado; directory='" << directory
//          << "' connected_=" << connected_;
//     std::vector<AvailableModel> result;
//     if (!connected_) {
//         LOG_WARN("grpc_backend") << "[list_available_models] não conectado, retornando lista vazia";
//         return result;
//     }

//     grpc::ClientContext                 ctx;
//     server::ListAvailableModelsRequest  req;
//     server::ListAvailableModelsResponse resp;

//     req.set_directory(directory);
//     LOG_DEBUG("grpc_backend") << "[list_available_models] chamando stub_->ListAvailableModels(directory='" << directory << "')";
//     auto st = stub_->ListAvailableModels(&ctx, req, &resp);
//     LOG_DEBUG("grpc_backend") << "[list_available_models] ListAvailableModels retornou ok=" << st.ok()
//          << " error_code=" << static_cast<int>(st.error_code())
//          << " error_message='" << st.error_message() << "'";
//     if (!st.ok()) {
//         LOG_WARN("grpc_backend") << "[list_available_models] FALHA RPC: " << st.error_message();
//         return result;
//     }

//     LOG_DEBUG("grpc_backend") << "[list_available_models] n_modelos=" << resp.models_size();
//     for (const auto& m : resp.models()) {
//         AvailableModel am;
//         am.filename        = m.filename();
//         am.path            = m.path();
//         am.extension       = m.extension();
//         am.file_size_bytes = m.file_size_bytes();
//         am.is_loaded       = m.is_loaded();
//         am.loaded_as       = m.loaded_as();

//         switch (m.backend()) {
//             case common::BACKEND_ONNX:   am.backend = "onnx";    break;
//             case common::BACKEND_PYTHON: am.backend = "python";  break;
//             default:                      am.backend = "unknown"; break;
//         }

//         LOG_DEBUG("grpc_backend") << "[list_available_models] modelo: filename='" << am.filename
//              << "' backend='" << am.backend << "' is_loaded=" << am.is_loaded
//              << " loaded_as='" << am.loaded_as << "' file_size_bytes=" << am.file_size_bytes;
//         result.push_back(std::move(am));
//     }

//     LOG_DEBUG("grpc_backend") << "[list_available_models] concluído; n_modelos=" << result.size();
//     return result;
// }

// // =============================================================================
// // Helpers privados
// // =============================================================================

// std::string GrpcClientBackend::dtype_str(common::DataType dt) {
//     switch (dt) {
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

// ModelInfo GrpcClientBackend::proto_to_model_info(const common::ModelInfo& p) {
//     LOG_DEBUG("grpc_backend") << "[proto_to_model_info] model_id='" << p.model_id()
//          << "' version='" << p.version() << "' backend_enum=" << static_cast<int>(p.backend());
//     ModelInfo info;
//     info.model_id           = p.model_id();
//     info.version            = p.version();
//     info.description        = p.description();
//     info.author             = p.author();
//     info.memory_usage_bytes = p.memory_usage_bytes();
//     info.is_warmed_up       = p.is_warmed_up();
//     info.loaded_at_unix     = p.loaded_at_unix();

//     LOG_DEBUG("grpc_backend") << "[proto_to_model_info] description='" << p.description()
//          << "' author='" << p.author() << "' memory_usage_bytes=" << p.memory_usage_bytes()
//          << " is_warmed_up=" << p.is_warmed_up() << " loaded_at_unix=" << p.loaded_at_unix();
//     switch (p.backend()) {
//         case common::BACKEND_ONNX:   info.backend = "onnx";    break;
//         case common::BACKEND_PYTHON: info.backend = "python";  break;
//         default:                      info.backend = "unknown"; break;
//     }
//     LOG_DEBUG("grpc_backend") << "[proto_to_model_info] backend='" << info.backend << "'";
//     LOG_DEBUG("grpc_backend") << "[proto_to_model_info] n_inputs=" << p.inputs_size()
//          << " n_outputs=" << p.outputs_size() << " n_tags=" << p.tags_size();
//     for (const auto& ts : p.inputs()) {
//         ModelInfo::TensorSpec s;
//         s.name        = ts.name();
//         s.dtype       = dtype_str(ts.dtype());
//         s.description = ts.description();
//         s.structured  = ts.structured();
//         for (int64_t d : ts.shape()) s.shape.push_back(d);
//         LOG_DEBUG("grpc_backend") << "[proto_to_model_info] input name='" << s.name
//              << "' dtype='" << s.dtype << "' structured=" << s.structured
//              << " shape_size=" << s.shape.size();
//         info.inputs.push_back(std::move(s));
//     }

//     for (const auto& ts : p.outputs()) {
//         ModelInfo::TensorSpec s;
//         s.name        = ts.name();
//         s.dtype       = dtype_str(ts.dtype());
//         s.description = ts.description();
//         s.structured  = ts.structured();
//         for (int64_t d : ts.shape()) s.shape.push_back(d);
//         LOG_DEBUG("grpc_backend") << "[proto_to_model_info] output name='" << s.name
//              << "' dtype='" << s.dtype << "' structured=" << s.structured
//              << " shape_size=" << s.shape.size();
//         info.outputs.push_back(std::move(s));
//     }

//     for (const auto& [k, v] : p.tags()) {
//         LOG_DEBUG("grpc_backend") << "[proto_to_model_info] tag '" << k << "'='" << v << "'";
//         info.tags[k] = v;
//     }

//     LOG_DEBUG("grpc_backend") << "[proto_to_model_info] concluído para model_id='" << info.model_id << "'";
//     return info;
// }

// }  // namespace client
// }  // namespace mlinference
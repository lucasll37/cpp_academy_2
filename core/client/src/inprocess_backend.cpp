// =============================================================================
// inprocess_backend.cpp — Backend in-process do cliente AsaMiia
//
// Roda o InferenceEngine diretamente no processo do cliente.
// Nenhuma camada gRPC — Object é repassado ao engine sem conversão.
// =============================================================================

#include "client/inprocess_backend.hpp"
#include "inference/backend_registry.hpp"
#include "common.pb.h"
#include "utils/logger.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <cstdlib>

namespace mlinference {
namespace client {

using inference::InferenceEngine;
using inference::TensorSpecData;

// =============================================================================
// Helpers locais
// =============================================================================

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

static ModelInfo::TensorSpec tensor_spec_from_data(const TensorSpecData& d) {
    ModelInfo::TensorSpec s;
    s.name        = d.name;
    s.dtype       = dtype_str(d.dtype);
    s.description = d.description;
    s.structured  = d.structured;
    s.shape       = d.shape;
    return s;
}

// =============================================================================
// Helpers privados da classe
// =============================================================================

ModelInfo InProcessBackend::proto_to_model_info(const common::ModelInfo& p) {
    LOG_DEBUG("inprocess_backend") << "[proto_to_model_info] model_id='" << p.model_id()
         << "' version='" << p.version() << "' backend=" << p.backend();

    ModelInfo info;
    info.model_id           = p.model_id();
    info.version            = p.version();
    info.description        = p.description();
    info.author             = p.author();
    info.memory_usage_bytes = p.memory_usage_bytes();
    info.is_warmed_up       = p.is_warmed_up();
    info.loaded_at_unix     = p.loaded_at_unix();

    LOG_DEBUG("inprocess_backend") << "[proto_to_model_info] description='" << p.description()
         << "' author='" << p.author() << "' memory_usage_bytes=" << p.memory_usage_bytes()
         << " is_warmed_up=" << p.is_warmed_up();

    switch (p.backend()) {
        case common::BACKEND_ONNX:   info.backend = "onnx";    break;
        case common::BACKEND_PYTHON: info.backend = "python";  break;
        default:                      info.backend = "unknown"; break;
    }
    LOG_DEBUG("inprocess_backend") << "[proto_to_model_info] backend resolvido='" << info.backend << "'";

    LOG_DEBUG("inprocess_backend") << "[proto_to_model_info] n_inputs=" << p.inputs_size()
         << " n_outputs=" << p.outputs_size();

    for (const auto& ts : p.inputs()) {
        ModelInfo::TensorSpec s;
        s.name        = ts.name();
        s.dtype       = dtype_str(ts.dtype());
        s.description = ts.description();
        s.structured  = ts.structured();
        for (int64_t d : ts.shape()) s.shape.push_back(d);
        LOG_DEBUG("inprocess_backend") << "[proto_to_model_info] input name='" << s.name
             << "' dtype='" << s.dtype << "' structured=" << s.structured
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
        LOG_DEBUG("inprocess_backend") << "[proto_to_model_info] output name='" << s.name
             << "' dtype='" << s.dtype << "' structured=" << s.structured
             << " shape_size=" << s.shape.size();
        info.outputs.push_back(std::move(s));
    }

    for (const auto& [k, v] : p.tags()) {
        LOG_DEBUG("inprocess_backend") << "[proto_to_model_info] tag '" << k << "'='" << v << "'";
        info.tags[k] = v;
    }

    LOG_DEBUG("inprocess_backend") << "[proto_to_model_info] concluído para model_id='" << info.model_id << "'";
    return info;
}

// =============================================================================
// Helpers de resolução de path
// =============================================================================

static std::string resolve_path(const std::string& path) {
    namespace fs = std::filesystem;
    LOG_DEBUG("inprocess_backend") << "[resolve_path] path recebido='" << path << "'";
    fs::path p(path);
    if (!p.is_absolute()) {
        p = fs::current_path() / p;
        LOG_DEBUG("inprocess_backend") << "[resolve_path] path relativo; current_path=" << fs::current_path().string();
    }
    std::string resolved = fs::weakly_canonical(p).string();
    LOG_DEBUG("inprocess_backend") << "[resolve_path] path resolvido='" << resolved << "' exists=" << fs::exists(resolved);
    return resolved;
}

// =============================================================================
// Conexão
// =============================================================================

bool InProcessBackend::connect() {
    LOG_DEBUG("inprocess_backend") << "[connect] chamado; connected_=" << connected_;

    engine_ = std::make_unique<InferenceEngine>(
        /*enable_gpu=*/false,
        /*gpu_device_id=*/0,
        /*num_threads=*/4);

    LOG_DEBUG("inprocess_backend") << "[connect] InferenceEngine criado; engine_=" << (void*)engine_.get();

    connected_  = true;
    start_time_ = std::chrono::steady_clock::now();

    LOG_INFO("inprocess_backend") << "[connect] engine in-process inicializado com sucesso";
    return true;
}

// =============================================================================
// Ciclo de vida dos modelos
// =============================================================================

bool InProcessBackend::load_model(const std::string& model_id,
                                   const std::string& model_path,
                                   const std::string& /*version*/) {
    LOG_DEBUG("inprocess_backend") << "[load_model] chamado; model_id='" << model_id
         << "' model_path='" << model_path << "' connected_=" << connected_;

    if (!connected_) {
        LOG_ERROR("inprocess_backend") << "[load_model] FALHA PRÉ-CONDIÇÃO: não conectado";
        return false;
    }

    namespace fs = std::filesystem;
    const std::string resolved = resolve_path(model_path);
    LOG_DEBUG("inprocess_backend") << "[load_model] path resolvido='" << resolved << "'";
    LOG_DEBUG("inprocess_backend") << "[load_model] arquivo existe=" << fs::exists(resolved)
         << " extensão='" << fs::path(resolved).extension().string() << "'";

    LOG_DEBUG("inprocess_backend") << "[load_model] chamando engine_->load_model('" << model_id << "', '" << resolved << "')";
    bool ok = engine_->load_model(model_id, resolved);
    LOG_DEBUG("inprocess_backend") << "[load_model] engine_->load_model retornou ok=" << ok;

    if (ok) {
        loaded_models_.push_back(model_id);
        LOG_INFO("inprocess_backend") << "[load_model] modelo carregado: model_id='" << model_id
             << "' total carregados=" << loaded_models_.size();
    } else {
        LOG_ERROR("inprocess_backend") << "[load_model] falha ao carregar model_id='" << model_id << "'";
    }

    return ok;
}

// bool InProcessBackend::load_model(const std::string& model_id,
//                                    const std::string& model_path,
//                                    const std::string& /*version*/) {
//     ...
// }

bool InProcessBackend::unload_model(const std::string& model_id) {
    LOG_DEBUG("inprocess_backend") << "[unload_model] chamado; model_id='" << model_id
         << "' connected_=" << connected_;

    if (!connected_) {
        LOG_ERROR("inprocess_backend") << "[unload_model] FALHA PRÉ-CONDIÇÃO: não conectado";
        return false;
    }

    LOG_DEBUG("inprocess_backend") << "[unload_model] chamando engine_->unload_model('" << model_id << "')";
    bool ok = engine_->unload_model(model_id);
    LOG_DEBUG("inprocess_backend") << "[unload_model] engine_->unload_model retornou ok=" << ok;

    if (ok) {
        auto it = std::find(loaded_models_.begin(), loaded_models_.end(), model_id);
        if (it != loaded_models_.end()) {
            loaded_models_.erase(it);
            LOG_INFO("inprocess_backend") << "[unload_model] modelo descarregado: model_id='" << model_id
                 << "' total restantes=" << loaded_models_.size();
        } else {
            LOG_WARN("inprocess_backend") << "[unload_model] model_id='" << model_id
                 << "' não encontrado em loaded_models_ (já removido?)";
        }
    } else {
        LOG_ERROR("inprocess_backend") << "[unload_model] falha ao descarregar model_id='" << model_id << "'";
    }

    return ok;
}

// =============================================================================
// Inferência
// =============================================================================

PredictionResult InProcessBackend::predict(const std::string& model_id,
                                            const Object& inputs) {
    LOG_DEBUG("inprocess_backend") << "[predict] chamado; model_id='" << model_id
         << "' n_inputs=" << inputs.size() << " connected_=" << connected_;

    PredictionResult result;
    if (!connected_) {
        LOG_ERROR("inprocess_backend") << "[predict] FALHA PRÉ-CONDIÇÃO: não conectado";
        result.error_message = "Not connected";
        return result;
    }

    for (const auto& [k, v] : inputs) {
        LOG_DEBUG("inprocess_backend") << "[predict] input key='" << k << "'";
    }

    LOG_DEBUG("inprocess_backend") << "[predict] chamando engine_->predict('" << model_id << "', inputs)";
    auto r = engine_->predict(model_id, inputs);

    LOG_DEBUG("inprocess_backend") << "[predict] engine_->predict retornou success=" << r.success
         << " inference_time_ms=" << r.inference_time_ms
         << " error_message='" << r.error_message << "'";
    LOG_DEBUG("inprocess_backend") << "[predict] n_outputs=" << r.outputs.size();

    for (const auto& [k, v] : r.outputs) {
        LOG_DEBUG("inprocess_backend") << "[predict] output key='" << k << "'";
    }

    result.success           = r.success;
    result.inference_time_ms = r.inference_time_ms;
    result.error_message     = r.error_message;
    result.outputs           = std::move(r.outputs);

    if (!result.success) {
        LOG_WARN("inprocess_backend") << "[predict] inferência sem sucesso; error_message='" << result.error_message << "'";
    }

    LOG_DEBUG("inprocess_backend") << "[predict] concluído; success=" << result.success
         << " inference_time_ms=" << result.inference_time_ms;
    return result;
}

std::vector<PredictionResult> InProcessBackend::batch_predict(
    const std::string& model_id,
    const std::vector<Object>& batch_inputs) {

    LOG_DEBUG("inprocess_backend") << "[batch_predict] chamado; model_id='" << model_id
         << "' batch_size=" << batch_inputs.size() << " connected_=" << connected_;

    std::vector<PredictionResult> results;
    results.reserve(batch_inputs.size());

    for (size_t i = 0; i < batch_inputs.size(); ++i) {
        LOG_DEBUG("inprocess_backend") << "[batch_predict] executando predict para item[" << i << "]";
        results.push_back(predict(model_id, batch_inputs[i]));
        LOG_DEBUG("inprocess_backend") << "[batch_predict] item[" << i << "] success="
             << results.back().success << " ms=" << results.back().inference_time_ms;
    }

    LOG_DEBUG("inprocess_backend") << "[batch_predict] concluído; n_results=" << results.size();
    return results;
}

// =============================================================================
// Introspecção
// =============================================================================

std::vector<ModelInfo> InProcessBackend::list_models() {
    LOG_DEBUG("inprocess_backend") << "[list_models] chamado; connected_=" << connected_;

    std::vector<ModelInfo> result;
    if (!connected_) {
        LOG_WARN("inprocess_backend") << "[list_models] não conectado, retornando lista vazia";
        return result;
    }

    auto ids = engine_->get_loaded_model_ids();
    LOG_DEBUG("inprocess_backend") << "[list_models] n_loaded_models=" << ids.size();

    for (const auto& id : ids) {
        LOG_DEBUG("inprocess_backend") << "[list_models] obtendo info para model_id='" << id << "'";
        result.push_back(proto_to_model_info(engine_->get_model_info(id)));
    }

    LOG_DEBUG("inprocess_backend") << "[list_models] retornando " << result.size() << " modelos";
    return result;
}

ModelInfo InProcessBackend::get_model_info(const std::string& model_id) {
    LOG_DEBUG("inprocess_backend") << "[get_model_info] chamado; model_id='" << model_id
         << "' connected_=" << connected_;

    if (!connected_) {
        LOG_WARN("inprocess_backend") << "[get_model_info] não conectado, retornando ModelInfo vazio";
        return {};
    }

    LOG_DEBUG("inprocess_backend") << "[get_model_info] chamando engine_->get_model_info('" << model_id << "')";
    auto info = proto_to_model_info(engine_->get_model_info(model_id));
    LOG_DEBUG("inprocess_backend") << "[get_model_info] retornando info para model_id='" << model_id
         << "' backend='" << info.backend << "'";
    return info;
}

ValidationResult InProcessBackend::validate_model(const std::string& path) {
    LOG_DEBUG("inprocess_backend") << "[validate_model] chamado; path='" << path
         << "' connected_=" << connected_;

    ValidationResult result;
    if (!connected_) {
        LOG_ERROR("inprocess_backend") << "[validate_model] FALHA PRÉ-CONDIÇÃO: não conectado";
        result.error_message = "Not connected";
        return result;
    }

    const std::string resolved = resolve_path(path);
    LOG_DEBUG("inprocess_backend") << "[validate_model] path resolvido='" << resolved << "'";

    LOG_DEBUG("inprocess_backend") << "[validate_model] chamando engine_->validate_model('" << resolved << "')";
    auto r = engine_->validate_model(resolved);

    LOG_DEBUG("inprocess_backend") << "[validate_model] resultado: valid=" << r.valid
         << " error_message='" << r.error_message << "'"
         << " n_warnings=" << r.warnings.size()
         << " n_inputs=" << r.inputs.size()
         << " n_outputs=" << r.outputs.size();

    result.valid         = r.valid;
    result.error_message = r.error_message;
    result.warnings      = r.warnings;

    for (const auto& w : r.warnings) {
        LOG_WARN("inprocess_backend") << "[validate_model] warning='" << w << "'";
    }

    switch (r.backend) {
        case common::BACKEND_ONNX:   result.backend = "onnx";    break;
        case common::BACKEND_PYTHON: result.backend = "python";  break;
        default:                      result.backend = "unknown"; break;
    }
    LOG_DEBUG("inprocess_backend") << "[validate_model] backend='" << result.backend << "'";

    for (const auto& s : r.inputs) {
        LOG_DEBUG("inprocess_backend") << "[validate_model] input spec name='" << s.name << "'";
        result.inputs.push_back(tensor_spec_from_data(s));
    }
    for (const auto& s : r.outputs) {
        LOG_DEBUG("inprocess_backend") << "[validate_model] output spec name='" << s.name << "'";
        result.outputs.push_back(tensor_spec_from_data(s));
    }

    if (!result.valid) {
        LOG_WARN("inprocess_backend") << "[validate_model] modelo inválido; error_message='" << result.error_message << "'";
    }

    LOG_DEBUG("inprocess_backend") << "[validate_model] concluído; valid=" << result.valid;
    return result;
}

WarmupResult InProcessBackend::warmup_model(const std::string& model_id,
                                             uint32_t num_runs) {
    LOG_DEBUG("inprocess_backend") << "[warmup_model] chamado; model_id='" << model_id
         << "' num_runs=" << num_runs << " connected_=" << connected_;

    if (!connected_) {
        LOG_ERROR("inprocess_backend") << "[warmup_model] FALHA PRÉ-CONDIÇÃO: não conectado";
        return {};
    }

    LOG_DEBUG("inprocess_backend") << "[warmup_model] chamando engine_->warmup_model('" << model_id << "', " << num_runs << ")";
    auto r = engine_->warmup_model(model_id, num_runs);

    LOG_DEBUG("inprocess_backend") << "[warmup_model] resultado: success=" << r.success
         << " runs_completed=" << r.runs_completed
         << " avg_time_ms=" << r.avg_time_ms
         << " min_time_ms=" << r.min_time_ms
         << " max_time_ms=" << r.max_time_ms
         << " error_message='" << r.error_message << "'";

    if (!r.success) {
        LOG_WARN("inprocess_backend") << "[warmup_model] warmup sem sucesso; error_message='" << r.error_message << "'";
    }

    WarmupResult result;
    result.success        = r.success;
    result.runs_completed = r.runs_completed;
    result.avg_time_ms    = r.avg_time_ms;
    result.min_time_ms    = r.min_time_ms;
    result.max_time_ms    = r.max_time_ms;
    result.error_message  = r.error_message;
    return result;
}

// =============================================================================
// Observabilidade
// =============================================================================

bool InProcessBackend::health_check() {
    LOG_DEBUG("inprocess_backend") << "[health_check] connected_=" << connected_;
    return connected_;
}

WorkerStatus InProcessBackend::get_status() {
    LOG_DEBUG("inprocess_backend") << "[get_status] chamado; connected_=" << connected_;

    WorkerStatus s;
    if (!connected_) {
        LOG_WARN("inprocess_backend") << "[get_status] não conectado, retornando WorkerStatus vazio";
        return s;
    }

    s.worker_id      = "inprocess";
    s.uptime_seconds = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time_).count());

    s.loaded_models      = engine_->get_loaded_model_ids();
    s.supported_backends = engine_->get_engine_info().supported_backends;

    LOG_DEBUG("inprocess_backend") << "[get_status] worker_id='" << s.worker_id
         << "' uptime_seconds=" << s.uptime_seconds
         << " n_loaded_models=" << s.loaded_models.size()
         << " n_supported_backends=" << s.supported_backends.size();

    for (const auto& id : s.loaded_models) {
        LOG_DEBUG("inprocess_backend") << "[get_status] loaded model='" << id << "'";
    }
    for (const auto& b : s.supported_backends) {
        LOG_DEBUG("inprocess_backend") << "[get_status] supported backend='" << b << "'";
    }

    return s;
}

ServerMetrics InProcessBackend::get_metrics() {
    LOG_DEBUG("inprocess_backend") << "[get_metrics] chamado (não implementado, retornando vazio)";
    return {};
}

// =============================================================================
// Descoberta de arquivos
// =============================================================================

std::vector<AvailableModel> InProcessBackend::list_available_models(
    const std::string& directory) {

    LOG_DEBUG("inprocess_backend") << "[list_available_models] chamado; directory='" << directory << "'";

    std::vector<AvailableModel> result;
    namespace fs = std::filesystem;

    std::string dir = directory.empty()
        ? (fs::current_path() / "models").string()
        : directory;

    LOG_DEBUG("inprocess_backend") << "[list_available_models] diretório efetivo='" << dir << "'";

    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        LOG_WARN("inprocess_backend") << "[list_available_models] diretório inválido ou inexistente: '" << dir << "'";
        return result;
    }

    auto loaded_ids = engine_->get_loaded_model_ids();
    LOG_DEBUG("inprocess_backend") << "[list_available_models] n_loaded_ids=" << loaded_ids.size();

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;

        auto ext = entry.path().extension().string();
        LOG_DEBUG("inprocess_backend") << "[list_available_models] examinando arquivo='"
             << entry.path().filename().string() << "' ext='" << ext << "'";

        if (ext != ".onnx" && ext != ".py") {
            LOG_DEBUG("inprocess_backend") << "[list_available_models] extensão ignorada: '" << ext << "'";
            continue;
        }

        AvailableModel am;
        am.filename        = entry.path().filename().string();
        am.path            = entry.path().string();
        am.extension       = ext;
        am.backend         = (ext == ".onnx") ? "onnx" : "python";
        am.file_size_bytes = static_cast<int64_t>(entry.file_size());
        am.is_loaded       = false;

        LOG_DEBUG("inprocess_backend") << "[list_available_models] modelo candidato: filename='" << am.filename
             << "' backend='" << am.backend << "' file_size_bytes=" << am.file_size_bytes;

        for (const auto& id : loaded_ids) {
            const std::string loaded_path = engine_->get_model_info(id).model_path();
            LOG_DEBUG("inprocess_backend") << "[list_available_models] comparando am.path='" << am.path
                 << "' com loaded_path='" << loaded_path << "' (id='" << id << "')";
            if (loaded_path == am.path) {
                am.is_loaded = true;
                am.loaded_as = id;
                LOG_DEBUG("inprocess_backend") << "[list_available_models] modelo '" << am.filename
                     << "' já carregado como id='" << id << "'";
                break;
            }
        }

        LOG_DEBUG("inprocess_backend") << "[list_available_models] adicionando modelo='" << am.filename
             << "' is_loaded=" << am.is_loaded << " loaded_as='" << am.loaded_as << "'";
        result.push_back(std::move(am));
    }

    LOG_DEBUG("inprocess_backend") << "[list_available_models] concluído; n_modelos_encontrados=" << result.size();
    return result;
}

}  // namespace client
}  // namespace mlinference
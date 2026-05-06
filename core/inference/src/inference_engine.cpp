// =============================================================================
// inference_engine.cpp — InferenceEngine implementation
// =============================================================================

#include "inference/inference_engine.hpp"
#include "inference/backend_registry.hpp"
#include "inference/onnx_backend.hpp"
#include "inference/python_backend.hpp"
#include "utils/logger.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

namespace mlinference {
namespace inference {

// =============================================================================
// Constructor / Destructor
// =============================================================================

InferenceEngine::InferenceEngine(bool enable_gpu,
                                 uint32_t gpu_device_id,
                                 uint32_t num_threads)
    : gpu_enabled_(enable_gpu)
    , gpu_device_id_(gpu_device_id)
    , num_threads_(num_threads)
    , start_time_(std::chrono::steady_clock::now()) {

    LOG_DEBUG("inference_engine") << "[ctor] InferenceEngine construído; enable_gpu=" << enable_gpu
         << " gpu_device_id=" << gpu_device_id << " num_threads=" << num_threads;

    auto& registry = BackendRegistry::instance();
    LOG_DEBUG("inference_engine") << "[ctor] BackendRegistry obtido; instance=" << (void*)&registry;

    if (!registry.supports(".onnx")) {
        LOG_DEBUG("inference_engine") << "[ctor] registrando backend '.onnx' (OnnxBackendFactory)";
        registry.register_backend(".onnx",
            std::make_unique<OnnxBackendFactory>(enable_gpu, gpu_device_id, num_threads));
        LOG_DEBUG("inference_engine") << "[ctor] backend '.onnx' registrado";
    } else {
        LOG_DEBUG("inference_engine") << "[ctor] backend '.onnx' já registrado, skip";
    }

    if (!registry.supports(".py")) {
        LOG_DEBUG("inference_engine") << "[ctor] registrando backend '.py' (PythonBackendFactory)";
        registry.register_backend(".py",
            std::make_unique<PythonBackendFactory>());
        LOG_DEBUG("inference_engine") << "[ctor] backend '.py' registrado";
    } else {
        LOG_DEBUG("inference_engine") << "[ctor] backend '.py' já registrado, skip";
    }

    engine_info_.gpu_enabled        = enable_gpu;
    engine_info_.gpu_device_id      = gpu_device_id;
    engine_info_.num_threads        = num_threads;
    engine_info_.supported_backends = registry.registered_backend_names();

    LOG_DEBUG("inference_engine") << "[ctor] n_supported_backends=" << engine_info_.supported_backends.size();
    for (const auto& b : engine_info_.supported_backends) {
        LOG_DEBUG("inference_engine") << "[ctor] supported backend='" << b << "'";
    }

    LOG_INFO("inference_engine") << "[ctor] inicialização concluída; gpu=" << enable_gpu
         << " threads=" << num_threads
         << " n_backends=" << engine_info_.supported_backends.size();
}

InferenceEngine::~InferenceEngine() {
    LOG_DEBUG("inference_engine") << "[dtor] InferenceEngine destruído; n_models=" << models_.size();
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [id, entry] : models_) {
        LOG_DEBUG("inference_engine") << "[dtor] descarregando model_id='" << id << "'";
        entry->backend->unload();
    }
    models_.clear();
    LOG_DEBUG("inference_engine") << "[dtor] todos os modelos descarregados e mapa limpo";
}

// =============================================================================
// Model Lifecycle
// =============================================================================

bool InferenceEngine::load_model(
    const std::string& model_id,
    const std::string& model_path,
    common::BackendType force_backend,
    const std::map<std::string, std::string>& backend_config) {

    LOG_DEBUG("inference_engine") << "[load_model] chamado; model_id='" << model_id
         << "' model_path='" << model_path << "' force_backend=" << static_cast<int>(force_backend)
         << " n_config_entries=" << backend_config.size();

    std::lock_guard<std::mutex> lock(mutex_);
    LOG_DEBUG("inference_engine") << "[load_model] mutex adquirido; n_models_carregados=" << models_.size();

    if (models_.count(model_id)) {
        LOG_WARN("inference_engine") << "[load_model] model_id='" << model_id << "' já carregado";
        return false;
    }

    try {
        auto& registry = BackendRegistry::instance();
        std::unique_ptr<ModelBackend> backend;

        if (force_backend != common::BACKEND_UNKNOWN) {
            LOG_DEBUG("inference_engine") << "[load_model] criando backend forçado; force_backend=" << static_cast<int>(force_backend);
            backend = registry.create_by_type(force_backend);
        } else {
            LOG_DEBUG("inference_engine") << "[load_model] detectando backend por extensão de arquivo: '" << model_path << "'";
            backend = registry.create_for_file(model_path);
        }

        LOG_DEBUG("inference_engine") << "[load_model] backend criado; ptr=" << (void*)backend.get();

        for (const auto& [k, v] : backend_config) {
            LOG_DEBUG("inference_engine") << "[load_model] backend_config['" << k << "']=" << v;
        }

        LOG_DEBUG("inference_engine") << "[load_model] chamando backend->load('" << model_path << "')";
        bool load_ok = backend->load(model_path, backend_config);
        LOG_DEBUG("inference_engine") << "[load_model] backend->load retornou=" << load_ok;

        if (!load_ok) {
            LOG_ERROR("inference_engine") << "[load_model] backend->load() falhou para path='" << model_path << "'";
            return false;
        }

        auto entry = std::make_unique<LoadedModel>();
        entry->model_id = model_id;
        entry->path     = model_path;
        entry->backend  = std::move(backend);

        LOG_DEBUG("inference_engine") << "[load_model] LoadedModel criado; model_id='" << entry->model_id
             << "' path='" << entry->path << "' backend_type=" << static_cast<int>(entry->backend->backend_type());

        models_[model_id] = std::move(entry);

        LOG_INFO("inference_engine") << "[load_model] modelo carregado: model_id='" << model_id
             << "' n_models_total=" << models_.size();
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("inference_engine") << "[load_model] exceção ao carregar model_id='" << model_id
             << "': " << e.what();
        return false;
    }
}

bool InferenceEngine::unload_model(const std::string& model_id) {
    LOG_DEBUG("inference_engine") << "[unload_model] chamado; model_id='" << model_id
         << "' n_models_atual=" << models_.size();

    std::lock_guard<std::mutex> lock(mutex_);
    LOG_DEBUG("inference_engine") << "[unload_model] mutex adquirido";

    auto it = models_.find(model_id);
    if (it == models_.end()) {
        LOG_WARN("inference_engine") << "[unload_model] model_id='" << model_id << "' não encontrado em models_";
        return false;
    }

    LOG_DEBUG("inference_engine") << "[unload_model] model_id='" << model_id
         << "' encontrado; backend_type=" << static_cast<int>(it->second->backend->backend_type())
         << " path='" << it->second->path << "'";

    LOG_DEBUG("inference_engine") << "[unload_model] chamando backend->unload()";
    it->second->backend->unload();
    LOG_DEBUG("inference_engine") << "[unload_model] backend->unload() concluído; removendo de models_";

    models_.erase(it);
    LOG_INFO("inference_engine") << "[unload_model] modelo descarregado: model_id='" << model_id
         << "' n_models_restantes=" << models_.size();
    return true;
}

bool InferenceEngine::is_model_loaded(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    bool loaded = models_.find(model_id) != models_.end();
    LOG_DEBUG("inference_engine") << "[is_model_loaded] model_id='" << model_id << "' loaded=" << loaded;
    return loaded;
}

// =============================================================================
// Inference
// =============================================================================

InferenceResult InferenceEngine::predict(const std::string& model_id,
                                         const client::Object& inputs) {
    LOG_DEBUG("inference_engine") << "[predict] chamado; model_id='" << model_id
         << "' n_inputs=" << inputs.size();

    for (const auto& [k, v] : inputs) {
        LOG_DEBUG("inference_engine") << "[predict] input key='" << k << "'";
    }

    check_auto_unload();

    // Mantém o mutex durante todo o predict — o OnnxBackend e o PythonBackend
    // não são thread-safe por contrato (model_backend.hpp: "Instances are not
    // thread-safe; the InferenceEngine serialises access"). Liberar o lock
    // antes de chamar backend->predict() criava uma race condition: outro
    // thread poderia chamar unload_model() e destruir o backend enquanto
    // predict() ainda o usava, resultando em acesso a ponteiro dangling e UB.
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(model_id);
    if (it == models_.end()) {
        LOG_ERROR("inference_engine") << "[predict] model_id='" << model_id << "' não encontrado em models_";
        return {false, {}, 0.0, "Model not loaded: " + model_id};
    }

    ModelBackend* backend = it->second->backend.get();
    LOG_DEBUG("inference_engine") << "[predict] backend encontrado; ptr=" << (void*)backend
         << " backend_type=" << static_cast<int>(backend->backend_type());

    LOG_DEBUG("inference_engine") << "[predict] chamando backend->predict() com lock mantido";
    auto result = backend->predict(inputs);

    LOG_DEBUG("inference_engine") << "[predict] backend->predict() retornou; success=" << result.success
         << " inference_time_ms=" << result.inference_time_ms
         << " error_message='" << result.error_message << "'"
         << " n_outputs=" << result.outputs.size();

    if (!result.success) {
        LOG_WARN("inference_engine") << "[predict] inferência sem sucesso; model_id='" << model_id
             << "' error_message='" << result.error_message << "'";
    }

    for (const auto& [k, v] : result.outputs) {
        LOG_DEBUG("inference_engine") << "[predict] output key='" << k << "'";
    }

    return result;
}

// =============================================================================
// Introspection
// =============================================================================

std::vector<std::string> InferenceEngine::get_loaded_model_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(models_.size());
    for (const auto& [id, _] : models_) ids.push_back(id);
    LOG_DEBUG("inference_engine") << "[get_loaded_model_ids] n_ids=" << ids.size();
    for (const auto& id : ids) {
        LOG_DEBUG("inference_engine") << "[get_loaded_model_ids] id='" << id << "'";
    }
    return ids;
}

common::ModelInfo InferenceEngine::get_model_info(const std::string& model_id) const {
    LOG_DEBUG("inference_engine") << "[get_model_info] chamado; model_id='" << model_id << "'";

    std::lock_guard<std::mutex> lock(mutex_);

    common::ModelInfo info;

    auto it = models_.find(model_id);
    if (it == models_.end()) {
        LOG_WARN("inference_engine") << "[get_model_info] model_id='" << model_id << "' não encontrado, retornando info vazio";
        return info;
    }

    const auto& entry   = it->second;
    const auto& backend = entry->backend;

    LOG_DEBUG("inference_engine") << "[get_model_info] entry encontrado; path='" << entry->path
         << "' backend_type=" << static_cast<int>(backend->backend_type());

    LOG_DEBUG("inference_engine") << "[get_model_info] chamando backend->get_schema()";
    auto schema = backend->get_schema();
    LOG_DEBUG("inference_engine") << "[get_model_info] schema: description='" << schema.description
         << "' author='" << schema.author
         << "' n_inputs=" << schema.inputs.size()
         << " n_outputs=" << schema.outputs.size()
         << " n_tags=" << schema.tags.size();

    info.set_model_id(model_id);
    info.set_backend(backend->backend_type());
    info.set_description(schema.description);
    info.set_author(schema.author);
    info.set_model_path(entry->path);
    info.set_memory_usage_bytes(backend->memory_usage_bytes());

    LOG_DEBUG("inference_engine") << "[get_model_info] memory_usage_bytes=" << backend->memory_usage_bytes();

    auto load_epoch = std::chrono::duration_cast<std::chrono::seconds>(
        backend->load_time().time_since_epoch()).count();
    info.set_loaded_at_unix(load_epoch);
    LOG_DEBUG("inference_engine") << "[get_model_info] loaded_at_unix=" << load_epoch;

    for (const auto& [k, v] : schema.tags) {
        LOG_DEBUG("inference_engine") << "[get_model_info] tag '" << k << "'='" << v << "'";
        (*info.mutable_tags())[k] = v;
    }

    auto fill_spec = [](const TensorSpecData& spec, common::TensorSpec* ts) {
        ts->set_name(spec.name);
        ts->set_dtype(spec.dtype);
        ts->set_description(spec.description);
        for (int64_t dim : spec.shape) ts->add_shape(dim);
        if (spec.has_constraints) {
            ts->set_min_value(spec.min_value);
            ts->set_max_value(spec.max_value);
        }
        ts->set_structured(spec.structured);
    };

    for (size_t i = 0; i < schema.inputs.size(); ++i) {
        const auto& spec = schema.inputs[i];
        LOG_DEBUG("inference_engine") << "[get_model_info] input[" << i << "] name='" << spec.name
             << "' shape_size=" << spec.shape.size()
             << " structured=" << spec.structured
             << " has_constraints=" << spec.has_constraints;
        fill_spec(spec, info.add_inputs());
    }

    for (size_t i = 0; i < schema.outputs.size(); ++i) {
        const auto& spec = schema.outputs[i];
        LOG_DEBUG("inference_engine") << "[get_model_info] output[" << i << "] name='" << spec.name
             << "' shape_size=" << spec.shape.size()
             << " structured=" << spec.structured;
        fill_spec(spec, info.add_outputs());
    }

    LOG_DEBUG("inference_engine") << "[get_model_info] concluído para model_id='" << model_id << "'";
    return info;
}

const RuntimeMetrics* InferenceEngine::get_model_metrics(
    const std::string& model_id) const {

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(model_id);
    if (it == models_.end()) {
        LOG_WARN("inference_engine") << "[get_model_metrics] model_id='" << model_id << "' não encontrado, retornando nullptr";
        return nullptr;
    }
    const RuntimeMetrics* m = &it->second->backend->metrics();
    LOG_DEBUG("inference_engine") << "[get_model_metrics] model_id='" << model_id
         << "' total_inferences=" << m->total_inferences
         << " failed_inferences=" << m->failed_inferences
         << " avg_time_ms=" << m->avg_time_ms();
    return m;
}

// =============================================================================
// Validation
// =============================================================================

InferenceEngine::ValidationResult InferenceEngine::validate_model(
    const std::string& model_path,
    common::BackendType force_backend) const {

    LOG_DEBUG("inference_engine") << "[validate_model] chamado; model_path='" << model_path
         << "' force_backend=" << static_cast<int>(force_backend);

    auto& registry = BackendRegistry::instance();
    ValidationResult result;

    try {
        std::unique_ptr<ModelBackend> backend;

        if (force_backend != common::BACKEND_UNKNOWN) {
            LOG_DEBUG("inference_engine") << "[validate_model] usando backend forçado=" << static_cast<int>(force_backend);
            backend = registry.create_by_type(force_backend);
            result.backend = force_backend;
        } else {
            result.backend = registry.detect_backend(model_path);
            LOG_DEBUG("inference_engine") << "[validate_model] backend detectado=" << static_cast<int>(result.backend)
                 << " para path='" << model_path << "'";

            if (result.backend == common::BACKEND_UNKNOWN) {
                result.error_message = "No backend found for: " + model_path;
                LOG_ERROR("inference_engine") << "[validate_model] nenhum backend encontrado para '" << model_path << "'";
                return result;
            }
            backend = registry.create_by_type(result.backend);
        }

        LOG_DEBUG("inference_engine") << "[validate_model] backend criado; ptr=" << (void*)backend.get();

        LOG_DEBUG("inference_engine") << "[validate_model] chamando backend->validate('" << model_path << "')";
        std::string err = backend->validate(model_path);
        LOG_DEBUG("inference_engine") << "[validate_model] backend->validate retornou err='" << err << "'";

        if (!err.empty()) {
            result.error_message = err;
            LOG_WARN("inference_engine") << "[validate_model] modelo inválido: " << err;
            return result;
        }

        LOG_DEBUG("inference_engine") << "[validate_model] chamando backend->load() para extração de schema";
        std::map<std::string, std::string> empty_config;
        bool load_ok = backend->load(model_path, empty_config);
        LOG_DEBUG("inference_engine") << "[validate_model] backend->load() retornou=" << load_ok;

        if (load_ok) {
            auto schema = backend->get_schema();
            LOG_DEBUG("inference_engine") << "[validate_model] schema extraído: n_inputs=" << schema.inputs.size()
                 << " n_outputs=" << schema.outputs.size();

            for (size_t i = 0; i < schema.inputs.size(); ++i) {
                LOG_DEBUG("inference_engine") << "[validate_model] input[" << i << "] name='" << schema.inputs[i].name
                     << "' shape_size=" << schema.inputs[i].shape.size();
            }
            for (size_t i = 0; i < schema.outputs.size(); ++i) {
                LOG_DEBUG("inference_engine") << "[validate_model] output[" << i << "] name='" << schema.outputs[i].name
                     << "' shape_size=" << schema.outputs[i].shape.size();
            }

            result.inputs  = schema.inputs;
            result.outputs = schema.outputs;

            LOG_DEBUG("inference_engine") << "[validate_model] chamando backend->unload()";
            backend->unload();
            LOG_DEBUG("inference_engine") << "[validate_model] backend->unload() concluído";
        } else {
            LOG_WARN("inference_engine") << "[validate_model] backend->load() falhou; schema não extraído para '" << model_path << "'";
        }

        result.valid = true;
        LOG_DEBUG("inference_engine") << "[validate_model] SUCESSO; valid=true para '" << model_path << "'";
        return result;

    } catch (const std::exception& e) {
        result.error_message = e.what();
        LOG_ERROR("inference_engine") << "[validate_model] exceção: " << e.what();
        return result;
    }
}

// =============================================================================
// Warmup
// =============================================================================

InferenceEngine::WarmupResult InferenceEngine::warmup_model(
    const std::string& model_id, uint32_t num_runs) {

    LOG_DEBUG("inference_engine") << "[warmup_model] chamado; model_id='" << model_id
         << "' num_runs=" << num_runs;

    WarmupResult result;

    ModelBackend* backend = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = models_.find(model_id);
        if (it == models_.end()) {
            result.error_message = "Model not loaded: " + model_id;
            LOG_ERROR("inference_engine") << "[warmup_model] model_id='" << model_id << "' não encontrado";
            return result;
        }
        backend = it->second->backend.get();
        LOG_DEBUG("inference_engine") << "[warmup_model] backend encontrado; ptr=" << (void*)backend
             << " backend_type=" << static_cast<int>(backend->backend_type());
    }

    if (num_runs == 0) {
        LOG_DEBUG("inference_engine") << "[warmup_model] num_runs=0 → ajustado para 5";
        num_runs = 5;
    }

    LOG_DEBUG("inference_engine") << "[warmup_model] chamando backend->get_schema()";
    auto schema = backend->get_schema();
    LOG_DEBUG("inference_engine") << "[warmup_model] schema: n_inputs=" << schema.inputs.size()
         << " n_outputs=" << schema.outputs.size();

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    client::Object dummy;
    for (const auto& spec : schema.inputs) {
        if (spec.structured) {
            LOG_DEBUG("inference_engine") << "[warmup_model] input '" << spec.name
                 << "' structured=true → Object vazio";
            dummy[spec.name] = client::Value{client::Object{}};
        } else {
            int64_t total = 1;
            for (int64_t dim : spec.shape) total *= (dim == -1) ? 1 : dim;

            LOG_DEBUG("inference_engine") << "[warmup_model] input '" << spec.name
                 << "' structured=false shape_size=" << spec.shape.size()
                 << " total_elements=" << total;

            client::Array arr;
            arr.reserve(static_cast<size_t>(total));
            for (int64_t i = 0; i < total; ++i)
                arr.push_back(client::Value{static_cast<double>(dist(rng))});

            dummy[spec.name] = client::Value{std::move(arr)};
            LOG_DEBUG("inference_engine") << "[warmup_model] dummy input '" << spec.name
                 << "' preenchido com " << total << " valores aleatórios";
        }
    }

    LOG_DEBUG("inference_engine") << "[warmup_model] iniciando " << num_runs << " runs de warmup";

    double total_ms = 0.0;
    result.min_time_ms = std::numeric_limits<double>::max();
    result.max_time_ms = 0.0;

    for (uint32_t i = 0; i < num_runs; ++i) {
        LOG_DEBUG("inference_engine") << "[warmup_model] run[" << i << "] iniciando predict()";
        auto r = backend->predict(dummy);
        LOG_DEBUG("inference_engine") << "[warmup_model] run[" << i << "] success=" << r.success
             << " inference_time_ms=" << r.inference_time_ms
             << " error_message='" << r.error_message << "'";

        if (!r.success) {
            result.error_message = "Warmup inference failed at run "
                                   + std::to_string(i) + ": " + r.error_message;
            result.runs_completed = i;
            LOG_ERROR("inference_engine") << "[warmup_model] falha no run[" << i << "]: " << result.error_message;
            return result;
        }

        total_ms += r.inference_time_ms;
        if (r.inference_time_ms < result.min_time_ms) result.min_time_ms = r.inference_time_ms;
        if (r.inference_time_ms > result.max_time_ms) result.max_time_ms = r.inference_time_ms;
        result.runs_completed++;

        LOG_DEBUG("inference_engine") << "[warmup_model] run[" << i << "] acumulado: total_ms=" << total_ms
             << " min=" << result.min_time_ms << " max=" << result.max_time_ms
             << " runs_completed=" << result.runs_completed;
    }

    result.success     = true;
    result.avg_time_ms = total_ms / num_runs;

    LOG_INFO("inference_engine") << "[warmup_model] warmup concluído: model_id='" << model_id
         << "' runs_completed=" << result.runs_completed
         << " avg_time_ms=" << result.avg_time_ms
         << " min_time_ms=" << result.min_time_ms
         << " max_time_ms=" << result.max_time_ms;

    return result;
}

// =============================================================================
// Auto-unload
// =============================================================================

void InferenceEngine::check_auto_unload() {
    LOG_DEBUG("inference_engine") << "[check_auto_unload] chamado; n_models=" << models_.size() << " (placeholder)";
    // Placeholder — auto-unload logic can be added here if needed.
}

}  // namespace inference
}  // namespace mlinference
#include "worker/inference_engine.hpp"
#include "worker/backend_registry.hpp"
#include "worker/onnx_backend.hpp"
#include "worker/python_backend.hpp"
#include <iostream>
#include <algorithm>

namespace mlinference {
namespace worker {

// ============================================
// Construction
// ============================================

InferenceEngine::InferenceEngine(bool enable_gpu,
                                 uint32_t gpu_device_id,
                                 uint32_t num_threads)
    : enable_gpu_(enable_gpu)
    , gpu_device_id_(gpu_device_id)
    , num_threads_(num_threads) {
    
    // Register all known backends on first construction.
    auto& registry = BackendRegistry::instance();
    
    if (!registry.supports(".onnx")) {
        registry.register_backend(".onnx",
            std::make_unique<OnnxBackendFactory>(enable_gpu, gpu_device_id, num_threads));
    }
    
    if (!registry.supports(".py")) {
        registry.register_backend(".py",
            std::make_unique<PythonBackendFactory>());
    }
    
    std::cout << "[InferenceEngine] Initialized"
              << " | GPU: " << (enable_gpu ? "yes" : "no")
              << " | Threads: " << num_threads
              << " | Backends:";
    for (const auto& name : registry.registered_backend_names()) {
        std::cout << " " << name;
    }
    std::cout << std::endl;
}

InferenceEngine::~InferenceEngine() {
    std::lock_guard<std::mutex> lock(mutex_);
    models_.clear();
    std::cout << "[InferenceEngine] Destroyed" << std::endl;
}

// ============================================
// Model Lifecycle
// ============================================

bool InferenceEngine::load_model(
    const std::string& model_id,
    const std::string& model_path,
    common::BackendType force_backend,
    const std::map<std::string, std::string>& backend_config) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (models_.find(model_id) != models_.end()) {
        std::cerr << "[InferenceEngine] Model already loaded: " << model_id << std::endl;
        return false;
    }
    
    auto& registry = BackendRegistry::instance();
    
    try {
        std::unique_ptr<ModelBackend> backend;
        
        if (force_backend != common::BACKEND_UNKNOWN) {
            backend = registry.create_by_type(force_backend);
        } else {
            backend = registry.create_for_file(model_path);
        }
        
        if (!backend->load(model_path, backend_config)) {
            std::cerr << "[InferenceEngine] Backend failed to load: " << model_path << std::endl;
            return false;
        }
        
        auto entry = std::make_unique<LoadedModel>();
        entry->model_id = model_id;
        entry->path = model_path;
        entry->backend = std::move(backend);
        
        models_[model_id] = std::move(entry);
        
        std::cout << "[InferenceEngine] Model loaded: " << model_id
                  << " (backend: " << models_[model_id]->backend->backend_type() << ")"
                  << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] Error loading " << model_id
                  << ": " << e.what() << std::endl;
        return false;
    }
}

bool InferenceEngine::unload_model(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = models_.find(model_id);
    if (it == models_.end()) {
        std::cerr << "[InferenceEngine] Model not found: " << model_id << std::endl;
        return false;
    }
    
    it->second->backend->unload();
    models_.erase(it);
    std::cout << "[InferenceEngine] Model unloaded: " << model_id << std::endl;
    return true;
}

bool InferenceEngine::is_model_loaded(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return models_.find(model_id) != models_.end();
}

// ============================================
// Inference
// ============================================

InferenceResult InferenceEngine::predict(
    const std::string& model_id,
    const std::map<std::string, std::vector<float>>& inputs) {
    
    check_auto_unload();
    
    ModelBackend* backend = nullptr;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = models_.find(model_id);
        if (it == models_.end()) {
            return {false, {}, 0.0, "Model not loaded: " + model_id};
        }
        backend = it->second->backend.get();
    }
    
    // Run inference outside the lock (backends are single-model, engine serialises).
    return backend->predict(inputs);
}

// ============================================
// Introspection
// ============================================

std::vector<std::string> InferenceEngine::get_loaded_model_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(models_.size());
    for (const auto& [id, _] : models_) ids.push_back(id);
    return ids;
}

common::ModelInfo InferenceEngine::get_model_info(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    common::ModelInfo info;
    
    auto it = models_.find(model_id);
    if (it == models_.end()) return info;
    
    const auto& entry = it->second;
    const auto& backend = entry->backend;
    auto schema = backend->get_schema();
    
    info.set_model_id(model_id);
    info.set_version(entry->version);
    info.set_backend(backend->backend_type());
    info.set_description(schema.description);
    info.set_author(schema.author);
    info.set_model_path(entry->path);
    info.set_memory_usage_bytes(backend->memory_usage_bytes());
    
    // Load timestamp
    auto load_epoch = std::chrono::duration_cast<std::chrono::seconds>(
        backend->load_time().time_since_epoch()).count();
    info.set_loaded_at_unix(load_epoch);
    
    // Tags
    for (const auto& [k, v] : schema.tags) {
        (*info.mutable_tags())[k] = v;
    }
    
    // Inputs
    for (const auto& spec : schema.inputs) {
        auto* ts = info.add_inputs();
        ts->set_name(spec.name);
        ts->set_dtype(spec.dtype);
        ts->set_description(spec.description);
        for (int64_t dim : spec.shape) ts->add_shape(dim);
        if (spec.has_constraints) {
            ts->set_min_value(spec.min_value);
            ts->set_max_value(spec.max_value);
        }
    }
    
    // Outputs
    for (const auto& spec : schema.outputs) {
        auto* ts = info.add_outputs();
        ts->set_name(spec.name);
        ts->set_dtype(spec.dtype);
        ts->set_description(spec.description);
        for (int64_t dim : spec.shape) ts->add_shape(dim);
    }
    
    return info;
}

const RuntimeMetrics* InferenceEngine::get_model_metrics(
    const std::string& model_id) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(model_id);
    if (it == models_.end()) return nullptr;
    return &it->second->backend->metrics();
}

// ============================================
// Validation
// ============================================

InferenceEngine::ValidationResult InferenceEngine::validate_model(
    const std::string& model_path,
    common::BackendType force_backend) const {
    
    auto& registry = BackendRegistry::instance();
    ValidationResult result;
    
    try {
        std::unique_ptr<ModelBackend> backend;
        
        if (force_backend != common::BACKEND_UNKNOWN) {
            backend = registry.create_by_type(force_backend);
            result.backend = force_backend;
        } else {
            result.backend = registry.detect_backend(model_path);
            if (result.backend == common::BACKEND_UNKNOWN) {
                result.error_message = "No backend found for: " + model_path;
                return result;
            }
            backend = registry.create_for_file(model_path);
        }
        
        std::string err = backend->validate(model_path);
        if (!err.empty()) {
            result.error_message = err;
            return result;
        }
        
        // Try a lightweight load to extract schema
        std::map<std::string, std::string> empty_config;
        if (backend->load(model_path, empty_config)) {
            result.schema = backend->get_schema();
            backend->unload();
        }
        
        result.valid = true;
        return result;
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
        return result;
    }
}

// ============================================
// Warmup
// ============================================

InferenceEngine::WarmupResult InferenceEngine::warmup_model(
    const std::string& model_id, uint32_t num_runs) {
    
    WarmupResult result;
    
    ModelBackend* backend = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = models_.find(model_id);
        if (it == models_.end()) {
            result.error_message = "Model not loaded: " + model_id;
            return result;
        }
        backend = it->second->backend.get();
    }
    
    if (num_runs == 0) num_runs = 5;
    
    auto schema = backend->get_schema();
    
    // Build dummy inputs
    std::map<std::string, std::vector<float>> dummy;
    for (const auto& spec : schema.inputs) {
        int64_t total = 1;
        for (int64_t dim : spec.shape) total *= (dim == -1) ? 1 : dim;
        dummy[spec.name] = std::vector<float>(static_cast<size_t>(total), 0.5f);
    }
    
    double total_ms = 0.0;
    result.min_time_ms = std::numeric_limits<double>::max();
    result.max_time_ms = 0.0;
    
    for (uint32_t i = 0; i < num_runs; ++i) {
        auto r = backend->predict(dummy);
        if (!r.success) {
            result.error_message = "Warmup inference failed at run "
                                   + std::to_string(i) + ": " + r.error_message;
            result.runs_completed = i;
            return result;
        }
        total_ms += r.inference_time_ms;
        if (r.inference_time_ms < result.min_time_ms) result.min_time_ms = r.inference_time_ms;
        if (r.inference_time_ms > result.max_time_ms) result.max_time_ms = r.inference_time_ms;
        result.runs_completed++;
    }
    
    result.success = true;
    result.avg_time_ms = total_ms / num_runs;
    return result;
}

// ============================================
// Engine Info
// ============================================

InferenceEngine::EngineInfo InferenceEngine::get_engine_info() const {
    EngineInfo info;
    info.gpu_enabled = enable_gpu_;
    info.num_threads = num_threads_;
    info.supported_backends = BackendRegistry::instance().registered_backend_names();
    return info;
}

// ============================================
// Auto-unload
// ============================================

void InferenceEngine::enable_auto_unload(uint32_t timeout) {
    auto_unload_enabled_ = true;
    idle_timeout_seconds_ = timeout;
}

void InferenceEngine::disable_auto_unload() {
    auto_unload_enabled_ = false;
}

void InferenceEngine::check_auto_unload() {
    if (!auto_unload_enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    
    std::vector<std::string> to_remove;
    for (const auto& [id, entry] : models_) {
        auto idle = std::chrono::duration_cast<std::chrono::seconds>(
            now - entry->backend->last_used());
        if (idle.count() > static_cast<int64_t>(idle_timeout_seconds_)) {
            to_remove.push_back(id);
        }
    }
    
    for (const auto& id : to_remove) {
        std::cout << "[InferenceEngine] Auto-unloading idle model: " << id << std::endl;
        models_[id]->backend->unload();
        models_.erase(id);
    }
}

}  // namespace worker
}  // namespace mlinference
// =============================================================================
// inference_client.cpp — Classe pública InferenceClient
//
// Detecta o modo (gRPC ou in-process) e delega para o backend correto.
// Toda a lógica de transporte vive em grpc_client_backend.cpp e
// inprocess_backend.cpp.
// =============================================================================

#include "client/inference_client.hpp"
#include "client/grpc_client_backend.hpp"
#include "client/inprocess_backend.hpp"

namespace miia {
namespace client {

static bool is_inprocess(const std::string& target) {
    return target == "inprocess" || target == "in_process" || target == "local";
}

InferenceClient::InferenceClient(const std::string& target) {
    if (is_inprocess(target))
        backend_ = std::make_unique<InProcessBackend>(target);
    else
        backend_ = std::make_unique<GrpcClientBackend>(target);
}

InferenceClient::~InferenceClient() = default;

bool InferenceClient::connect()           { return backend_->connect();      }
bool InferenceClient::is_connected() const { return backend_->is_connected(); }

bool InferenceClient::load_model(const std::string& id,
                                  const std::string& path,
                                  const std::string& version) {
    return backend_->load_model(id, path, version);
}

bool InferenceClient::unload_model(const std::string& id) {
    return backend_->unload_model(id);
}

PredictionResult InferenceClient::predict(const std::string& id,
                                           const Object& inputs) {
    return backend_->predict(id, inputs);
}

std::vector<PredictionResult> InferenceClient::batch_predict(
    const std::string& id,
    const std::vector<Object>& batch) {
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

bool          InferenceClient::health_check() { return backend_->health_check(); }
WorkerStatus  InferenceClient::get_status()   { return backend_->get_status();   }
ServerMetrics InferenceClient::get_metrics()  { return backend_->get_metrics();  }

std::vector<AvailableModel> InferenceClient::list_available_models(
    const std::string& directory) {
    return backend_->list_available_models(directory);
}

}  // namespace client
}  // namespace miia

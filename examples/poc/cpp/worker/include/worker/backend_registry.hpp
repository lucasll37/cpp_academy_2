#ifndef ML_INFERENCE_BACKEND_REGISTRY_HPP
#define ML_INFERENCE_BACKEND_REGISTRY_HPP

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>
#include "model_backend.hpp"

namespace mlinference {
namespace worker {

/// Singleton registry that maps file extensions to BackendFactory instances.
/// 
/// Usage:
///   BackendRegistry::instance().register_backend(".onnx", std::make_unique<OnnxBackendFactory>(...));
///   auto backend = BackendRegistry::instance().create_for_file("model.onnx");
///
/// To add a new backend in the future:
///   1. Implement ModelBackend + BackendFactory for your format.
///   2. Register it: registry.register_backend(".myext", std::make_unique<MyBackendFactory>());
///   That's it. No other code changes required.
class BackendRegistry {
public:
    static BackendRegistry& instance() {
        static BackendRegistry registry;
        return registry;
    }
    
    /// Register a factory for the given file extension (e.g. ".onnx", ".py").
    void register_backend(const std::string& extension,
                          std::unique_ptr<BackendFactory> factory) {
        factories_[extension] = std::move(factory);
    }
    
    /// Create a backend for the given file path, auto-detecting by extension.
    /// Throws std::runtime_error if no backend matches.
    std::unique_ptr<ModelBackend> create_for_file(const std::string& path) const {
        std::string ext = get_extension(path);
        auto it = factories_.find(ext);
        if (it == factories_.end()) {
            throw std::runtime_error("No backend registered for extension: " + ext);
        }
        return it->second->create();
    }
    
    /// Create a backend by explicit BackendType enum.
    /// Throws std::runtime_error if no backend matches.
    std::unique_ptr<ModelBackend> create_by_type(common::BackendType type) const {
        for (const auto& [ext, factory] : factories_) {
            if (factory->backend_type() == type) {
                return factory->create();
            }
        }
        throw std::runtime_error("No backend registered for type: " + std::to_string(type));
    }
    
    /// Detect which backend would handle a given path.
    common::BackendType detect_backend(const std::string& path) const {
        std::string ext = get_extension(path);
        auto it = factories_.find(ext);
        if (it == factories_.end()) {
            return common::BACKEND_UNKNOWN;
        }
        return it->second->backend_type();
    }
    
    /// List all registered extensions.
    std::vector<std::string> registered_extensions() const {
        std::vector<std::string> exts;
        exts.reserve(factories_.size());
        for (const auto& [ext, _] : factories_) {
            exts.push_back(ext);
        }
        return exts;
    }
    
    /// List all registered backend names.
    std::vector<std::string> registered_backend_names() const {
        std::vector<std::string> names;
        names.reserve(factories_.size());
        for (const auto& [_, factory] : factories_) {
            names.push_back(factory->name());
        }
        return names;
    }
    
    /// Check if an extension is supported.
    bool supports(const std::string& extension) const {
        return factories_.find(extension) != factories_.end();
    }

private:
    BackendRegistry() = default;
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;
    
    std::map<std::string, std::unique_ptr<BackendFactory>> factories_;
    
    static std::string get_extension(const std::string& path) {
        auto pos = path.rfind('.');
        if (pos == std::string::npos) return "";
        return path.substr(pos);
    }
};

}  // namespace worker
}  // namespace mlinference

#endif  // ML_INFERENCE_BACKEND_REGISTRY_HPP
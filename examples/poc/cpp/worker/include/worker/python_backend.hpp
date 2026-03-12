#ifndef ML_INFERENCE_PYTHON_BACKEND_HPP
#define ML_INFERENCE_PYTHON_BACKEND_HPP

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include "model_backend.hpp"

// Forward-declare Python types so we don't leak Python.h into every TU.
// The actual #include <Python.h> lives only in python_backend.cpp.
struct _object;
typedef _object PyObject;

namespace mlinference {
namespace worker {

/// Python backend — loads a ``.py`` file, finds a :class:`MiiaModel`
/// subclass, instantiates it, and forwards ``predict()`` calls through
/// the embedded CPython interpreter.
///
/// Thread safety: the GIL is acquired around every Python call.
/// The caller (InferenceEngine) already serialises access, but the
/// GIL acquisition is kept as a safety net.
///
/// Lifecycle:
///   1. ``load()`` — initialise interpreter (if first backend),
///      import module, find & instantiate MiiaModel subclass, call
///      ``model.load()``, cache schema.
///   2. ``predict()`` — convert C++ float maps → numpy arrays,
///      call ``model.predict()``, convert numpy arrays → C++ float maps.
///   3. ``unload()`` — release Python object references.
///
/// Requirements:
///   - Python 3.10+ with numpy installed in the environment.
///   - ``miia_model.py`` must be importable (placed in PYTHONPATH or
///     in the same directory as the ``.py`` model).
class PythonBackend : public ModelBackend {
public:
    PythonBackend();
    ~PythonBackend() override;

    // --- ModelBackend interface ---

    bool load(const std::string& path,
              const std::map<std::string, std::string>& config) override;

    void unload() override;

    InferenceResult predict(
        const std::map<std::string, std::vector<float>>& inputs) override;

    ModelSchema get_schema() const override;

    common::BackendType backend_type() const override {
        return common::BACKEND_PYTHON;
    }

    int64_t memory_usage_bytes() const override;

    std::string validate(const std::string& path) const override;

    void warmup(uint32_t n) override;

private:
    // Global interpreter management (ref-counted across instances).
    static std::mutex init_mutex_;
    static int instance_count_;
    static void ensure_interpreter();
    static void release_interpreter();

    // Python object handles (prevent Python.h leak).
    PyObject* py_model_instance_ = nullptr;   // The MiiaModel subclass instance
    PyObject* py_predict_method_ = nullptr;    // Bound method: model.predict
    PyObject* py_schema_method_  = nullptr;    // Bound method: model.get_schema

    // Cached schema (extracted once on load, avoid repeated Python calls).
    ModelSchema cached_schema_;
    std::string model_dir_;    // Directory containing the .py file
    std::string module_name_;  // Module name (filename without .py)

    // --- Helpers ---

    /// Import the module and find the first MiiaModel subclass.
    /// Returns new-ref to the class, or nullptr on failure.
    PyObject* find_model_class(const std::string& path);

    /// Convert C++ map<string, vector<float>> → Python dict[str, np.ndarray].
    PyObject* inputs_to_py_dict(
        const std::map<std::string, std::vector<float>>& inputs) const;

    /// Convert Python dict[str, np.ndarray] → C++ map<string, vector<float>>.
    bool py_dict_to_outputs(
        PyObject* py_dict,
        std::map<std::string, std::vector<float>>& outputs,
        std::string& error) const;

    /// Call model.get_schema() and parse the returned ModelSchema dataclass.
    ModelSchema extract_schema_from_python() const;

    /// Parse a Python TensorSpec dataclass into C++ TensorSpecData.
    TensorSpecData parse_tensor_spec(PyObject* py_spec) const;
};

// ============================================
// Factory
// ============================================

class PythonBackendFactory : public BackendFactory {
public:
    std::unique_ptr<ModelBackend> create() const override {
        return std::make_unique<PythonBackend>();
    }

    common::BackendType backend_type() const override {
        return common::BACKEND_PYTHON;
    }

    std::string name() const override { return "python"; }
};

}  // namespace worker
}  // namespace mlinference

#endif  // ML_INFERENCE_PYTHON_BACKEND_HPP
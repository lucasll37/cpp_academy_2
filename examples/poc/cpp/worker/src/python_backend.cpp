#include "worker/python_backend.hpp"

// Python.h must come before any standard headers to avoid redefinition warnings.
// On some platforms (POSIX), Python.h redefines _POSIX_C_SOURCE.
#include <Python.h>

// numpy C-API.  We use the "import_array" style; PY_ARRAY_UNIQUE_SYMBOL
// prevents symbol collisions if numpy is initialised elsewhere.
#define PY_ARRAY_UNIQUE_SYMBOL MIIA_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

namespace mlinference {
namespace worker {

// ============================================
// Static members
// ============================================

std::mutex PythonBackend::init_mutex_;
int PythonBackend::instance_count_ = 0;

// ============================================
// Interpreter management
// ============================================

// Track whether numpy was already initialised (separate from interpreter)
static bool numpy_initialised_ = false;

void PythonBackend::ensure_interpreter() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (!Py_IsInitialized()) {
        Py_Initialize();
        numpy_initialised_ = false;
    }

    if (!numpy_initialised_) {
        // sys.path injection...
        PyRun_SimpleString(
            "import sys, os, glob\n"
            "for base in [os.getcwd(), os.path.dirname(os.getcwd())]:\n"
            "    venv = os.path.join(base, 'python', '.venv')\n"
            "    if os.path.isdir(venv):\n"
            "        for sp in glob.glob(os.path.join(venv, 'lib', 'python*', 'site-packages')):\n"
            "            if sp not in sys.path:\n"
            "                sys.path.insert(0, sp)\n"
            "        py_dir = os.path.join(base, 'python')\n"
            "        if py_dir not in sys.path:\n"
            "            sys.path.insert(0, py_dir)\n"
            "        break\n"
        );

        if (_import_array() < 0) {
            std::cerr << "[PythonBackend] WARNING: numpy import_array failed." << std::endl;
            PyErr_Print();
        } else {
            numpy_initialised_ = true;
        }

        std::cout << "[PythonBackend] Python interpreter initialised ("
                  << Py_GetVersion() << ")" << std::endl;

        // CRÍTICO: libera o GIL da thread principal para que outras threads
        // possam adquiri-lo via PyGILState_Ensure()
        PyEval_SaveThread();
    }

    instance_count_++;
}

void PythonBackend::release_interpreter() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    instance_count_--;
    // We intentionally do NOT call Py_Finalize() because:
    // 1. It's not safe to re-initialise after finalise in many builds.
    // 2. Multiple backends may share the interpreter.
    // The interpreter lives for the lifetime of the process.
}

// ============================================
// Construction / Destruction
// ============================================

PythonBackend::PythonBackend() = default;

PythonBackend::~PythonBackend() {
    if (loaded_) unload();
}

// ============================================
// Lifecycle: load
// ============================================

bool PythonBackend::load(const std::string& path,
                         const std::map<std::string, std::string>& config) {
    if (loaded_) {
        std::cerr << "[PythonBackend] Already loaded. Unload first." << std::endl;
        return false;
    }

    ensure_interpreter();
    PyGILState_STATE gstate = PyGILState_Ensure();

    bool success = false;

    try {
        // 1. Resolve paths
        fs::path file_path = fs::absolute(path);
        if (!fs::exists(file_path)) {
            std::cerr << "[PythonBackend] File not found: " << path << std::endl;
            PyGILState_Release(gstate);
            release_interpreter();
            return false;
        }

        model_dir_ = file_path.parent_path().string();
        module_name_ = file_path.stem().string();

        // 2. Add model directory to sys.path
        PyObject* sys_path = PySys_GetObject("path");  // borrowed ref
        PyObject* py_dir = PyUnicode_FromString(model_dir_.c_str());
        if (PySequence_Contains(sys_path, py_dir) == 0) {
            PyList_Insert(sys_path, 0, py_dir);
        }
        Py_DECREF(py_dir);

        // Also add the directory containing miia_model.py if specified.
        auto it = config.find("sdk_path");
        if (it != config.end()) {
            PyObject* sdk_dir = PyUnicode_FromString(it->second.c_str());
            if (PySequence_Contains(sys_path, sdk_dir) == 0) {
                PyList_Insert(sys_path, 0, sdk_dir);
            }
            Py_DECREF(sdk_dir);
        }

        // 3. Find and instantiate the MiiaModel subclass
        PyObject* py_class = find_model_class(path);
        if (!py_class) {
            std::cerr << "[PythonBackend] No MiiaModel subclass found in: "
                      << path << std::endl;
            PyErr_Print();
            PyGILState_Release(gstate);
            release_interpreter();
            return false;
        }

        // 4. Instantiate: model = MyModelClass()
        py_model_instance_ = PyObject_CallNoArgs(py_class);
        Py_DECREF(py_class);

        if (!py_model_instance_) {
            std::cerr << "[PythonBackend] Failed to instantiate model class."
                      << std::endl;
            PyErr_Print();
            PyGILState_Release(gstate);
            release_interpreter();
            return false;
        }

        // 5. Call model.load()
        PyObject* load_result = PyObject_CallMethod(
            py_model_instance_, "load", nullptr);
        if (!load_result) {
            std::cerr << "[PythonBackend] model.load() raised an exception."
                      << std::endl;
            PyErr_Print();
            Py_CLEAR(py_model_instance_);
            PyGILState_Release(gstate);
            release_interpreter();
            return false;
        }
        Py_DECREF(load_result);

        // 6. Cache bound methods for fast invocation
        py_predict_method_ = PyObject_GetAttrString(py_model_instance_, "predict");
        py_schema_method_  = PyObject_GetAttrString(py_model_instance_, "get_schema");

        if (!py_predict_method_ || !py_schema_method_) {
            std::cerr << "[PythonBackend] Model is missing predict or get_schema."
                      << std::endl;
            PyErr_Print();
            Py_CLEAR(py_predict_method_);
            Py_CLEAR(py_schema_method_);
            Py_CLEAR(py_model_instance_);
            PyGILState_Release(gstate);
            release_interpreter();
            return false;
        }

        // 7. Cache schema
        cached_schema_ = extract_schema_from_python();

        loaded_ = true;
        load_time_ = std::chrono::steady_clock::now();
        last_used_ = load_time_;
        success = true;

        std::cout << "[PythonBackend] Loaded: " << path << std::endl;
        std::cout << "  Module: " << module_name_ << std::endl;
        // std::cout << "  Inputs (" << cached_schema_.inputs.size() << "):" << std::endl;
        // for (const auto& s : cached_schema_.inputs) {
        //     std::cout << "    " << s.name << " [";
        //     for (size_t j = 0; j < s.shape.size(); ++j) {
        //         if (j > 0) std::cout << ", ";
        //         std::cout << s.shape[j];
        //     }
        //     std::cout << "]" << std::endl;
        // }
        // std::cout << "  Outputs (" << cached_schema_.outputs.size() << "):" << std::endl;
        // for (const auto& s : cached_schema_.outputs) {
        //     std::cout << "    " << s.name << " [";
        //     for (size_t j = 0; j < s.shape.size(); ++j) {
        //         if (j > 0) std::cout << ", ";
        //         std::cout << s.shape[j];
        //     }
        //     std::cout << "]" << std::endl;
        // }

    } catch (const std::exception& e) {
        std::cerr << "[PythonBackend] C++ exception during load: "
                  << e.what() << std::endl;
    }

    PyGILState_Release(gstate);
    if (!success) release_interpreter();
    return success;
}

// ============================================
// Lifecycle: unload
// ============================================

void PythonBackend::unload() {
    if (!loaded_) return;

    PyGILState_STATE gstate = PyGILState_Ensure();

    // Call model.unload() if it exists
    if (py_model_instance_) {
        PyObject* result = PyObject_CallMethod(
            py_model_instance_, "unload", nullptr);
        if (result) {
            Py_DECREF(result);
        } else {
            PyErr_Clear();  // unload() is optional; ignore errors
        }
    }

    Py_CLEAR(py_predict_method_);
    Py_CLEAR(py_schema_method_);
    Py_CLEAR(py_model_instance_);

    PyGILState_Release(gstate);
    release_interpreter();

    loaded_ = false;
    cached_schema_ = {};
    std::cout << "[PythonBackend] Unloaded: " << module_name_ << std::endl;
}

// ============================================
// Inference
// ============================================

InferenceResult PythonBackend::predict(
    const std::map<std::string, std::vector<float>>& inputs) {

    if (!loaded_ || !py_predict_method_) {
        return {false, {}, 0.0, "Model not loaded"};
    }

    auto start = std::chrono::high_resolution_clock::now();

    PyGILState_STATE gstate = PyGILState_Ensure();
    InferenceResult result;

    // 1. Convert inputs to Python dict of numpy arrays
    PyObject* py_inputs = inputs_to_py_dict(inputs);
    if (!py_inputs) {
        std::string err = "Failed to convert inputs to numpy";
        if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
        PyGILState_Release(gstate);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics_.record(ms, false);
        return {false, {}, ms, err};
    }

    // 2. Call model.predict(inputs)
    PyObject* py_args = PyTuple_Pack(1, py_inputs);
    PyObject* py_result = PyObject_CallObject(py_predict_method_, py_args);
    Py_DECREF(py_args);
    Py_DECREF(py_inputs);

    if (!py_result) {
        std::string err = "model.predict() raised an exception";
        if (PyErr_Occurred()) { PyErr_Print(); PyErr_Clear(); }
        PyGILState_Release(gstate);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics_.record(ms, false);
        return {false, {}, ms, err};
    }

    // 3. Convert Python dict outputs → C++ map
    std::string convert_err;
    bool ok = py_dict_to_outputs(py_result, result.outputs, convert_err);
    Py_DECREF(py_result);

    PyGILState_Release(gstate);

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    result.success = ok;
    result.inference_time_ms = ms;
    if (!ok) result.error_message = convert_err;

    touch();
    metrics_.record(ms, ok);

    return result;
}

// ============================================
// Introspection
// ============================================

ModelSchema PythonBackend::get_schema() const {
    return cached_schema_;
}

int64_t PythonBackend::memory_usage_bytes() const {
    if (!loaded_ || !py_model_instance_) return 0;

    PyGILState_STATE gstate = PyGILState_Ensure();
    int64_t bytes = 0;

    PyObject* result = PyObject_CallMethod(
        py_model_instance_, "memory_usage_bytes", nullptr);
    if (result && PyLong_Check(result)) {
        bytes = PyLong_AsLongLong(result);
    }
    Py_XDECREF(result);
    if (PyErr_Occurred()) PyErr_Clear();

    PyGILState_Release(gstate);
    return bytes;
}

std::string PythonBackend::validate(const std::string& path) const {
    // Quick checks without loading
    fs::path file_path(path);

    if (!fs::exists(file_path)) {
        return "File not found: " + path;
    }

    if (file_path.extension() != ".py") {
        return "Not a Python file: " + path;
    }

    // Check that the file contains "MiiaModel" somewhere (crude but fast)
    std::ifstream f(path);
    if (!f.is_open()) return "Cannot open file: " + path;

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    if (content.find("MiiaModel") == std::string::npos) {
        return "File does not appear to contain a MiiaModel subclass";
    }

    return "";  // Looks valid
}

void PythonBackend::warmup(uint32_t n) {
    if (!loaded_ || !py_model_instance_) return;

    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* py_n = PyLong_FromUnsignedLong(n);
    PyObject* result = PyObject_CallMethod(
        py_model_instance_, "warmup", "(O)", py_n);
    Py_DECREF(py_n);

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Print();
        PyErr_Clear();
    }

    PyGILState_Release(gstate);
}

// ============================================
// find_model_class: import module → scan for MiiaModel subclass
// ============================================

PyObject* PythonBackend::find_model_class(const std::string& path) {
    // 1. Import the miia_model module to get the MiiaModel base class
    PyObject* base_module = PyImport_ImportModule("miia_model");
    if (!base_module) {
        std::cerr << "[PythonBackend] Cannot import miia_model.py — "
                  << "ensure it is in PYTHONPATH or the model directory."
                  << std::endl;
        PyErr_Print();
        return nullptr;
    }

    PyObject* base_class = PyObject_GetAttrString(base_module, "MiiaModel");
    Py_DECREF(base_module);

    if (!base_class) {
        std::cerr << "[PythonBackend] miia_model.MiiaModel not found."
                  << std::endl;
        PyErr_Print();
        return nullptr;
    }

    // 2. Import the user module
    PyObject* py_module_name = PyUnicode_FromString(module_name_.c_str());
    PyObject* user_module = PyImport_Import(py_module_name);
    Py_DECREF(py_module_name);

    if (!user_module) {
        std::cerr << "[PythonBackend] Cannot import module: "
                  << module_name_ << std::endl;
        PyErr_Print();
        Py_DECREF(base_class);
        return nullptr;
    }

    // 3. Scan module dict for a class that is a subclass of MiiaModel
    //    (but is not MiiaModel itself)
    PyObject* module_dict = PyModule_GetDict(user_module);  // borrowed
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    PyObject* found_class = nullptr;

    while (PyDict_Next(module_dict, &pos, &key, &value)) {
        if (!PyType_Check(value)) continue;

        int is_subclass = PyObject_IsSubclass(value, base_class);
        if (is_subclass == 1 && value != base_class) {
            found_class = value;
            Py_INCREF(found_class);
            break;
        }
        if (is_subclass == -1) {
            PyErr_Clear();  // Not a type-safe comparison, skip
        }
    }

    Py_DECREF(user_module);
    Py_DECREF(base_class);

    return found_class;  // new ref or nullptr
}

// ============================================
// inputs_to_py_dict: C++ → numpy
// ============================================

PyObject* PythonBackend::inputs_to_py_dict(
    const std::map<std::string, std::vector<float>>& inputs) const {

    PyObject* dict = PyDict_New();

    for (const auto& [name, data] : inputs) {
        // Find the expected shape from schema
        std::vector<int64_t> shape;
        for (const auto& spec : cached_schema_.inputs) {
            if (spec.name == name) {
                shape = spec.shape;
                break;
            }
        }

        // Build numpy shape, resolving -1 dynamically
        npy_intp np_shape[NPY_MAXDIMS];
        int ndim;

        if (!shape.empty()) {
            ndim = static_cast<int>(shape.size());
            int64_t static_product = 1;
            for (int i = 0; i < ndim; ++i) {
                if (shape[i] != -1) {
                    static_product *= shape[i];
                }
            }
            for (int i = 0; i < ndim; ++i) {
                if (shape[i] == -1 && static_product > 0) {
                    np_shape[i] = static_cast<npy_intp>(data.size()) / static_cast<npy_intp>(static_product);
                } else {
                    np_shape[i] = static_cast<npy_intp>(shape[i]);
                }
            }
        } else {
            // Fallback: flat 1D array
            ndim = 1;
            np_shape[0] = static_cast<npy_intp>(data.size());
        }

        // Create numpy array (copy data — we cannot guarantee lifetime)
        PyObject* arr = PyArray_SimpleNew(ndim, np_shape, NPY_FLOAT32);
        if (!arr) {
            Py_DECREF(dict);
            return nullptr;
        }

        float* arr_data = static_cast<float*>(PyArray_DATA(
            reinterpret_cast<PyArrayObject*>(arr)));
        std::copy(data.begin(), data.end(), arr_data);

        PyObject* py_key = PyUnicode_FromString(name.c_str());
        PyDict_SetItem(dict, py_key, arr);
        Py_DECREF(py_key);
        Py_DECREF(arr);
    }

    return dict;
}

// ============================================
// py_dict_to_outputs: numpy → C++
// ============================================

bool PythonBackend::py_dict_to_outputs(
    PyObject* py_dict,
    std::map<std::string, std::vector<float>>& outputs,
    std::string& error) const {

    if (!PyDict_Check(py_dict)) {
        error = "predict() did not return a dict";
        return false;
    }

    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(py_dict, &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) {
            error = "Output dict has non-string key";
            return false;
        }

        const char* name = PyUnicode_AsUTF8(key);

        // Ensure it's a numpy array and convert to float32 if needed
        PyObject* arr = PyArray_FROMANY(
            value, NPY_FLOAT32, 0, 0,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST);

        if (!arr) {
            error = std::string("Cannot convert output '") + name
                    + "' to float32 numpy array";
            PyErr_Clear();
            return false;
        }

        auto* np_arr = reinterpret_cast<PyArrayObject*>(arr);
        npy_intp total = PyArray_SIZE(np_arr);
        const float* ptr = static_cast<const float*>(PyArray_DATA(np_arr));

        outputs[name] = std::vector<float>(ptr, ptr + total);
        Py_DECREF(arr);
    }

    return true;
}

// ============================================
// extract_schema_from_python: model.get_schema() → C++ ModelSchema
// ============================================

ModelSchema PythonBackend::extract_schema_from_python() const {
    ModelSchema schema;

    if (!py_schema_method_) return schema;

    PyObject* py_schema = PyObject_CallNoArgs(py_schema_method_);
    if (!py_schema) {
        PyErr_Print();
        PyErr_Clear();
        return schema;
    }

    // description
    PyObject* desc = PyObject_GetAttrString(py_schema, "description");
    if (desc && PyUnicode_Check(desc)) {
        schema.description = PyUnicode_AsUTF8(desc);
    }
    Py_XDECREF(desc);

    // author
    PyObject* author = PyObject_GetAttrString(py_schema, "author");
    if (author && PyUnicode_Check(author)) {
        schema.author = PyUnicode_AsUTF8(author);
    }
    Py_XDECREF(author);

    // tags (dict[str,str])
    PyObject* tags = PyObject_GetAttrString(py_schema, "tags");
    if (tags && PyDict_Check(tags)) {
        PyObject* k;
        PyObject* v;
        Py_ssize_t pos = 0;
        while (PyDict_Next(tags, &pos, &k, &v)) {
            if (PyUnicode_Check(k) && PyUnicode_Check(v)) {
                schema.tags[PyUnicode_AsUTF8(k)] = PyUnicode_AsUTF8(v);
            }
        }
    }
    Py_XDECREF(tags);

    // inputs (list[TensorSpec])
    PyObject* inputs = PyObject_GetAttrString(py_schema, "inputs");
    if (inputs && PyList_Check(inputs)) {
        Py_ssize_t n = PyList_Size(inputs);
        for (Py_ssize_t i = 0; i < n; ++i) {
            schema.inputs.push_back(
                parse_tensor_spec(PyList_GetItem(inputs, i)));
        }
    }
    Py_XDECREF(inputs);

    // outputs (list[TensorSpec])
    PyObject* outputs = PyObject_GetAttrString(py_schema, "outputs");
    if (outputs && PyList_Check(outputs)) {
        Py_ssize_t n = PyList_Size(outputs);
        for (Py_ssize_t i = 0; i < n; ++i) {
            schema.outputs.push_back(
                parse_tensor_spec(PyList_GetItem(outputs, i)));
        }
    }
    Py_XDECREF(outputs);

    Py_DECREF(py_schema);

    if (PyErr_Occurred()) PyErr_Clear();
    return schema;
}

// ============================================
// parse_tensor_spec: Python TensorSpec dataclass → C++ TensorSpecData
// ============================================

TensorSpecData PythonBackend::parse_tensor_spec(PyObject* py_spec) const {
    TensorSpecData spec;

    // name
    PyObject* name = PyObject_GetAttrString(py_spec, "name");
    if (name && PyUnicode_Check(name)) {
        spec.name = PyUnicode_AsUTF8(name);
    }
    Py_XDECREF(name);

    // dtype
    PyObject* dtype = PyObject_GetAttrString(py_spec, "dtype");
    if (dtype && PyUnicode_Check(dtype)) {
        std::string dt = PyUnicode_AsUTF8(dtype);
        if (dt == "float32")      spec.dtype = common::FLOAT32;
        else if (dt == "float64") spec.dtype = common::FLOAT64;
        else if (dt == "int32")   spec.dtype = common::INT32;
        else if (dt == "int64")   spec.dtype = common::INT64;
        else if (dt == "uint8")   spec.dtype = common::UINT8;
        else if (dt == "bool")    spec.dtype = common::BOOL;
        else if (dt == "float16") spec.dtype = common::FLOAT16;
    }
    Py_XDECREF(dtype);

    // description
    PyObject* desc = PyObject_GetAttrString(py_spec, "description");
    if (desc && PyUnicode_Check(desc)) {
        spec.description = PyUnicode_AsUTF8(desc);
    }
    Py_XDECREF(desc);

    // shape (list[int])
    PyObject* shape = PyObject_GetAttrString(py_spec, "shape");
    if (shape && PyList_Check(shape)) {
        Py_ssize_t n = PyList_Size(shape);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* dim = PyList_GetItem(shape, i);  // borrowed
            spec.shape.push_back(PyLong_AsLongLong(dim));
        }
    }
    Py_XDECREF(shape);

    // min_value (optional)
    PyObject* min_val = PyObject_GetAttrString(py_spec, "min_value");
    if (min_val && min_val != Py_None && PyFloat_Check(min_val)) {
        spec.min_value = PyFloat_AsDouble(min_val);
        spec.has_constraints = true;
    }
    Py_XDECREF(min_val);

    // max_value (optional)
    PyObject* max_val = PyObject_GetAttrString(py_spec, "max_value");
    if (max_val && max_val != Py_None && PyFloat_Check(max_val)) {
        spec.max_value = PyFloat_AsDouble(max_val);
        spec.has_constraints = true;
    }
    Py_XDECREF(max_val);

    if (PyErr_Occurred()) PyErr_Clear();
    return spec;
}

}  // namespace worker
}  // namespace mlinference
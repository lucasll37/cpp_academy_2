// =============================================================================
// python_backend.cpp — CPython-embedded backend implementation
// =============================================================================

// Python.h MUST be included before any standard headers on some platforms.
#define PY_ARRAY_UNIQUE_SYMBOL MIIA_NUMPY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "inference/python_backend.hpp"
#include "utils/logger.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <dlfcn.h>

namespace miia {
namespace inference {

// ============================================
// Helper: captura exceção Python como string
// ============================================

// Captura o traceback atual do interpretador como string e limpa o estado
// de erro. Usar no lugar de PyErr_Print() para rotear erros Python pelo
// sistema de logging em vez de imprimir direto no stderr.
static std::string fetch_python_error() {
    if (!PyErr_Occurred()) return "";

    PyObject* ptype  = nullptr;
    PyObject* pvalue = nullptr;
    PyObject* ptrace = nullptr;
    PyErr_Fetch(&ptype, &pvalue, &ptrace);
    PyErr_NormalizeException(&ptype, &pvalue, &ptrace);

    std::string result;

    // Tenta formatar via traceback.format_exception (inclui traceback completo)
    PyObject* tb_mod = PyImport_ImportModule("traceback");
    if (tb_mod) {
        PyObject* fmt_fn = PyObject_GetAttrString(tb_mod, "format_exception");
        Py_DECREF(tb_mod);
        if (fmt_fn) {
            PyObject* lines = PyObject_CallFunction(
                fmt_fn, "OOO",
                ptype  ? ptype  : Py_None,
                pvalue ? pvalue : Py_None,
                ptrace ? ptrace : Py_None);
            Py_DECREF(fmt_fn);
            if (lines && PyList_Check(lines)) {
                std::ostringstream oss;
                for (Py_ssize_t i = 0; i < PyList_Size(lines); ++i) {
                    PyObject* s = PyList_GetItem(lines, i);  // borrowed
                    if (PyUnicode_Check(s))
                        oss << PyUnicode_AsUTF8(s);
                }
                result = oss.str();
            }
            Py_XDECREF(lines);
        }
    }

    // Fallback: apenas str(exception)
    if (result.empty() && pvalue) {
        PyObject* s = PyObject_Str(pvalue);
        if (s && PyUnicode_Check(s))
            result = PyUnicode_AsUTF8(s);
        Py_XDECREF(s);
    }

    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptrace);
    PyErr_Clear();
    return result;
}

// ============================================
// Static member definitions
// ============================================

std::mutex PythonBackend::init_mutex_;
int PythonBackend::instance_count_ = 0;

// ============================================
// Constructor / Destructor
// ============================================

PythonBackend::PythonBackend() {
    LOG_DEBUG("python_backend") << "[ctor] PythonBackend construído; instance_count_=" << instance_count_;
}

PythonBackend::~PythonBackend() {
    LOG_DEBUG("python_backend") << "[dtor] PythonBackend destruído; loaded_=" << loaded_ << " module=" << module_name_;
    if (loaded_) unload();
}

// ============================================
// Interpreter management
// ============================================

// FIX: import_array() é uma macro que expande para "return NULL" (ou "return <val>")
// em caso de falha — se chamada diretamente em inject_venv, sai da função sem
// liberar o GIL, causando deadlock no PyGILState_Ensure() seguinte.
// A solução é envolvê-la em uma função auxiliar dedicada, onde o "return" é seguro,
// e usar um flag estático para garantir idempotência entre múltiplas instâncias.
static bool init_numpy_capi() {
    import_array1(false);  // retorna false em caso de falha
    return true;
}

void PythonBackend::inject_venv_from_model_dir(const std::string& model_dir) {
    namespace fs = std::filesystem;

    LOG_DEBUG("python_backend") << "[inject_venv] model_dir=" << model_dir;

    fs::path venv = fs::path(model_dir) / ".venv";
    LOG_DEBUG("python_backend") << "[inject_venv] caminho .venv esperado=" << venv.string();

    // ── Adquire o GIL antes de qualquer operação Python ──────────────────────
    // IMPORTANTE: a numpy C-API deve ser inicializada independentemente de
    // haver um .venv — caso contrário, qualquer chamada posterior que toque
    // PyArray_* segfaulta silenciosamente.
    PyGILState_STATE g = PyGILState_Ensure();
    LOG_DEBUG("python_backend") << "[inject_venv] GIL adquirido";

    if (!fs::is_directory(venv)) {
        LOG_WARN("python_backend") << "[inject_venv] .venv não encontrado em " << venv.string()
                                   << "; prosseguindo sem injeção de site-packages";
    } else {
        std::string venv_abs = fs::canonical(venv).string();
        LOG_DEBUG("python_backend") << "[inject_venv] .venv encontrado; caminho canônico=" << venv_abs;

        std::string script =
            "import sys, glob as _g\n"
            "_venv = '" + venv_abs + "'\n"
            "for _p in _g.glob(_venv + '/lib/python*/site-packages'):\n"
            "    if _p not in sys.path:\n"
            "        sys.path.insert(0, _p)\n";

        LOG_DEBUG("python_backend") << "[inject_venv] executando script de injeção de venv";

        if (PyRun_SimpleString(script.c_str()) != 0) {
            PyErr_Clear();
            LOG_ERROR("python_backend") << "[inject_venv] PyRun_SimpleString falhou ao injetar venv";
        } else {
            LOG_INFO("python_backend") << "[inject_venv] venv injetado com sucesso: " << venv_abs;
        }
    }

    // ── numpy C-API — sempre inicializar, independente do .venv ──────────────
    static bool numpy_capi_ready = false;
    if (!numpy_capi_ready) {
        LOG_DEBUG("python_backend") << "[inject_venv] inicializando numpy C-API";
        numpy_capi_ready = init_numpy_capi();
        if (!numpy_capi_ready) {
            PyErr_Clear();
            LOG_WARN("python_backend") << "[inject_venv] import_array1 falhou; tentando fallback via PyRun_SimpleString";
            if (PyRun_SimpleString("import numpy") == 0) {
                numpy_capi_ready = true;
                LOG_INFO("python_backend") << "[inject_venv] numpy importado via fallback PyRun_SimpleString";
            } else {
                PyObject* exc = PyErr_GetRaisedException();
                if (exc) {
                    PyObject* str = PyObject_Str(exc);
                    if (str) {
                        LOG_ERROR("python_backend") << "[inject_venv] erro do import numpy: " << PyUnicode_AsUTF8(str);
                        Py_DECREF(str);
                    }
                    Py_DECREF(exc);
                }
                PyErr_Clear();
                LOG_ERROR("python_backend") << "[inject_venv] numpy indisponível após fallback";
            }
        } else {
            LOG_DEBUG("python_backend") << "[inject_venv] numpy C-API inicializada via init_numpy_capi()";
        }
    } else {
        LOG_DEBUG("python_backend") << "[inject_venv] numpy C-API já inicializada; skip";
    }

    PyGILState_Release(g);
    LOG_DEBUG("python_backend") << "[inject_venv] GIL liberado; inject_venv concluído";
}

void PythonBackend::ensure_interpreter() {
    LOG_DEBUG("python_backend") << "[ensure_interpreter] chamado; instance_count_=" << instance_count_;
    std::lock_guard<std::mutex> lk(init_mutex_);

    if (instance_count_++ == 0) {
        LOG_DEBUG("python_backend") << "[ensure_interpreter] primeira instância; Py_IsInitialized()=" << Py_IsInitialized();
        if (!Py_IsInitialized()) {
            LOG_DEBUG("python_backend") << "[ensure_interpreter] chamando dlopen(libpython) com RTLD_GLOBAL";
            dlopen("libpython3.12.so.1.0", RTLD_NOW | RTLD_GLOBAL);
            LOG_DEBUG("python_backend") << "[ensure_interpreter] chamando Py_Initialize()";
            Py_Initialize();
            LOG_DEBUG("python_backend") << "[ensure_interpreter] Py_Initialize() concluído; liberando GIL via PyEval_SaveThread()";
            PyEval_SaveThread();
        } else {
            // FIX: processo hospedeiro já inicializou o Python.
            // Verificamos se a thread atual possui o GIL antes de liberá-lo.
            LOG_DEBUG("python_backend") << "[ensure_interpreter] intérprete já inicializado pela app hospedeira";
            if (PyGILState_Check()) {
                LOG_DEBUG("python_backend") << "[ensure_interpreter] thread possui o GIL; liberando via PyEval_SaveThread()";
                PyEval_SaveThread();
            } else {
                LOG_DEBUG("python_backend") << "[ensure_interpreter] thread não possui o GIL; skip PyEval_SaveThread()";
            }
        }
    } else {
        LOG_DEBUG("python_backend") << "[ensure_interpreter] intérprete já ativo; instance_count_=" << instance_count_;
    }
}

// void PythonBackend::ensure_interpreter() {
//     LOG_DEBUG("python_backend") << "[ensure_interpreter] chamado; instance_count_=" << instance_count_;
//     std::lock_guard<std::mutex> lk(init_mutex_);

//     if (instance_count_++ == 0) {
//         LOG_DEBUG("python_backend") << "[ensure_interpreter] primeira instância; Py_IsInitialized()=" << Py_IsInitialized();
//         if (!Py_IsInitialized()) {
//             LOG_DEBUG("python_backend") << "[ensure_interpreter] chamando Py_Initialize()";
//             Py_Initialize();
//             LOG_DEBUG("python_backend") << "[ensure_interpreter] Py_Initialize() concluído; liberando GIL via PyEval_SaveThread()";
//             PyEval_SaveThread();
//         } else {
//             // FIX: processo hospedeiro já inicializou o Python.
//             // Verificamos se a thread atual possui o GIL antes de liberá-lo.
//             LOG_DEBUG("python_backend") << "[ensure_interpreter] intérprete já inicializado pela app hospedeira";
//             if (PyGILState_Check()) {
//                 LOG_DEBUG("python_backend") << "[ensure_interpreter] thread possui o GIL; liberando via PyEval_SaveThread()";
//                 PyEval_SaveThread();
//             } else {
//                 LOG_DEBUG("python_backend") << "[ensure_interpreter] thread não possui o GIL; skip PyEval_SaveThread()";
//             }
//         }
//     } else {
//         LOG_DEBUG("python_backend") << "[ensure_interpreter] intérprete já ativo; instance_count_=" << instance_count_;
//     }
// }

void PythonBackend::release_interpreter() {
    std::lock_guard<std::mutex> lk(init_mutex_);
    --instance_count_;
    LOG_DEBUG("python_backend") << "[release_interpreter] instance_count_ decrementado para " << instance_count_;
    if (instance_count_ == 0) {
        LOG_DEBUG("python_backend") << "[release_interpreter] última instância liberada — Py_Finalize() NÃO chamado (unsafe)";
    }
}

// ============================================
// Lifecycle: load
// ============================================

bool PythonBackend::load(const std::string& path,
                         const std::map<std::string, std::string>& /*config*/) {
    LOG_DEBUG("python_backend") << "[load] chamado; path=" << path;

    ensure_interpreter();

    namespace fs = std::filesystem;
    fs::path p(path);
    model_dir_   = p.parent_path().string();
    module_name_ = p.stem().string();

    LOG_DEBUG("python_backend") << "[load] model_dir_=" << model_dir_
         << " module_name_=" << module_name_
         << " exists=" << fs::exists(path);

    inject_venv_from_model_dir(model_dir_);

    bool success = false;

    LOG_DEBUG("python_backend") << "[load] adquirindo GIL";
    PyGILState_STATE gstate = PyGILState_Ensure();
    LOG_DEBUG("python_backend") << "[load] GIL adquirido";

    try {
        // 1. Prepend model directory to sys.path (sem duplicatas)
        LOG_DEBUG("python_backend") << "[load] inserindo model_dir_ no início de sys.path (se ausente)";
        PyObject* sys_path = PySys_GetObject("path");  // borrowed
        if (!sys_path) {
            LOG_ERROR("python_backend") << "[load] PySys_GetObject('path') retornou nullptr";
        }
        // FIX: verificar duplicata antes de inserir para evitar crescimento
        // ilimitado do sys.path quando o mesmo model_dir é carregado N vezes
        // no mesmo processo (e.g. durante testes com múltiplos load_model()).
        PyObject* dir_str = PyUnicode_FromString(model_dir_.c_str());
        if (PySequence_Contains(sys_path, dir_str) == 0)
            PyList_Insert(sys_path, 0, dir_str);
        Py_DECREF(dir_str);

        // 2. Find the MiiaModel subclass
        LOG_DEBUG("python_backend") << "[load] chamando find_model_class()";
        PyObject* cls = find_model_class(path);
        if (!cls) {
            LOG_ERROR("python_backend") << "[load] find_model_class retornou nullptr para path=" << path;
            PyGILState_Release(gstate);
            release_interpreter();
            return false;
        }
        LOG_DEBUG("python_backend") << "[load] find_model_class retornou classe válida";

        // 3. Instantiate
        LOG_DEBUG("python_backend") << "[load] instanciando classe do modelo";
        py_model_instance_ = PyObject_CallNoArgs(cls);
        Py_DECREF(cls);
        if (!py_model_instance_) {
            LOG_ERROR("python_backend") << "[load] PyObject_CallNoArgs() falhou ao instanciar modelo:\n" << fetch_python_error();
            PyGILState_Release(gstate);
            release_interpreter();
            return false;
        }
        LOG_DEBUG("python_backend") << "[load] instância criada; py_model_instance_=" << (void*)py_model_instance_;

        // 4. Call model.load()
        LOG_DEBUG("python_backend") << "[load] chamando model.load()";
        PyObject* load_result = PyObject_CallMethod(py_model_instance_, "load", nullptr);
        if (!load_result) {
            LOG_ERROR("python_backend") << "[load] model.load() lançou exceção:\n" << fetch_python_error();
            Py_CLEAR(py_model_instance_);
            PyGILState_Release(gstate);
            release_interpreter();
            return false;
        }
        LOG_DEBUG("python_backend") << "[load] model.load() retornou com sucesso";
        Py_DECREF(load_result);

        // 5. Cache bound methods
        LOG_DEBUG("python_backend") << "[load] obtendo atributos 'predict' e 'get_schema'";
        py_predict_method_ = PyObject_GetAttrString(py_model_instance_, "predict");
        py_schema_method_  = PyObject_GetAttrString(py_model_instance_, "get_schema");

        if (!py_predict_method_ || !py_schema_method_) {
            LOG_ERROR("python_backend") << "[load] modelo não possui predict() e/ou get_schema():\n" << fetch_python_error();
            Py_CLEAR(py_predict_method_);
            Py_CLEAR(py_schema_method_);
            Py_CLEAR(py_model_instance_);
            PyGILState_Release(gstate);
            release_interpreter();
            return false;
        }
        LOG_DEBUG("python_backend") << "[load] métodos predict e get_schema cacheados";

        // 6. Cache schema
        LOG_DEBUG("python_backend") << "[load] extraindo schema via extract_schema_from_python()";
        cached_schema_ = extract_schema_from_python();
        LOG_DEBUG("python_backend") << "[load] schema extraído: description='" << cached_schema_.description
             << "' author='" << cached_schema_.author
             << "' n_inputs=" << cached_schema_.inputs.size()
             << " n_outputs=" << cached_schema_.outputs.size();

        loaded_    = true;
        load_time_ = std::chrono::steady_clock::now();
        last_used_ = load_time_;
        success    = true;

        LOG_INFO("python_backend") << "[load] modelo carregado: module=" << module_name_ << " path=" << path;

    } catch (const std::exception& e) {
        LOG_ERROR("python_backend") << "[load] exceção C++ durante load: " << e.what();
    }

    PyGILState_Release(gstate);
    LOG_DEBUG("python_backend") << "[load] GIL liberado; success=" << success;

    if (!success) release_interpreter();
    return success;
}

// ============================================
// Lifecycle: unload
// ============================================

void PythonBackend::unload() {
    LOG_DEBUG("python_backend") << "[unload] chamado; loaded_=" << loaded_ << " module=" << module_name_;
    if (!loaded_) {
        LOG_DEBUG("python_backend") << "[unload] modelo não estava carregado, retornando";
        return;
    }

    LOG_DEBUG("python_backend") << "[unload] adquirindo GIL";
    PyGILState_STATE gstate = PyGILState_Ensure();

    if (py_model_instance_) {
        LOG_DEBUG("python_backend") << "[unload] chamando model.unload()";
        PyObject* result = PyObject_CallMethod(py_model_instance_, "unload", nullptr);
        if (result) {
            LOG_DEBUG("python_backend") << "[unload] model.unload() retornou com sucesso";
            Py_DECREF(result);
        } else {
            LOG_WARN("python_backend") << "[unload] model.unload() retornou nullptr (erro suprimido)";
            PyErr_Clear();
        }
    } else {
        LOG_DEBUG("python_backend") << "[unload] py_model_instance_ é nullptr, skip model.unload()";
    }

    Py_CLEAR(py_predict_method_);
    Py_CLEAR(py_schema_method_);
    Py_CLEAR(py_model_instance_);

    PyGILState_Release(gstate);
    LOG_DEBUG("python_backend") << "[unload] GIL liberado; chamando release_interpreter()";
    release_interpreter();

    loaded_ = false;
    cached_schema_ = {};
    LOG_INFO("python_backend") << "[unload] modelo descarregado: module=" << module_name_;
}

// ============================================
// Inference
// ============================================

InferenceResult PythonBackend::predict(const client::Object& inputs) {
    LOG_DEBUG("python_backend") << "[predict] chamado; loaded_=" << loaded_ << " n_inputs=" << inputs.size();

    if (!loaded_ || !py_predict_method_) {
        LOG_ERROR("python_backend") << "[predict] FALHA PRÉ-CONDIÇÃO: loaded_=" << loaded_
             << " py_predict_method_=" << (void*)py_predict_method_;
        return {false, {}, 0.0, "Model not loaded"};
    }

    for (const auto& [k, v] : inputs) {
        LOG_DEBUG("python_backend") << "[predict] input key='" << k << "'";
    }

    auto start = std::chrono::high_resolution_clock::now();

    LOG_DEBUG("python_backend") << "[predict] adquirindo GIL";
    PyGILState_STATE gstate = PyGILState_Ensure();

    InferenceResult result;

    // 1. Convert Object → Python dict
    LOG_DEBUG("python_backend") << "[predict] convertendo inputs para Python dict";
    PyObject* py_inputs = inputs_to_py_dict(inputs);
    if (!py_inputs) {
        std::string err = "Failed to convert inputs to Python dict";
        LOG_ERROR("python_backend") << "[predict] inputs_to_py_dict() retornou nullptr";
        if (PyErr_Occurred()) { LOG_ERROR("python_backend") << "[predict] erro ao converter inputs:\n" << fetch_python_error(); }
        PyGILState_Release(gstate);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics_.record(ms, false);
        return {false, {}, ms, err};
    }

    // 2. Call model.predict(inputs)
    LOG_DEBUG("python_backend") << "[predict] chamando model.predict(inputs)";
    PyObject* py_args   = PyTuple_Pack(1, py_inputs);
    PyObject* py_result = PyObject_CallObject(py_predict_method_, py_args);
    Py_DECREF(py_args);
    Py_DECREF(py_inputs);

    if (!py_result) {
        std::string py_err = fetch_python_error();
        std::string err = "model.predict() raised an exception";
        LOG_ERROR("python_backend") << "[predict] model.predict() lançou exceção:\n" << py_err;
        PyGILState_Release(gstate);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        metrics_.record(ms, false);
        return {false, {}, ms, py_err.empty() ? err : py_err};
    }
    LOG_DEBUG("python_backend") << "[predict] model.predict() retornou py_result=" << (void*)py_result;

    // 3. Convert Python dict[str, np.ndarray] → client::Object
    LOG_DEBUG("python_backend") << "[predict] convertendo resultado para client::Object";
    std::string conv_err;
    bool ok = py_dict_to_outputs(py_result, result.outputs, conv_err);
    Py_DECREF(py_result);

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    LOG_DEBUG("python_backend") << "[predict] py_dict_to_outputs ok=" << ok
         << " inference_time_ms=" << ms;

    for (const auto& [k, v] : result.outputs) {
        LOG_DEBUG("python_backend") << "[predict] output key='" << k << "'";
    }

    PyGILState_Release(gstate);
    LOG_DEBUG("python_backend") << "[predict] GIL liberado";

    if (!ok) {
        LOG_ERROR("python_backend") << "[predict] falha na conversão do resultado: " << conv_err;
        metrics_.record(ms, false);
        return {false, {}, ms, conv_err};
    }

    touch();
    metrics_.record(ms, true);

    result.success           = true;
    result.inference_time_ms = ms;

    LOG_DEBUG("python_backend") << "[predict] SUCESSO; inference_time_ms=" << ms
         << " n_outputs=" << result.outputs.size();
    return result;
}

// ============================================
// find_model_class
// ============================================

PyObject* PythonBackend::find_model_class(const std::string& /*path*/) {
    LOG_DEBUG("python_backend") << "[find_model_class] chamado; module_name_=" << module_name_;

    // 1. Import miia_model to get the base class
    LOG_DEBUG("python_backend") << "[find_model_class] importando módulo 'miia_model'";
    PyObject* base_module = PyImport_ImportModule("miia_model");
    if (!base_module) {
        LOG_ERROR("python_backend") << "[find_model_class] PyImport_ImportModule('miia_model') retornou nullptr:\n" << fetch_python_error();
        return nullptr;
    }
    LOG_DEBUG("python_backend") << "[find_model_class] 'miia_model' importado; base_module=" << (void*)base_module;

    PyObject* base_class = PyObject_GetAttrString(base_module, "MiiaModel");
    Py_DECREF(base_module);
    if (!base_class) {
        LOG_ERROR("python_backend") << "[find_model_class] atributo 'MiiaModel' não encontrado em 'miia_model':\n" << fetch_python_error();
        return nullptr;
    }
    LOG_DEBUG("python_backend") << "[find_model_class] base_class MiiaModel=" << (void*)base_class;

    // 2. Import the user module
    // FIX: remover entrada stale de sys.modules antes de importar para forçar
    // re-import limpo. Sem isso, reloads do mesmo módulo (e.g. após unload +
    // load_model() em testes) re-usam o objeto cacheado, incluindo erros de
    // import de módulos que não existem mais (ex: "ModuleNotFoundError: modelo").
    LOG_DEBUG("python_backend") << "[find_model_class] limpando sys.modules['" << module_name_ << "'] para re-import limpo";
    PyObject* sys_modules = PyImport_GetModuleDict();  // borrowed
    PyDict_DelItemString(sys_modules, module_name_.c_str());
    PyErr_Clear();  // ignora KeyError se o módulo não estava em sys.modules

    LOG_DEBUG("python_backend") << "[find_model_class] importando módulo do usuário: '" << module_name_ << "'";
    PyObject* py_name  = PyUnicode_FromString(module_name_.c_str());
    PyObject* user_mod = PyImport_Import(py_name);
    Py_DECREF(py_name);

    if (!user_mod) {
        LOG_ERROR("python_backend") << "[find_model_class] não foi possível importar módulo '" << module_name_ << "':\n" << fetch_python_error();
        Py_DECREF(base_class);
        return nullptr;
    }
    LOG_DEBUG("python_backend") << "[find_model_class] módulo '" << module_name_ << "' importado; user_mod=" << (void*)user_mod;

    // 3. Scan module dict for a MiiaModel subclass (not MiiaModel itself)
    LOG_DEBUG("python_backend") << "[find_model_class] varrendo dict do módulo em busca de subclasse de MiiaModel";
    PyObject* module_dict = PyModule_GetDict(user_mod);  // borrowed
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos      = 0;
    PyObject* found_cls = nullptr;
    int checked_types   = 0;

    while (PyDict_Next(module_dict, &pos, &key, &value)) {
        if (!PyType_Check(value)) continue;
        checked_types++;

        const char* type_name = PyUnicode_Check(key) ? PyUnicode_AsUTF8(key) : "<non-string>";
        LOG_DEBUG("python_backend") << "[find_model_class] verificando tipo '" << type_name << "'";

        int is_sub = PyObject_IsSubclass(value, base_class);

        if (is_sub == 1 && value != base_class) {
            PyObject* abstract_methods = PyObject_GetAttrString(value, "__abstractmethods__");
            bool is_concrete = false;
            if (abstract_methods == nullptr) {
                PyErr_Clear();
                is_concrete = true;
            } else {
                Py_ssize_t n = PySet_Size(abstract_methods);
                if (n == -1) { PyErr_Clear(); n = 0; }
                is_concrete = (n == 0);
                Py_DECREF(abstract_methods);
            }

            if (is_concrete) {
                LOG_DEBUG("python_backend") << "[find_model_class] ENCONTRADO (concreta): '" << type_name << "'";
                found_cls = value;
                Py_INCREF(found_cls);
                break;
            }
            LOG_DEBUG("python_backend") << "[find_model_class] ignorando abstrata '" << type_name
                << "' (__abstractmethods__ não vazio)";
        }
        if (is_sub == -1) {
            LOG_DEBUG("python_backend") << "[find_model_class] PyObject_IsSubclass retornou -1 para '" << type_name << "'; limpando erro";
            PyErr_Clear();
        }
    }
    
    // while (PyDict_Next(module_dict, &pos, &key, &value)) {
    //     if (!PyType_Check(value)) continue;
    //     checked_types++;

    //     const char* type_name = PyUnicode_Check(key) ? PyUnicode_AsUTF8(key) : "<non-string>";
    //     LOG_DEBUG("python_backend") << "[find_model_class] verificando tipo '" << type_name << "'";

    //     int is_sub = PyObject_IsSubclass(value, base_class);

    //     if (is_sub == 1 && value != base_class) {
    //         LOG_DEBUG("python_backend") << "[find_model_class] ENCONTRADO: subclasse '" << type_name << "'";
    //         found_cls = value;
    //         Py_INCREF(found_cls);
    //         break;
    //     }
    //     if (is_sub == -1) {
    //         LOG_DEBUG("python_backend") << "[find_model_class] PyObject_IsSubclass retornou -1 para '" << type_name << "'; limpando erro";
    //         PyErr_Clear();
    //     }
    // }

    LOG_DEBUG("python_backend") << "[find_model_class] varredura concluída; tipos_verificados=" << checked_types
         << " found_cls=" << (void*)found_cls;

    Py_DECREF(user_mod);
    Py_DECREF(base_class);

    if (!found_cls) {
        LOG_WARN("python_backend") << "[find_model_class] nenhuma subclasse de MiiaModel encontrada no módulo '" << module_name_ << "'";
    }

    return found_cls;  // new ref or nullptr
}

// ============================================
// value_to_py: Value → PyObject* (recursivo)
// ============================================

static PyObject* value_to_py(const client::Value& val)
{
    if (val.is_null()) {
        Py_RETURN_NONE;
    }

    if (val.is_number()) {
        return PyFloat_FromDouble(val.as_number());
    }

    if (val.is_bool()) {
        return PyBool_FromLong(static_cast<long>(val.as_bool()));
    }

    if (val.is_string()) {
        return PyUnicode_FromString(val.as_string().c_str());
    }

    if (val.is_array()) {
        const auto& arr = val.as_array();
        PyObject* list = PyList_New(static_cast<Py_ssize_t>(arr.size()));
        if (!list) return nullptr;
        for (std::size_t i = 0; i < arr.size(); ++i) {
            PyObject* elem = value_to_py(arr[i]);
            if (!elem) { Py_DECREF(list); return nullptr; }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), elem);  // steals ref
        }
        return list;
    }

    if (val.is_object()) {
        const auto& obj = val.as_object();
        PyObject* dict = PyDict_New();
        if (!dict) return nullptr;
        for (const auto& [k, v] : obj) {
            PyObject* py_val = value_to_py(v);
            if (!py_val) { Py_DECREF(dict); return nullptr; }
            PyObject* py_key = PyUnicode_FromString(k.c_str());
            if (!py_key) { Py_DECREF(py_val); Py_DECREF(dict); return nullptr; }
            int rc = PyDict_SetItem(dict, py_key, py_val);
            Py_DECREF(py_key);
            Py_DECREF(py_val);
            if (rc != 0) { Py_DECREF(dict); return nullptr; }
        }
        return dict;
    }

    Py_RETURN_NONE;
}

// ============================================
// inputs_to_py_dict: Object → Python dict
// ============================================

PyObject* PythonBackend::inputs_to_py_dict(const client::Object& inputs) const
{
    LOG_DEBUG("python_backend") << "[inputs_to_py_dict] chamado; n_keys=" << inputs.size();

    PyObject* dict = PyDict_New();
    if (!dict) {
        LOG_ERROR("python_backend") << "[inputs_to_py_dict] PyDict_New retornou nullptr";
        return nullptr;
    }

    for (const auto& [name, val] : inputs) {
        LOG_DEBUG("python_backend") << "[inputs_to_py_dict] convertendo key='" << name << "'";
        PyObject* py_val = value_to_py(val);
        if (!py_val) {
            LOG_ERROR("python_backend") << "[inputs_to_py_dict] value_to_py falhou para key='" << name << "'";
            Py_DECREF(dict);
            return nullptr;
        }

        PyObject* py_key = PyUnicode_FromString(name.c_str());
        if (!py_key) {
            LOG_ERROR("python_backend") << "[inputs_to_py_dict] PyUnicode_FromString falhou para key='" << name << "'";
            Py_DECREF(py_val);
            Py_DECREF(dict);
            return nullptr;
        }

        int rc = PyDict_SetItem(dict, py_key, py_val);
        Py_DECREF(py_key);
        Py_DECREF(py_val);
        if (rc != 0) {
            LOG_ERROR("python_backend") << "[inputs_to_py_dict] PyDict_SetItem rc=" << rc << " para key='" << name << "'";
            Py_DECREF(dict);
            return nullptr;
        }
    }

    LOG_DEBUG("python_backend") << "[inputs_to_py_dict] dict construído com sucesso; ptr=" << (void*)dict;
    return dict;
}

// ============================================
// py_dict_to_outputs: dict[str, np.ndarray] → Object
// ============================================

bool PythonBackend::py_dict_to_outputs(
    PyObject* py_dict,
    client::Object& outputs,
    std::string& error) const {

    LOG_DEBUG("python_backend") << "[py_dict_to_outputs] chamado; py_dict=" << (void*)py_dict;

    if (!PyDict_Check(py_dict)) {
        error = "predict() did not return a dict";
        LOG_ERROR("python_backend") << "[py_dict_to_outputs] py_dict não é um dict Python";
        return false;
    }

    LOG_DEBUG("python_backend") << "[py_dict_to_outputs] PyDict_Size=" << PyDict_Size(py_dict);

    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(py_dict, &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) {
            error = "Output dict has non-string key";
            LOG_ERROR("python_backend") << "[py_dict_to_outputs] chave não é string na posição " << pos;
            return false;
        }

        const char* name = PyUnicode_AsUTF8(key);
        LOG_DEBUG("python_backend") << "[py_dict_to_outputs] processando output key='" << name
             << "' type=" << Py_TYPE(value)->tp_name;

        PyObject* arr = PyArray_FROMANY(
            value, NPY_FLOAT32, 0, 0,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST);

        if (!arr) {
            error = std::string("Cannot convert output '") + name
                    + "' to float32 numpy array";
            LOG_ERROR("python_backend") << "[py_dict_to_outputs] PyArray_FROMANY falhou para key='" << name << "'";
            PyErr_Clear();
            return false;
        }

        auto*    np_arr = reinterpret_cast<PyArrayObject*>(arr);
        npy_intp total  = PyArray_SIZE(np_arr);
        int      ndim   = PyArray_NDIM(np_arr);
        const float* ptr = static_cast<const float*>(PyArray_DATA(np_arr));

        // Log shape
        {
            std::ostringstream shape_oss;
            shape_oss << "[";
            for (int d = 0; d < ndim; ++d) {
                shape_oss << PyArray_DIM(np_arr, d);
                if (d < ndim - 1) shape_oss << ", ";
            }
            shape_oss << "]";
            LOG_DEBUG("python_backend") << "[py_dict_to_outputs] key='" << name
                 << "' ndim=" << ndim << " shape=" << shape_oss.str() << " total=" << total;
        }

        if (total == 1) {
            LOG_DEBUG("python_backend") << "[py_dict_to_outputs] key='" << name << "' escalar=" << ptr[0];
            outputs[name] = client::Value{static_cast<double>(ptr[0])};
        } else {
            std::ostringstream vals_oss;
            npy_intp preview = std::min(total, (npy_intp)8);
            vals_oss << "[";
            for (npy_intp i = 0; i < preview; ++i) {
                vals_oss << ptr[i];
                if (i < preview - 1) vals_oss << ", ";
            }
            if (total > 8) vals_oss << ", ...";
            vals_oss << "]";
            LOG_DEBUG("python_backend") << "[py_dict_to_outputs] key='" << name
                 << "' primeiros valores=" << vals_oss.str();

            client::Array out_arr;
            out_arr.reserve(static_cast<size_t>(total));
            for (npy_intp i = 0; i < total; ++i)
                out_arr.push_back(client::Value{static_cast<double>(ptr[i])});
            outputs[name] = client::Value{std::move(out_arr)};
        }

        Py_DECREF(arr);
        LOG_DEBUG("python_backend") << "[py_dict_to_outputs] key='" << name << "' convertido e inserido";
    }

    LOG_DEBUG("python_backend") << "[py_dict_to_outputs] concluído; n_outputs=" << outputs.size();
    return true;
}

// ============================================
// Schema extraction helpers
// ============================================

ModelSchema PythonBackend::get_schema() const {
    LOG_DEBUG("python_backend") << "[get_schema] retornando cached_schema_; description='" << cached_schema_.description << "'";
    return cached_schema_;
}

ModelSchema PythonBackend::extract_schema_from_python() const {
    LOG_DEBUG("python_backend") << "[extract_schema_from_python] chamado; py_schema_method_=" << (void*)py_schema_method_;
    ModelSchema schema;

    if (!py_schema_method_) {
        LOG_WARN("python_backend") << "[extract_schema_from_python] py_schema_method_ é nullptr, retornando schema vazio";
        return schema;
    }

    LOG_DEBUG("python_backend") << "[extract_schema_from_python] chamando get_schema()";
    PyObject* py_schema = PyObject_CallNoArgs(py_schema_method_);
    if (!py_schema) {
        LOG_ERROR("python_backend") << "[extract_schema_from_python] get_schema() retornou nullptr";
        PyErr_Clear();
        return schema;
    }
    LOG_DEBUG("python_backend") << "[extract_schema_from_python] py_schema=" << (void*)py_schema;

    auto get_str = [](PyObject* obj, const char* attr) -> std::string {
        PyObject* v = PyObject_GetAttrString(obj, attr);
        if (!v) { PyErr_Clear(); return ""; }
        std::string s;
        if (PyUnicode_Check(v)) s = PyUnicode_AsUTF8(v);
        Py_DECREF(v);
        return s;
    };

    schema.description = get_str(py_schema, "description");
    schema.author      = get_str(py_schema, "author");
    LOG_DEBUG("python_backend") << "[extract_schema_from_python] description='" << schema.description
         << "' author='" << schema.author << "'";

    auto parse_list = [&](const char* attr, std::vector<TensorSpecData>& vec) {
        LOG_DEBUG("python_backend") << "[extract_schema_from_python] parse_list attr='" << attr << "'";
        PyObject* lst = PyObject_GetAttrString(py_schema, attr);
        if (!lst) {
            LOG_WARN("python_backend") << "[extract_schema_from_python] atributo '" << attr << "' não encontrado no schema";
            PyErr_Clear();
            return;
        }
        Py_ssize_t n = PyList_Size(lst);
        LOG_DEBUG("python_backend") << "[extract_schema_from_python] '" << attr << "' tem " << n << " entradas";
        for (Py_ssize_t i = 0; i < n; ++i) {
            TensorSpecData spec = parse_tensor_spec(PyList_GetItem(lst, i));
            LOG_DEBUG("python_backend") << "[extract_schema_from_python] " << attr << "[" << i
                 << "] name='" << spec.name << "' shape_size=" << spec.shape.size()
                 << " structured=" << spec.structured;
            vec.push_back(spec);
        }
        Py_DECREF(lst);
    };

    parse_list("inputs",  schema.inputs);
    parse_list("outputs", schema.outputs);

    Py_DECREF(py_schema);
    LOG_DEBUG("python_backend") << "[extract_schema_from_python] concluído; n_inputs=" << schema.inputs.size()
         << " n_outputs=" << schema.outputs.size();
    return schema;
}

TensorSpecData PythonBackend::parse_tensor_spec(PyObject* py_spec) const {
    LOG_DEBUG("python_backend") << "[parse_tensor_spec] chamado; py_spec=" << (void*)py_spec;
    TensorSpecData spec;

    auto get_str = [](PyObject* obj, const char* attr) -> std::string {
        PyObject* v = PyObject_GetAttrString(obj, attr);
        if (!v) { PyErr_Clear(); return ""; }
        std::string s;
        if (PyUnicode_Check(v)) s = PyUnicode_AsUTF8(v);
        Py_DECREF(v);
        return s;
    };

    spec.name        = get_str(py_spec, "name");
    spec.description = get_str(py_spec, "description");
    LOG_DEBUG("python_backend") << "[parse_tensor_spec] name='" << spec.name
         << "' description='" << spec.description << "'";

    // shape
    PyObject* py_shape = PyObject_GetAttrString(py_spec, "shape");
    if (py_shape && PyList_Check(py_shape)) {
        Py_ssize_t n = PyList_Size(py_shape);
        std::ostringstream shape_oss;
        shape_oss << "[";
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* d = PyList_GetItem(py_shape, i);
            long long dim = PyLong_Check(d) ? PyLong_AsLongLong(d) : -1;
            spec.shape.push_back(dim);
            shape_oss << dim;
            if (i < n - 1) shape_oss << ", ";
        }
        shape_oss << "]";
        LOG_DEBUG("python_backend") << "[parse_tensor_spec] name='" << spec.name
             << "' shape=" << shape_oss.str();
    } else {
        LOG_DEBUG("python_backend") << "[parse_tensor_spec] atributo 'shape' ausente ou não é lista";
        if (py_shape) PyErr_Clear();
    }
    Py_XDECREF(py_shape);

    // structured
    PyObject* py_structured = PyObject_GetAttrString(py_spec, "structured");
    if (py_structured) {
        spec.structured = (PyObject_IsTrue(py_structured) == 1);
        LOG_DEBUG("python_backend") << "[parse_tensor_spec] name='" << spec.name
             << "' structured=" << spec.structured;
        Py_DECREF(py_structured);
    } else {
        LOG_DEBUG("python_backend") << "[parse_tensor_spec] atributo 'structured' ausente; usando default false";
        PyErr_Clear();
    }

    return spec;
}

// ============================================
// Memory usage
// ============================================

int64_t PythonBackend::memory_usage_bytes() const {
    LOG_DEBUG("python_backend") << "[memory_usage_bytes] retornando 0 (não implementado)";
    return 0;  // Not trivially available from Python side
}

// ============================================
// Validate
// ============================================

std::string PythonBackend::validate(const std::string& path) const {
    namespace fs = std::filesystem;
    LOG_DEBUG("python_backend") << "[validate] path='" << path
         << "' exists=" << fs::exists(path)
         << " extension='" << fs::path(path).extension().string() << "'";
    if (!fs::exists(path)) return "File not found: " + path;
    if (fs::path(path).extension() != ".py")
        return "Not a Python file: " + path;
    LOG_DEBUG("python_backend") << "[validate] path válido";
    return "";
}

// ============================================
// Warmup override
// ============================================

void PythonBackend::warmup(uint32_t n) {
    LOG_DEBUG("python_backend") << "[warmup] chamado; n=" << n;
    ModelBackend::warmup(n);
    LOG_DEBUG("python_backend") << "[warmup] concluído";
}

}  // namespace inference
}  // namespace miia
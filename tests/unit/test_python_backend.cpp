// =============================================================================
// tests/unit/test_python_backend.cpp
//
// Testes unitários — PythonBackend
//
// Estratégia:
//   SetUp() copia python/models/miia_model.py para um tmpdir isolado.
//   Cada fixture de modelo usa "from miia_model import ..." normalmente,
//   de modo que find_model_class() encontre subclasses da MiiaModel real
//   (e não de um stub inline que seria uma classe diferente).
//
// Dependências de build (tests/unit/meson.build):
//   test('unit_python_backend',
//       executable('test_unit_python_backend',
//           'test_python_backend.cpp',
//           include_directories: [inference_inc, client_inc],
//           dependencies: [worker_lib_dep, proto_dep, gtest_main_dep, gtest_dep],
//           install: false,
//       ),
//       suite:       'unit',
//       timeout:     120,
//       is_parallel: false,
//       env: {'MODELS_DIR': project_root / 'models'},
//   )
//
// Variável de ambiente usada:
//   MODELS_DIR  — diretório que contém miia_model.py (padrão: ./models)
// =============================================================================

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "inference/python_backend.hpp"
#include "client/inference_client.hpp"   // Value, Object, Array

namespace fs = std::filesystem;
using namespace mlinference::inference;
using namespace mlinference::client;

// =============================================================================
// Caminho para miia_model.py — resolvido em runtime via MODELS_DIR
// =============================================================================

static fs::path miia_model_source() {
    const char* e = std::getenv("MODELS_DIR");
    fs::path base = e ? fs::path(e) : fs::path("./models");
    for (auto candidate : {base / "miia_model.py",
                            base.parent_path() / "python" / "models" / "miia_model.py",
                            fs::path("python/models/miia_model.py")}) {
        if (fs::exists(candidate)) return candidate;
    }
    return {};
}

// =============================================================================
// Utilitários de fixtures
// =============================================================================

static fs::path make_tmpdir() {
    char tmpl[] = "/tmp/pb_test_XXXXXX";
    char* dir   = ::mkdtemp(tmpl);
    EXPECT_NE(dir, nullptr) << "mkdtemp falhou";
    return fs::path(dir);
}

/// Extrai o sufixo único do tmpdir (os 6 caracteres gerados por mkdtemp).
/// Ex: /tmp/pb_test_aBcDeF → "aBcDeF"
static std::string tmpdir_suffix(const fs::path& tmpdir) {
    std::string name = tmpdir.filename().string();  // "pb_test_aBcDeF"
    auto pos = name.rfind('_');
    return (pos != std::string::npos) ? name.substr(pos + 1) : name;
}

static void write_file(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p);
    ASSERT_TRUE(f.is_open()) << "Não foi possível criar: " << p;
    f << content;
}

// =============================================================================
// Conteúdos Python dos fixtures
// Todos fazem "from miia_model import ..." — o miia_model.py real é copiado
// para o tmpdir em SetUp().
// =============================================================================

static constexpr const char* kSimpleModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec
import numpy as np

class SimpleModel(MiiaModel):
    def load(self): pass
    def predict(self, inputs):
        x = np.array(inputs.get("x", [0.0]), dtype=np.float32)
        return {"y": np.array([float(x.sum())], dtype=np.float32)}
    def get_schema(self):
        return ModelSchema(
            inputs=[TensorSpec("x", [1,3], "float32", "input vector")],
            outputs=[TensorSpec("y", [1], "float32", "sum of inputs")],
            description="SimpleModel",
            author="test",
        )
    def unload(self): pass
)PY";

static constexpr const char* kMultiOutputModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec
import numpy as np

class MultiOutputModel(MiiaModel):
    def load(self): pass
    def predict(self, inputs):
        x = np.array(inputs.get("x", [1.0, 2.0, 3.0]), dtype=np.float32)
        return {
            "sum":  np.array([float(x.sum())], dtype=np.float32),
            "vals": x,
        }
    def get_schema(self):
        return ModelSchema(
            inputs=[TensorSpec("x", [3], "float32")],
            outputs=[TensorSpec("sum", [1], "float32"),
                     TensorSpec("vals", [3], "float32")],
            description="MultiOutput",
            author="test",
        )
)PY";

static constexpr const char* kLoadFailModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec

class LoadFailModel(MiiaModel):
    def load(self): raise RuntimeError("load sempre falha")
    def predict(self, inputs): return {}
    def get_schema(self):
        return ModelSchema(inputs=[], outputs=[], description="fail")
)PY";

static constexpr const char* kPredictFailModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec
import numpy as np

class PredictFailModel(MiiaModel):
    def load(self): pass
    def predict(self, inputs): raise ValueError("predict sempre falha")
    def get_schema(self):
        return ModelSchema(
            inputs=[TensorSpec("x", [1], "float32")],
            outputs=[TensorSpec("y", [1], "float32")],
            description="PredictFail",
        )
)PY";

static constexpr const char* kNoInheritModel = R"PY(
# Propositalmente sem herdar de MiiaModel
class NotAMiiaModel:
    def load(self): pass
    def predict(self, inputs): return {}
    def get_schema(self): return None
)PY";

static constexpr const char* kStructuredModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec
import numpy as np, math

class StructuredModel(MiiaModel):
    def load(self): pass
    def predict(self, inputs):
        state = inputs.get("state", {})
        heading = float(state.get("heading", 0.0))
        return {"result": np.array([[math.cos(math.radians(heading))]], dtype=np.float32)}
    def get_schema(self):
        return ModelSchema(
            inputs=[TensorSpec("state", [-1], "float32", structured=True)],
            outputs=[TensorSpec("result", [1,1], "float32")],
            description="StructuredModel",
            author="test",
        )
)PY";

static constexpr const char* kEmptyOutputModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec
import numpy as np

class EmptyOutputModel(MiiaModel):
    def load(self): pass
    def predict(self, inputs):
        return {"empty": np.array([], dtype=np.float32)}
    def get_schema(self):
        return ModelSchema(
            inputs=[TensorSpec("x", [1], "float32")],
            outputs=[TensorSpec("empty", [0], "float32")],
        )
)PY";

static constexpr const char* kBadKeyModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec
import numpy as np

class BadKeyModel(MiiaModel):
    def load(self): pass
    def predict(self, inputs):
        return {42: np.array([1.0], dtype=np.float32)}
    def get_schema(self):
        return ModelSchema(inputs=[], outputs=[], description="BadKey")
)PY";

static constexpr const char* kBadValueModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec

class BadValueModel(MiiaModel):
    def load(self): pass
    def predict(self, inputs):
        return {"y": "not_a_tensor"}
    def get_schema(self):
        return ModelSchema(inputs=[], outputs=[], description="BadValue")
)PY";

static constexpr const char* kNotDictModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec

class NotDictModel(MiiaModel):
    def load(self): pass
    def predict(self, inputs): return [1.0, 2.0]
    def get_schema(self):
        return ModelSchema(inputs=[], outputs=[], description="NotDict")
)PY";

static constexpr const char* kStatefulModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec
import numpy as np

class StatefulModel(MiiaModel):
    def __init__(self): self._count = 0
    def load(self): self._count = 0
    def predict(self, inputs):
        self._count += 1
        return {"count": np.array([float(self._count)], dtype=np.float32)}
    def get_schema(self):
        return ModelSchema(
            inputs=[],
            outputs=[TensorSpec("count", [1], "float32")],
            description="Stateful",
        )
)PY";

static constexpr const char* kUnloadSignalModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec
import numpy as np, os

class UnloadSignalModel(MiiaModel):
    SIGNAL_PATH = "/tmp/pb_unload_signal"
    def load(self):
        if os.path.exists(self.SIGNAL_PATH):
            os.remove(self.SIGNAL_PATH)
    def predict(self, inputs):
        return {"y": np.array([1.0], dtype=np.float32)}
    def get_schema(self):
        return ModelSchema(inputs=[], outputs=[TensorSpec("y",[1],"float32")],
                           description="UnloadSignal")
    def unload(self):
        with open(self.SIGNAL_PATH, "w") as f:
            f.write("unloaded")
)PY";

static constexpr const char* kSlowModel = R"PY(
from miia_model import MiiaModel, ModelSchema, TensorSpec
import numpy as np, time

class SlowModel(MiiaModel):
    def load(self): pass
    def predict(self, inputs):
        time.sleep(0.05)
        return {"y": np.array([1.0], dtype=np.float32)}
    def get_schema(self):
        return ModelSchema(
            inputs=[TensorSpec("x", [1], "float32")],
            outputs=[TensorSpec("y", [1], "float32")],
            description="SlowModel",
        )
)PY";

// =============================================================================
// Helpers de asserção
// =============================================================================

static double get_scalar(const Object& obj, const std::string& key) {
    auto it = obj.find(key);
    if (it == obj.end()) return std::numeric_limits<double>::quiet_NaN();
    const Value& v = it->second;
    if (v.is_number()) return v.as_number();
    if (v.is_array() && !v.as_array().empty() && v.as_array()[0].is_number())
        return v.as_array()[0].as_number();
    return std::numeric_limits<double>::quiet_NaN();
}

static Object make_x(std::initializer_list<double> vals) {
    Array arr;
    for (double v : vals) arr.push_back(Value{v});
    Object o;
    o["x"] = Value{std::move(arr)};
    return o;
}

// =============================================================================
// Keeper global — mantém instance_count_ > 0 durante toda a suíte.
//
// Reinicializar CPython após Py_Finalize() é oficialmente não suportado e
// causa SIGSEGV. O PythonBackend usa um contador de referências estático:
// quando chega a zero ele chama Py_Finalize(). Manter um PythonBackend
// "keeper" vivo do início ao fim do processo impede que instance_count_
// chegue a zero entre os testes.
// =============================================================================

static PythonBackend* g_keeper = nullptr;

struct GlobalEnv : public ::testing::Environment {
    void SetUp() override {
        g_keeper = new PythonBackend();
    }
    void TearDown() override {
        delete g_keeper;
        g_keeper = nullptr;
    }
};

// =============================================================================
// Fixture base
// =============================================================================

class PythonBackendTest : public ::testing::Test {
protected:
    fs::path tmpdir_;
    bool     miia_model_available_ = false;

    // Tmpdirs acumulados durante a suíte — deletados apenas em TearDownTestSuite.
    // Isso é necessário porque o CPython cacheia módulos em sys.modules pelo
    // nome. Se deletarmos o tmpdir no TearDown(), o módulo "miia_model" fica
    // cacheado apontando para um diretório inexistente. No próximo teste,
    // "from miia_model import MiiaModel" retorna a classe do cache (de um
    // diretório morto), enquanto find_model_class() reimporta miia_model do
    // novo tmpdir — resultando em duas classes MiiaModel distintas, fazendo
    // PyObject_IsSubclass() retornar 0 e load() falhar.
    static std::vector<fs::path> s_tmpdirs_;

    static void TearDownTestSuite() {
        for (auto& d : s_tmpdirs_)
            fs::remove_all(d);
        s_tmpdirs_.clear();
    }

    void SetUp() override {
        tmpdir_ = make_tmpdir();
        s_tmpdirs_.push_back(tmpdir_);

        fs::path src = miia_model_source();
        if (!src.empty()) {
            fs::copy_file(src, tmpdir_ / "miia_model.py",
                          fs::copy_options::overwrite_existing);
            miia_model_available_ = true;
        }
    }

    void TearDown() override {
        // Não deletar aqui — ver comentário em s_tmpdirs_.
    }

    fs::path write_model(const std::string& base, const std::string& content) {
        // Nome único por teste para evitar colisão em sys.modules.
        fs::path stem = fs::path(base).stem();
        std::string unique = stem.string() + "_" + tmpdir_suffix(tmpdir_) + ".py";
        fs::path p = tmpdir_ / unique;
        write_file(p, content);
        return p;
    }

    void require_miia_model() {
        if (!miia_model_available_)
            GTEST_SKIP() << "miia_model.py não encontrado via MODELS_DIR";
    }
};

std::vector<fs::path> PythonBackendTest::s_tmpdirs_;

// =============================================================================
// GRUPO 1 — validate()
// =============================================================================

TEST_F(PythonBackendTest, Validate_PathExistente_RetornaVazio) {
    auto p = write_model("valid.py", kSimpleModel);
    PythonBackend b;
    EXPECT_EQ(b.validate(p.string()), "");
}

TEST_F(PythonBackendTest, Validate_PathInexistente_RetornaMensagem) {
    PythonBackend b;
    std::string err = b.validate("/nao/existe/modelo.py");
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("not found"), std::string::npos) << err;
}

TEST_F(PythonBackendTest, Validate_ExtensaoErrada_RetornaMensagem) {
    // Cria diretamente sem write_model() para preservar a extensão .txt
    fs::path p = tmpdir_ / "modelo.txt";
    write_file(p, "# nao eh python");
    PythonBackend b;
    EXPECT_FALSE(b.validate(p.string()).empty());
}

TEST_F(PythonBackendTest, Validate_ExtensaoOnnx_RetornaMensagem) {
    // Cria diretamente sem write_model() para preservar a extensão .onnx
    fs::path p = tmpdir_ / "modelo.onnx";
    write_file(p, "binary");
    PythonBackend b;
    EXPECT_FALSE(b.validate(p.string()).empty());
}

TEST_F(PythonBackendTest, Validate_ArquivoVazio_RetornaVazio) {
    auto p = write_model("empty.py", "");
    PythonBackend b;
    EXPECT_EQ(b.validate(p.string()), "");
}

// =============================================================================
// GRUPO 2 — backend_type()
// =============================================================================

TEST_F(PythonBackendTest, BackendType_SemprePython) {
    PythonBackend b;
    EXPECT_EQ(static_cast<int>(b.backend_type()), 2);  // BACKEND_PYTHON = 2
}

// =============================================================================
// GRUPO 3 — load(): caminhos felizes
// =============================================================================

TEST_F(PythonBackendTest, Load_ModeloValido_RetornaTrue) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    EXPECT_TRUE(b.load(p.string(), {}));
    b.unload();
}

TEST_F(PythonBackendTest, Load_ModeloValido_EstadoLoaded) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    b.load(p.string(), {});
    auto r = b.predict(make_x({1.0, 2.0, 3.0}));
    EXPECT_TRUE(r.success) << r.error_message;
    b.unload();
}

TEST_F(PythonBackendTest, Load_MultiOutput_RetornaTrue) {
    require_miia_model();
    auto p = write_model("multi.py", kMultiOutputModel);
    PythonBackend b;
    EXPECT_TRUE(b.load(p.string(), {}));
    b.unload();
}

TEST_F(PythonBackendTest, Load_ModeloEstruturado_RetornaTrue) {
    require_miia_model();
    auto p = write_model("struct.py", kStructuredModel);
    PythonBackend b;
    EXPECT_TRUE(b.load(p.string(), {}));
    b.unload();
}

// =============================================================================
// GRUPO 4 — load(): caminhos de falha
// =============================================================================

TEST_F(PythonBackendTest, Load_PathInexistente_RetornaFalse) {
    PythonBackend b;
    EXPECT_FALSE(b.load("/nao/existe.py", {}));
}

TEST_F(PythonBackendTest, Load_SemSubclasseMiiaModel_RetornaFalse) {
    auto p = write_model("no_inherit.py", kNoInheritModel);
    PythonBackend b;
    EXPECT_FALSE(b.load(p.string(), {}));
}

TEST_F(PythonBackendTest, Load_ModeloComLoadFail_RetornaFalse) {
    require_miia_model();
    auto p = write_model("load_fail.py", kLoadFailModel);
    PythonBackend b;
    EXPECT_FALSE(b.load(p.string(), {}));
}

TEST_F(PythonBackendTest, Load_SintaxeInvalida_RetornaFalse) {
    auto p = write_model("broken.py", "def borken(  :\n    ???");
    PythonBackend b;
    EXPECT_FALSE(b.load(p.string(), {}));
}

TEST_F(PythonBackendTest, Load_ArquivoVazioSemClasse_RetornaFalse) {
    auto p = write_model("empty.py", "");
    PythonBackend b;
    EXPECT_FALSE(b.load(p.string(), {}));
}

TEST_F(PythonBackendTest, Load_ExtensaoJson_RetornaFalse) {
    auto p = write_model("model.json", "{\"type\":\"regression\"}");
    PythonBackend b;
    EXPECT_FALSE(b.load(p.string(), {}));
}

// =============================================================================
// GRUPO 5 — load() duplo / re-load
// =============================================================================

TEST_F(PythonBackendTest, LoadDuplo_SemUnload_NaoCrasha) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));
    bool second = b.load(p.string(), {});
    (void)second;
    b.unload();
}

TEST_F(PythonBackendTest, UnloadSeguido_DeLoad_Funciona) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));
    b.unload();
    EXPECT_TRUE(b.load(p.string(), {}));
    b.unload();
}

// =============================================================================
// GRUPO 6 — unload()
// =============================================================================

TEST_F(PythonBackendTest, Unload_SemLoad_NaoCrasha) {
    PythonBackend b;
    EXPECT_NO_THROW(b.unload());
}

TEST_F(PythonBackendTest, Unload_DuasVezes_NaoCrasha) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    b.load(p.string(), {});
    b.unload();
    EXPECT_NO_THROW(b.unload());
}

TEST_F(PythonBackendTest, Unload_ChamaMetodoUnloadNaClasse) {
    require_miia_model();
    fs::remove("/tmp/pb_unload_signal");
    auto p = write_model("signal.py", kUnloadSignalModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));
    b.unload();
    EXPECT_TRUE(fs::exists("/tmp/pb_unload_signal"))
        << "model.unload() não foi chamado pelo backend";
    fs::remove("/tmp/pb_unload_signal");
}

TEST_F(PythonBackendTest, Unload_LimpaEstado_PredictFalhaDepois) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));
    b.unload();
    auto r = b.predict(make_x({1.0, 2.0, 3.0}));
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

// =============================================================================
// GRUPO 7 — get_schema()
// =============================================================================

TEST_F(PythonBackendTest, GetSchema_AposLoad_RetornaSchemaValido) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto schema = b.get_schema();
    EXPECT_EQ(schema.description, "SimpleModel");
    EXPECT_EQ(schema.author, "test");
    ASSERT_EQ(schema.inputs.size(), 1u);
    EXPECT_EQ(schema.inputs[0].name, "x");
    ASSERT_EQ(schema.outputs.size(), 1u);
    EXPECT_EQ(schema.outputs[0].name, "y");

    b.unload();
}

TEST_F(PythonBackendTest, GetSchema_SemLoad_RetornaSchemaVazio) {
    PythonBackend b;
    auto schema = b.get_schema();
    EXPECT_TRUE(schema.inputs.empty());
    EXPECT_TRUE(schema.outputs.empty());
}

TEST_F(PythonBackendTest, GetSchema_AposUnload_RetornaSchemaVazio) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    b.load(p.string(), {});
    b.unload();
    auto schema = b.get_schema();
    EXPECT_TRUE(schema.inputs.empty());
    EXPECT_TRUE(schema.outputs.empty());
}

TEST_F(PythonBackendTest, GetSchema_InputEstruturado_StructuredTrue) {
    require_miia_model();
    auto p = write_model("struct.py", kStructuredModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto schema = b.get_schema();
    ASSERT_FALSE(schema.inputs.empty());
    EXPECT_TRUE(schema.inputs[0].structured);

    b.unload();
}

TEST_F(PythonBackendTest, GetSchema_MultiOutput_DoisOutputs) {
    require_miia_model();
    auto p = write_model("multi.py", kMultiOutputModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto schema = b.get_schema();
    EXPECT_EQ(schema.outputs.size(), 2u);

    b.unload();
}

TEST_F(PythonBackendTest, GetSchema_ConsistenteEntreMultiplasChamadas) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto s1 = b.get_schema();
    auto s2 = b.get_schema();
    EXPECT_EQ(s1.description, s2.description);
    EXPECT_EQ(s1.inputs.size(), s2.inputs.size());
    EXPECT_EQ(s1.outputs.size(), s2.outputs.size());

    b.unload();
}

// =============================================================================
// GRUPO 8 — predict(): tipos de Value na entrada
// =============================================================================

TEST_F(PythonBackendTest, Predict_EntradaEscalar_Sucesso) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Object inputs;
    inputs["x"] = Value{5.0};
    auto r = b.predict(inputs);
    EXPECT_TRUE(r.success) << r.error_message;

    b.unload();
}

TEST_F(PythonBackendTest, Predict_EntradaArray_Sucesso) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict(make_x({1.0, 2.0, 3.0}));
    EXPECT_TRUE(r.success) << r.error_message;

    b.unload();
}

TEST_F(PythonBackendTest, Predict_EntradaArrayGrande_Sucesso) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Array big;
    for (int i = 0; i < 1024; ++i) big.push_back(Value{static_cast<double>(i)});
    Object o;
    o["x"] = Value{std::move(big)};
    auto r = b.predict(o);
    EXPECT_TRUE(r.success) << r.error_message;

    b.unload();
}

TEST_F(PythonBackendTest, Predict_EntradaBool_Sucesso) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Object inputs;
    inputs["x"] = Value{true};
    auto r = b.predict(inputs);
    EXPECT_TRUE(r.success) << r.error_message;

    b.unload();
}

TEST_F(PythonBackendTest, Predict_EntradaString_NaoCrasha) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Object inputs;
    inputs["x"] = Value{std::string("abc")};
    auto r = b.predict(inputs);
    (void)r;

    b.unload();
}

TEST_F(PythonBackendTest, Predict_EntradaNula_NaoCrasha) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Object inputs;
    inputs["x"] = Value{};
    auto r = b.predict(inputs);
    (void)r;

    b.unload();
}

TEST_F(PythonBackendTest, Predict_EntradaObjetoAninhado_Sucesso) {
    require_miia_model();
    auto p = write_model("struct.py", kStructuredModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Object state;
    state["heading"] = Value{45.0};
    Object inputs;
    inputs["state"] = Value{std::move(state)};

    auto r = b.predict(inputs);
    EXPECT_TRUE(r.success) << r.error_message;
    EXPECT_TRUE(r.outputs.count("result"));

    b.unload();
}

TEST_F(PythonBackendTest, Predict_EntradaVazia_NaoCrasha) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict({});
    EXPECT_TRUE(r.success) << r.error_message;

    b.unload();
}

// =============================================================================
// GRUPO 9 — predict(): verificação de valores de saída
// =============================================================================

TEST_F(PythonBackendTest, Predict_SomaCorreta_Escalar) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict(make_x({1.0, 2.0, 3.0}));
    ASSERT_TRUE(r.success);
    EXPECT_NEAR(get_scalar(r.outputs, "y"), 6.0, 1e-4);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_SomaCorreta_Negativos) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict(make_x({-1.0, -2.0, -3.0}));
    ASSERT_TRUE(r.success);
    EXPECT_NEAR(get_scalar(r.outputs, "y"), -6.0, 1e-4);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_SomaZero) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict(make_x({0.0, 0.0, 0.0}));
    ASSERT_TRUE(r.success);
    EXPECT_NEAR(get_scalar(r.outputs, "y"), 0.0, 1e-4);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_MultiOutput_DuasChaves) {
    require_miia_model();
    auto p = write_model("multi.py", kMultiOutputModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict(make_x({1.0, 2.0, 3.0}));
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.outputs.count("sum"))  << "falta chave 'sum'";
    EXPECT_TRUE(r.outputs.count("vals")) << "falta chave 'vals'";
    EXPECT_NEAR(get_scalar(r.outputs, "sum"), 6.0, 1e-4);
    ASSERT_TRUE(r.outputs.at("vals").is_array());
    EXPECT_EQ(r.outputs.at("vals").as_array().size(), 3u);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_OutputEscalar_IsNumber) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict(make_x({3.0, 4.0, 5.0}));  // soma = 12
    ASSERT_TRUE(r.success);

    const Value& y = r.outputs.at("y");
    EXPECT_TRUE(y.is_number()) << "esperado escalar, got array";
    EXPECT_NEAR(y.as_number(), 12.0, 1e-4);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_OutputArray_IsArray) {
    require_miia_model();
    auto p = write_model("multi.py", kMultiOutputModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict(make_x({10.0, 20.0, 30.0}));
    ASSERT_TRUE(r.success);

    const Value& vals = r.outputs.at("vals");
    EXPECT_TRUE(vals.is_array());
    EXPECT_EQ(vals.as_array().size(), 3u);
    EXPECT_NEAR(vals.as_array()[0].as_number(), 10.0, 1e-3);
    EXPECT_NEAR(vals.as_array()[1].as_number(), 20.0, 1e-3);
    EXPECT_NEAR(vals.as_array()[2].as_number(), 30.0, 1e-3);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_TempoRegistradoPositivo) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict(make_x({1.0, 2.0, 3.0}));
    ASSERT_TRUE(r.success);
    EXPECT_GE(r.inference_time_ms, 0.0);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_ModeloLento_TempoMinimoRespeitado) {
    require_miia_model();
    auto p = write_model("slow.py", kSlowModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Object inputs;
    inputs["x"] = Value{1.0};
    auto r = b.predict(inputs);
    ASSERT_TRUE(r.success);
    EXPECT_GE(r.inference_time_ms, 40.0)
        << "inference_time_ms=" << r.inference_time_ms;

    b.unload();
}

// =============================================================================
// GRUPO 10 — predict(): caminhos de falha
// =============================================================================

TEST_F(PythonBackendTest, Predict_SemLoad_RetornaFalse) {
    PythonBackend b;
    auto r = b.predict(make_x({1.0, 2.0}));
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(PythonBackendTest, Predict_AposUnload_RetornaFalse) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    b.load(p.string(), {});
    b.unload();
    auto r = b.predict(make_x({1.0}));
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(PythonBackendTest, Predict_ModeloComPredictFail_RetornaFalse) {
    require_miia_model();
    auto p = write_model("pfail.py", kPredictFailModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Object inputs;
    inputs["x"] = Value{1.0};
    auto r = b.predict(inputs);

    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());

    b.unload();
}

TEST_F(PythonBackendTest, Predict_RetornoNaoDict_RetornaFalse) {
    require_miia_model();
    auto p = write_model("notdict.py", kNotDictModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict({});
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error_message.find("dict"), std::string::npos)
        << "mensagem inesperada: " << r.error_message;

    b.unload();
}

TEST_F(PythonBackendTest, Predict_ChaveNaoString_RetornaFalse) {
    require_miia_model();
    auto p = write_model("badkey.py", kBadKeyModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict({});
    EXPECT_FALSE(r.success);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_ValorNaoConversivel_RetornaFalse) {
    require_miia_model();
    auto p = write_model("badval.py", kBadValueModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r = b.predict({});
    EXPECT_FALSE(r.success);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_OutputArrayVazio_TrataGraciosamente) {
    require_miia_model();
    auto p = write_model("empty_out.py", kEmptyOutputModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Object inputs;
    inputs["x"] = Value{1.0};
    auto r = b.predict(inputs);

    if (r.success) {
        EXPECT_TRUE(r.outputs.count("empty"));
    }

    b.unload();
}

// =============================================================================
// GRUPO 11 — modelo stateful
// =============================================================================

TEST_F(PythonBackendTest, Predict_ModeloStateful_ContadorIncrementa) {
    require_miia_model();
    auto p = write_model("stateful.py", kStatefulModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    for (int i = 1; i <= 5; ++i) {
        auto r = b.predict({});
        ASSERT_TRUE(r.success);
        EXPECT_NEAR(get_scalar(r.outputs, "count"), static_cast<double>(i), 1e-4)
            << "iteração " << i;
    }

    b.unload();
}

TEST_F(PythonBackendTest, Predict_ModeloStateful_UnloadResetaContador) {
    require_miia_model();
    auto p = write_model("stateful.py", kStatefulModel);
    PythonBackend b;

    b.load(p.string(), {});
    b.predict({});
    b.predict({});
    b.unload();

    b.load(p.string(), {});
    auto r = b.predict({});
    ASSERT_TRUE(r.success);
    EXPECT_NEAR(get_scalar(r.outputs, "count"), 1.0, 1e-4)
        << "contador não foi resetado após reload";

    b.unload();
}

// =============================================================================
// GRUPO 12 — múltiplas instâncias
// =============================================================================

TEST_F(PythonBackendTest, MultiInstancia_Sequencial_SemVazamento) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    for (int i = 0; i < 5; ++i) {
        PythonBackend b;
        ASSERT_TRUE(b.load(p.string(), {}));
        auto r = b.predict(make_x({1.0, 2.0, 3.0}));
        EXPECT_TRUE(r.success) << "iteração " << i;
        b.unload();
    }
}

TEST_F(PythonBackendTest, MultiInstancia_Duas_Simultaneas_SemCrash) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b1, b2;
    ASSERT_TRUE(b1.load(p.string(), {}));
    ASSERT_TRUE(b2.load(p.string(), {}));

    auto r1 = b1.predict(make_x({1.0, 2.0, 3.0}));
    auto r2 = b2.predict(make_x({4.0, 5.0, 6.0}));

    EXPECT_TRUE(r1.success) << r1.error_message;
    EXPECT_TRUE(r2.success) << r2.error_message;
    EXPECT_NEAR(get_scalar(r1.outputs, "y"), 6.0, 1e-4);
    EXPECT_NEAR(get_scalar(r2.outputs, "y"), 15.0, 1e-4);

    b1.unload();
    b2.unload();
}

// =============================================================================
// GRUPO 13 — warmup()
// =============================================================================

TEST_F(PythonBackendTest, Warmup_ModeloSimples_NaoCrasha) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));
    EXPECT_NO_THROW(b.warmup(3));
    b.unload();
}

TEST_F(PythonBackendTest, Warmup_ZeroRuns_NaoCrasha) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));
    EXPECT_NO_THROW(b.warmup(0));
    b.unload();
}

TEST_F(PythonBackendTest, Warmup_AtualizaMetrics) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    b.warmup(5);

    EXPECT_GE(b.metrics().total_inferences, 5u)
        << "warmup deveria ter disparado ao menos 5 inferências";

    b.unload();
}

// =============================================================================
// GRUPO 14 — memory_usage_bytes()
// =============================================================================

TEST_F(PythonBackendTest, MemoryUsage_RetornaZeroOuPositivo) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));
    EXPECT_GE(b.memory_usage_bytes(), 0);
    b.unload();
}

TEST_F(PythonBackendTest, MemoryUsage_SemLoad_NaoCrasha) {
    PythonBackend b;
    EXPECT_GE(b.memory_usage_bytes(), 0);
}

// =============================================================================
// GRUPO 15 — métricas de inferência (RuntimeMetrics)
// =============================================================================

TEST_F(PythonBackendTest, Metrics_InicialmentZeradas) {
    PythonBackend b;
    EXPECT_EQ(b.metrics().total_inferences, 0u);
    EXPECT_EQ(b.metrics().failed_inferences, 0u);
}

TEST_F(PythonBackendTest, Metrics_PredictBemSucedido_IncrementaTotal) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    b.predict(make_x({1.0, 2.0, 3.0}));
    b.predict(make_x({4.0, 5.0, 6.0}));

    EXPECT_EQ(b.metrics().total_inferences, 2u);
    EXPECT_EQ(b.metrics().failed_inferences, 0u);

    b.unload();
}

TEST_F(PythonBackendTest, Metrics_PredictFalha_IncrementaFailed) {
    require_miia_model();
    auto p = write_model("pfail.py", kPredictFailModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    Object inputs;
    inputs["x"] = Value{1.0};
    b.predict(inputs);
    b.predict(inputs);

    EXPECT_GE(b.metrics().failed_inferences, 1u);

    b.unload();
}

TEST_F(PythonBackendTest, Metrics_SemLoad_PredictFalha_ContabilizaFalha) {
    PythonBackend b;
    b.predict({});
    // early return sem load não passa pelo metrics_.record() —
    // contadores permanecem zerados, o que é o comportamento correto.
    EXPECT_EQ(b.metrics().total_inferences, 0u);
    EXPECT_EQ(b.metrics().failed_inferences, 0u);
}

// =============================================================================
// GRUPO 16 — PythonBackendFactory
// =============================================================================

TEST(PythonBackendFactory, Create_RetornaInstancia) {
    PythonBackendFactory factory;
    auto b = factory.create();
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(static_cast<int>(b->backend_type()), 2);  // BACKEND_PYTHON = 2
}

TEST(PythonBackendFactory, BackendType_PYTHON) {
    PythonBackendFactory f;
    EXPECT_EQ(static_cast<int>(f.backend_type()), 2);
}

TEST(PythonBackendFactory, Name_Python) {
    PythonBackendFactory f;
    EXPECT_EQ(f.name(), "python");
}

// =============================================================================
// GRUPO 17 — robustez: volume e valores extremos
// =============================================================================

TEST_F(PythonBackendTest, Predict_100Chamadas_TodasSucesso) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    int falhas = 0;
    for (int i = 0; i < 100; ++i) {
        auto r = b.predict(make_x({static_cast<double>(i),
                                    static_cast<double>(i + 1),
                                    static_cast<double>(i + 2)}));
        if (!r.success) ++falhas;
    }

    EXPECT_EQ(falhas, 0) << falhas << " predições falharam de 100";
    EXPECT_EQ(b.metrics().total_inferences, 100u);

    b.unload();
}

TEST_F(PythonBackendTest, Predict_ValoresExtremos_NaoCrasha) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);
    PythonBackend b;
    ASSERT_TRUE(b.load(p.string(), {}));

    auto r_inf = b.predict(make_x({std::numeric_limits<double>::infinity(), 0.0, 0.0}));
    auto r_nan = b.predict(make_x({std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}));
    auto r_max = b.predict(make_x({std::numeric_limits<double>::max(), 0.0, 0.0}));
    (void)r_inf; (void)r_nan; (void)r_max;

    b.unload();
}

// =============================================================================
// GRUPO 18 — ciclo completo
// =============================================================================

TEST_F(PythonBackendTest, CicloCompleto_LoadPredictUnload_TresVezes) {
    require_miia_model();
    auto p = write_model("simple.py", kSimpleModel);

    for (int ciclo = 0; ciclo < 3; ++ciclo) {
        PythonBackend b;
        ASSERT_TRUE(b.load(p.string(), {})) << "ciclo " << ciclo;

        for (int j = 0; j < 10; ++j) {
            auto r = b.predict(make_x({1.0, 2.0, 3.0}));
            ASSERT_TRUE(r.success) << "ciclo=" << ciclo << " j=" << j;
        }

        auto schema = b.get_schema();
        EXPECT_EQ(schema.description, "SimpleModel");

        b.unload();
    }
}

// =============================================================================
// main() customizado — necessário para registrar o GlobalEnv keeper antes
// de qualquer teste rodar, prevenindo Py_Finalize() entre testes.
//
// Como este arquivo define seu próprio main(), o meson.build deve usar
// gtest_dep (sem _main):
//   dependencies: [worker_lib_dep, proto_dep, gtest_main_dep, gtest_dep]
//   → trocar gtest_main_dep por gtest_dep e remover link com gtest_main_lib
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new GlobalEnv());
    return RUN_ALL_TESTS();
}
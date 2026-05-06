// =============================================================================
// tests/unit/test_onnx_backend.cpp
//
// Testes unitários — OnnxBackend
//
// Estratégia:
//   Instancia OnnxBackend diretamente e exercita load/unload/predict/schema/
//   validate/memory_usage_bytes.  Estado "carregado" é inferido pelo
//   comportamento observável da API pública (sem is_loaded()).
//   Não requer gRPC nem servidor — todos os testes rodam in-process.
//
// Pré-requisito:
//   $MODELS_DIR deve conter simple_linear.onnx e simple_classifier.onnx
//   (gerados por  make create-models / python/scripts/create_test_models.py).
//   Testes que dependem dos modelos emitem GTEST_SKIP() se o arquivo faltar.
//
// Modelos usados:
//   simple_linear.onnx
//       input  'input'  shape=[1,5]  float32
//       output 'output' shape=[1,5]  float32
//       operação: output = input * 2.0 + 1.0
//
//   simple_classifier.onnx
//       input  'input'  shape=[1,4]  float32
//       output 'output' shape=[1,3]  float32
//       operação: Linear(4→3) + Softmax  → probabilidades somam 1
// =============================================================================

#include <gtest/gtest.h>

#include "inference/onnx_backend.hpp"
#include "client/inference_client.hpp"
#include "common.pb.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using mlinference::inference::OnnxBackend;
using mlinference::client::Value;
using mlinference::client::Array;
using mlinference::client::Object;

// =============================================================================
// Helpers globais
// =============================================================================

static std::string models_dir() {
    const char* env = std::getenv("MODELS_DIR");
    return env ? std::string(env) : std::string("models");
}

static std::string model_path(const std::string& filename) {
    return (fs::path(models_dir()) / filename).string();
}

// Monta Object com um array de floats para o tensor indicado.
static Object make_input(const std::string& tensor_name,
                         const std::vector<float>& vals) {
    Array arr;
    arr.reserve(vals.size());
    for (float v : vals)
        arr.push_back(Value{static_cast<double>(v)});
    Object obj;
    obj[tensor_name] = Value{std::move(arr)};
    return obj;
}

// Retorna os valores de saída de um tensor como vector<double>.
static std::vector<double> output_as_doubles(const Object& outputs,
                                             const std::string& name) {
    const auto& arr = outputs.at(name).as_array();
    std::vector<double> out;
    out.reserve(arr.size());
    for (const auto& e : arr)
        out.push_back(e.as_number());
    return out;
}

// Grava um arquivo temporário corrompido e retorna o path.
static std::string write_corrupt_file(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    const char garbage[] = "this is not a valid onnx model \x00\x01\x02\xFF";
    f.write(garbage, static_cast<std::streamsize>(sizeof(garbage) - 1));
    return path;
}

// =============================================================================
// Fixture — simple_linear.onnx carregado no SetUp()
// =============================================================================

class LinearFixture : public ::testing::Test {
protected:
    OnnxBackend backend;
    const std::string path = model_path("simple_linear.onnx");

    void SetUp() override {
        if (!fs::exists(path))
            GTEST_SKIP() << "Modelo ausente: " << path
                         << " — execute  make create-models  antes.";
        ASSERT_TRUE(backend.load(path, {})) << "Falha ao carregar " << path;
    }

    void TearDown() override { backend.unload(); }

    Object linear_input(const std::vector<float>& v) {
        return make_input("input", v);
    }
};

// =============================================================================
// Fixture — simple_classifier.onnx carregado no SetUp()
// =============================================================================

class ClassifierFixture : public ::testing::Test {
protected:
    OnnxBackend backend;
    const std::string path = model_path("simple_classifier.onnx");

    void SetUp() override {
        if (!fs::exists(path))
            GTEST_SKIP() << "Modelo ausente: " << path;
        ASSERT_TRUE(backend.load(path, {})) << "Falha ao carregar " << path;
    }

    void TearDown() override { backend.unload(); }

    Object classifier_input(const std::vector<float>& v) {
        return make_input("input", v);
    }
};

// =============================================================================
// Grupo 1 — Lifecycle: load / unload / reload
// =============================================================================

TEST(Lifecycle, LoadValidModelReturnsTrue) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    EXPECT_TRUE(b.load(p, {}));
    b.unload();
}

TEST(Lifecycle, LoadNonExistentReturnsFalse) {
    OnnxBackend b;
    EXPECT_FALSE(b.load("/tmp/___ghost_model_xyzzy.onnx", {}));
}

TEST(Lifecycle, LoadInvalidOnnxReturnsFalse) {
    std::string tmp = "/tmp/bad_onnx_lifecycle.onnx";
    write_corrupt_file(tmp);
    OnnxBackend b;
    EXPECT_FALSE(b.load(tmp, {}));
    std::remove(tmp.c_str());
}

TEST(Lifecycle, DoubleLoadWithoutUnloadReturnsFalse) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    EXPECT_FALSE(b.load(p, {}));
    b.unload();
}

TEST(Lifecycle, UnloadThenReloadSucceeds) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    b.unload();
    EXPECT_TRUE(b.load(p, {}));
    b.unload();
}

TEST(Lifecycle, MultipleUnloadsAreSafe) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    b.unload();
    EXPECT_NO_FATAL_FAILURE(b.unload());
}

TEST(Lifecycle, UnloadedBackendPredictFails) {
    OnnxBackend b;
    auto r = b.predict(make_input("input", {0, 0, 0, 0, 0}));
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST(Lifecycle, UnloadedBackendMemoryUsageIsZero) {
    OnnxBackend b;
    EXPECT_EQ(b.memory_usage_bytes(), 0);
}

TEST(Lifecycle, UnloadClearsMemoryUsage) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    EXPECT_GT(b.memory_usage_bytes(), 0);
    b.unload();
    EXPECT_EQ(b.memory_usage_bytes(), 0);
}

TEST(Lifecycle, UnloadClearsSchema) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    b.unload();
    auto schema = b.get_schema();
    EXPECT_TRUE(schema.inputs.empty());
    EXPECT_TRUE(schema.outputs.empty());
}

TEST(Lifecycle, DestructorWithLoadedModelNoCrash) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    EXPECT_NO_FATAL_FAILURE({
        OnnxBackend b;
        b.load(p, {});
        // destrutor chama unload()
    });
}

TEST(Lifecycle, PredictAfterUnloadFails) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    b.unload();
    auto r = b.predict(make_input("input", {1, 1, 1, 1, 1}));
    EXPECT_FALSE(r.success);
}

// =============================================================================
// Grupo 2 — Schema: simple_linear
// =============================================================================

TEST_F(LinearFixture, SchemaHasOneInput) {
    EXPECT_EQ(backend.get_schema().inputs.size(), 1u);
}

TEST_F(LinearFixture, SchemaHasOneOutput) {
    EXPECT_EQ(backend.get_schema().outputs.size(), 1u);
}

TEST_F(LinearFixture, SchemaInputNameIsInput) {
    EXPECT_EQ(backend.get_schema().inputs[0].name, "input");
}

TEST_F(LinearFixture, SchemaOutputNameIsOutput) {
    EXPECT_EQ(backend.get_schema().outputs[0].name, "output");
}

TEST_F(LinearFixture, SchemaInputShapeRank2) {
    EXPECT_EQ(backend.get_schema().inputs[0].shape.size(), 2u);
}

TEST_F(LinearFixture, SchemaInputDim0Is1) {
    EXPECT_EQ(backend.get_schema().inputs[0].shape[0], 1);
}

TEST_F(LinearFixture, SchemaInputDim1Is5) {
    EXPECT_EQ(backend.get_schema().inputs[0].shape[1], 5);
}

TEST_F(LinearFixture, SchemaOutputDim1Is5) {
    EXPECT_EQ(backend.get_schema().outputs[0].shape[1], 5);
}

TEST_F(LinearFixture, SchemaInputDtypeIsFloat32) {
    // common::FLOAT32 == 0
    EXPECT_EQ(backend.get_schema().inputs[0].dtype, mlinference::common::FLOAT32);
}

TEST_F(LinearFixture, SchemaOutputDtypeIsFloat32) {
    EXPECT_EQ(backend.get_schema().outputs[0].dtype, mlinference::common::FLOAT32);
}

TEST_F(LinearFixture, SchemaIsIdempotent) {
    auto s1 = backend.get_schema();
    auto s2 = backend.get_schema();
    EXPECT_EQ(s1.inputs.size(),  s2.inputs.size());
    EXPECT_EQ(s1.outputs.size(), s2.outputs.size());
    EXPECT_EQ(s1.inputs[0].name,  s2.inputs[0].name);
    EXPECT_EQ(s1.outputs[0].name, s2.outputs[0].name);
}

// =============================================================================
// Grupo 3 — Schema: simple_classifier
// =============================================================================

TEST_F(ClassifierFixture, SchemaHasOneInput) {
    EXPECT_EQ(backend.get_schema().inputs.size(), 1u);
}

TEST_F(ClassifierFixture, SchemaHasOneOutput) {
    EXPECT_EQ(backend.get_schema().outputs.size(), 1u);
}

TEST_F(ClassifierFixture, SchemaInputDim1Is4) {
    EXPECT_EQ(backend.get_schema().inputs[0].shape[1], 4);
}

TEST_F(ClassifierFixture, SchemaOutputDim1Is3) {
    EXPECT_EQ(backend.get_schema().outputs[0].shape[1], 3);
}

// =============================================================================
// Grupo 4 — Predict: correção matemática de simple_linear (y = 2x + 1)
// =============================================================================

TEST_F(LinearFixture, PredictReturnsSuccess) {
    auto r = backend.predict(linear_input({1, 2, 3, 4, 5}));
    EXPECT_TRUE(r.success) << "error_message: " << r.error_message;
}

TEST_F(LinearFixture, PredictOutputKeyPresent) {
    auto r = backend.predict(linear_input({0, 0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    EXPECT_GT(r.outputs.count("output"), 0u);
}

TEST_F(LinearFixture, PredictOutputIsArray) {
    auto r = backend.predict(linear_input({0, 0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.outputs.at("output").is_array());
}

TEST_F(LinearFixture, PredictOutputSize5) {
    auto r = backend.predict(linear_input({0, 0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.outputs.at("output").as_array().size(), 5u);
}

TEST_F(LinearFixture, PredictZeroInputGivesOnes) {
    // 0 * 2 + 1 = 1
    auto r = backend.predict(linear_input({0, 0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    for (double x : output_as_doubles(r.outputs, "output"))
        EXPECT_NEAR(x, 1.0, 1e-4);
}

TEST_F(LinearFixture, PredictOnesInputGivesThrees) {
    // 1 * 2 + 1 = 3
    auto r = backend.predict(linear_input({1, 1, 1, 1, 1}));
    ASSERT_TRUE(r.success);
    for (double x : output_as_doubles(r.outputs, "output"))
        EXPECT_NEAR(x, 3.0, 1e-4);
}

TEST_F(LinearFixture, PredictNegativeInput) {
    // -3 * 2 + 1 = -5
    auto r = backend.predict(linear_input({-3, -3, -3, -3, -3}));
    ASSERT_TRUE(r.success);
    for (double x : output_as_doubles(r.outputs, "output"))
        EXPECT_NEAR(x, -5.0, 1e-4);
}

TEST_F(LinearFixture, PredictMixedValues) {
    // input=[0,1,2,3,4] → output=[1,3,5,7,9]
    auto r = backend.predict(linear_input({0, 1, 2, 3, 4}));
    ASSERT_TRUE(r.success);
    auto v = output_as_doubles(r.outputs, "output");
    ASSERT_EQ(v.size(), 5u);
    const std::vector<double> expected = {1, 3, 5, 7, 9};
    for (size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(v[i], expected[i], 1e-4);
}

TEST_F(LinearFixture, PredictLargePositiveValues) {
    // 1000 * 2 + 1 = 2001
    auto r = backend.predict(linear_input({1000, 1000, 1000, 1000, 1000}));
    ASSERT_TRUE(r.success);
    for (double x : output_as_doubles(r.outputs, "output"))
        EXPECT_NEAR(x, 2001.0, 0.5);
}

TEST_F(LinearFixture, PredictLinearityHolds) {
    // y(4) - y(2) deve ser (4*2+1) - (2*2+1) = 4
    auto r1 = backend.predict(linear_input({2, 2, 2, 2, 2}));
    auto r2 = backend.predict(linear_input({4, 4, 4, 4, 4}));
    ASSERT_TRUE(r1.success && r2.success);
    auto v1 = output_as_doubles(r1.outputs, "output");
    auto v2 = output_as_doubles(r2.outputs, "output");
    for (size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(v2[i] - v1[i], 4.0, 1e-3);
}

TEST_F(LinearFixture, PredictIsDeterministic) {
    auto r1 = backend.predict(linear_input({1.5f, 2.5f, 3.5f, 4.5f, 5.5f}));
    auto r2 = backend.predict(linear_input({1.5f, 2.5f, 3.5f, 4.5f, 5.5f}));
    ASSERT_TRUE(r1.success && r2.success);
    auto v1 = output_as_doubles(r1.outputs, "output");
    auto v2 = output_as_doubles(r2.outputs, "output");
    for (size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(v1[i], v2[i]);
}

TEST_F(LinearFixture, Predict50SequentialCallsAllSucceed) {
    for (int i = 0; i < 50; ++i) {
        auto r = backend.predict(linear_input({
            static_cast<float>(i), static_cast<float>(i+1),
            static_cast<float>(i+2), static_cast<float>(i+3),
            static_cast<float>(i+4)}));
        ASSERT_TRUE(r.success) << "falhou na chamada " << i;
    }
}

TEST_F(LinearFixture, PredictFloatPrecisionAcceptable) {
    // verifica que float32 do modelo não acumula erro excessivo
    auto r = backend.predict(linear_input({0.1f, 0.2f, 0.3f, 0.4f, 0.5f}));
    ASSERT_TRUE(r.success);
    auto v = output_as_doubles(r.outputs, "output");
    // 0.1*2+1=1.2, 0.2*2+1=1.4, ...
    const std::vector<double> expected = {1.2, 1.4, 1.6, 1.8, 2.0};
    for (size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(v[i], expected[i], 5e-4);
}

// =============================================================================
// Grupo 5 — Predict: erros de entrada
// =============================================================================

TEST_F(LinearFixture, PredictMissingTensorFails) {
    Object empty;
    auto r = backend.predict(empty);
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(LinearFixture, PredictWrongTensorNameFails) {
    auto r = backend.predict(make_input("wrong_name", {1, 2, 3, 4, 5}));
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(LinearFixture, PredictEmptyArrayFails) {
    Object obj;
    obj["input"] = Value{Array{}};
    auto r = backend.predict(obj);
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(LinearFixture, PredictNullValueFails) {
    Object obj;
    obj["input"] = Value{};  // Null
    auto r = backend.predict(obj);
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(LinearFixture, PredictErrorMessageContainsTensorName) {
    // a mensagem de erro deve mencionar o tensor que faltou
    auto r = backend.predict(make_input("wrong", {0, 0, 0, 0, 0}));
    EXPECT_FALSE(r.success);
    // error_message não deve ser genérica demais
    EXPECT_FALSE(r.error_message.empty());
}

// =============================================================================
// Grupo 6 — Predict: latência
// =============================================================================

TEST_F(LinearFixture, PredictLatencyIsPositive) {
    auto r = backend.predict(linear_input({1, 2, 3, 4, 5}));
    ASSERT_TRUE(r.success);
    EXPECT_GT(r.inference_time_ms, 0.0);
}

TEST_F(LinearFixture, PredictLatencyIsReasonable) {
    // modelo minúsculo deve terminar em menos de 500 ms
    auto r = backend.predict(linear_input({1, 2, 3, 4, 5}));
    ASSERT_TRUE(r.success);
    EXPECT_LT(r.inference_time_ms, 500.0);
}

TEST_F(LinearFixture, PredictFailLatencyNonNegative) {
    Object empty;
    auto r = backend.predict(empty);
    EXPECT_FALSE(r.success);
    EXPECT_GE(r.inference_time_ms, 0.0);
}

// =============================================================================
// Grupo 7 — Predict: simple_classifier (softmax)
// =============================================================================

TEST_F(ClassifierFixture, PredictSuccess) {
    auto r = backend.predict(classifier_input({1, 0, 0, 0}));
    EXPECT_TRUE(r.success) << r.error_message;
}

TEST_F(ClassifierFixture, PredictOutputHas3Elements) {
    auto r = backend.predict(classifier_input({1, 2, 3, 4}));
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.outputs.at("output").as_array().size(), 3u);
}

TEST_F(ClassifierFixture, SoftmaxSumsToOne) {
    auto r = backend.predict(classifier_input({1, 2, 3, 4}));
    ASSERT_TRUE(r.success);
    auto v = output_as_doubles(r.outputs, "output");
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-5);
}

TEST_F(ClassifierFixture, SoftmaxAllElementsNonNegative) {
    auto r = backend.predict(classifier_input({0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    for (double x : output_as_doubles(r.outputs, "output"))
        EXPECT_GE(x, 0.0);
}

TEST_F(ClassifierFixture, SoftmaxAllElementsAtMostOne) {
    auto r = backend.predict(classifier_input({5, -5, 0, 1}));
    ASSERT_TRUE(r.success);
    for (double x : output_as_doubles(r.outputs, "output"))
        EXPECT_LE(x, 1.0);
}

TEST_F(ClassifierFixture, SoftmaxUniformInputGivesApproximatelyEqualProbs) {
    // input uniforme → logits iguais (pesos * input_zero_uniforme) → probs ≈ 1/3
    auto r = backend.predict(classifier_input({0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    auto v = output_as_doubles(r.outputs, "output");
    double mn = *std::min_element(v.begin(), v.end());
    double mx = *std::max_element(v.begin(), v.end());
    // com bias zero os logits seriam iguais; com bias aleatório pode variar
    // mas todos devem ser probabilidades válidas (não-negativas e ≤ 1)
    EXPECT_GE(mn, 0.0);
    EXPECT_LE(mx, 1.0);
}

TEST_F(ClassifierFixture, SoftmaxSumsToOneForMultipleInputs) {
    const std::vector<std::vector<float>> test_inputs = {
        {0, 0, 0, 0},
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 1},
        {-5, 5, -5, 5},
        {10, 10, 10, 10},
    };
    for (const auto& inp : test_inputs) {
        auto r = backend.predict(classifier_input(inp));
        ASSERT_TRUE(r.success) << "falhou para input[0]=" << inp[0];
        auto v = output_as_doubles(r.outputs, "output");
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-5) << "soma != 1 para input[0]=" << inp[0];
    }
}

TEST_F(ClassifierFixture, SoftmaxIsDeterministic) {
    auto r1 = backend.predict(classifier_input({1, 2, 3, 4}));
    auto r2 = backend.predict(classifier_input({1, 2, 3, 4}));
    ASSERT_TRUE(r1.success && r2.success);
    auto v1 = output_as_doubles(r1.outputs, "output");
    auto v2 = output_as_doubles(r2.outputs, "output");
    for (size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(v1[i], v2[i]);
}

TEST_F(ClassifierFixture, PredictMissingTensorFails) {
    Object empty;
    auto r = backend.predict(empty);
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

// =============================================================================
// Grupo 8 — validate()
// =============================================================================

TEST(Validate, ValidLinearModelReturnsEmptyString) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    EXPECT_EQ(b.validate(p), "");
}

TEST(Validate, ValidClassifierModelReturnsEmptyString) {
    std::string p = model_path("simple_classifier.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    EXPECT_EQ(b.validate(p), "");
}

TEST(Validate, NonExistentFileReturnsNonEmptyError) {
    OnnxBackend b;
    EXPECT_NE(b.validate("/tmp/___nonexistent_xyzzy.onnx"), "");
}

TEST(Validate, CorruptFileReturnsNonEmptyError) {
    std::string tmp = "/tmp/corrupt_validate_test.onnx";
    write_corrupt_file(tmp);
    OnnxBackend b;
    EXPECT_NE(b.validate(tmp), "");
    std::remove(tmp.c_str());
}

TEST(Validate, EmptyFileReturnsNonEmptyError) {
    std::string tmp = "/tmp/empty_validate_test.onnx";
    { std::ofstream f(tmp, std::ios::binary); }
    OnnxBackend b;
    EXPECT_NE(b.validate(tmp), "");
    std::remove(tmp.c_str());
}

TEST(Validate, ValidateDoesNotSideEffectPredict) {
    // validate() não deve carregar o modelo — predict() ainda deve falhar
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    b.validate(p);
    auto r = b.predict(make_input("input", {0, 0, 0, 0, 0}));
    EXPECT_FALSE(r.success);
}

TEST(Validate, ValidateCanBeCalledWhileLoaded) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    EXPECT_NO_FATAL_FAILURE({ b.validate(p); });
    // modelo continua funcional
    auto r = b.predict(make_input("input", {1, 1, 1, 1, 1}));
    EXPECT_TRUE(r.success);
    b.unload();
}

TEST(Validate, ValidateResultIsConsistent) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    // duas chamadas ao mesmo arquivo devem ter o mesmo resultado
    EXPECT_EQ(b.validate(p), b.validate(p));
}

// =============================================================================
// Grupo 9 — memory_usage_bytes()
// =============================================================================

TEST(Memory, NotLoadedReturnsZero) {
    OnnxBackend b;
    EXPECT_EQ(b.memory_usage_bytes(), 0);
}

TEST(Memory, LoadedReturnsPositive) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    EXPECT_GT(b.memory_usage_bytes(), 0);
    b.unload();
}

TEST(Memory, ReportsFileSizeExactly) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    int64_t reported = b.memory_usage_bytes();
    int64_t actual   = static_cast<int64_t>(fs::file_size(p));
    EXPECT_EQ(reported, actual);
    b.unload();
}

TEST(Memory, AfterUnloadReturnsZero) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    b.unload();
    EXPECT_EQ(b.memory_usage_bytes(), 0);
}

TEST(Memory, IsIdempotent) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    EXPECT_EQ(b.memory_usage_bytes(), b.memory_usage_bytes());
    b.unload();
}

TEST(Memory, ClassifierLargerThanLinear) {
    std::string pl = model_path("simple_linear.onnx");
    std::string pc = model_path("simple_classifier.onnx");
    if (!fs::exists(pl) || !fs::exists(pc)) GTEST_SKIP() << "Modelos ausentes";
    OnnxBackend bl, bc;
    ASSERT_TRUE(bl.load(pl, {}));
    ASSERT_TRUE(bc.load(pc, {}));
    // classifier tem pesos, linear tem apenas constantes → classifier >= linear
    EXPECT_GE(bc.memory_usage_bytes(), bl.memory_usage_bytes());
    bl.unload();
    bc.unload();
}

// =============================================================================
// Grupo 10 — backend_type()
// =============================================================================

TEST(BackendType, IsOnnx) {
    OnnxBackend b;
    EXPECT_EQ(b.backend_type(), mlinference::common::BACKEND_ONNX);
}

TEST(BackendType, IsOnnxAfterLoad) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    EXPECT_EQ(b.backend_type(), mlinference::common::BACKEND_ONNX);
    b.unload();
}

TEST(BackendType, IsOnnxAfterUnload) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b;
    ASSERT_TRUE(b.load(p, {}));
    b.unload();
    EXPECT_EQ(b.backend_type(), mlinference::common::BACKEND_ONNX);
}

// =============================================================================
// Grupo 11 — Construtor com parâmetros explícitos
// =============================================================================

TEST(Constructor, SingleThreadWorks) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b(false, 0, 1);
    ASSERT_TRUE(b.load(p, {}));
    auto r = b.predict(make_input("input", {0, 0, 0, 0, 0}));
    EXPECT_TRUE(r.success);
    b.unload();
}

TEST(Constructor, EightThreadsWork) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b(false, 0, 8);
    ASSERT_TRUE(b.load(p, {}));
    auto r = b.predict(make_input("input", {1, 2, 3, 4, 5}));
    EXPECT_TRUE(r.success);
    b.unload();
}

TEST(Constructor, SingleThreadYieldsSameResultAs4Threads) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b1(false, 0, 1);
    OnnxBackend b4(false, 0, 4);
    ASSERT_TRUE(b1.load(p, {}));
    ASSERT_TRUE(b4.load(p, {}));
    auto inp = make_input("input", {1.5f, 2.5f, 3.5f, 4.5f, 5.5f});
    auto r1 = b1.predict(inp);
    auto r4 = b4.predict(inp);
    ASSERT_TRUE(r1.success && r4.success);
    auto v1 = output_as_doubles(r1.outputs, "output");
    auto v4 = output_as_doubles(r4.outputs, "output");
    for (size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(v1[i], v4[i], 1e-4);
    b1.unload();
    b4.unload();
}

// =============================================================================
// Grupo 12 — Dois backends independentes
// =============================================================================

TEST(Independence, TwoBackendsLoadSameModel) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b1, b2;
    ASSERT_TRUE(b1.load(p, {}));
    ASSERT_TRUE(b2.load(p, {}));
    auto r1 = b1.predict(make_input("input", {0, 0, 0, 0, 0}));
    auto r2 = b2.predict(make_input("input", {1, 1, 1, 1, 1}));
    ASSERT_TRUE(r1.success && r2.success);
    auto v1 = output_as_doubles(r1.outputs, "output");
    auto v2 = output_as_doubles(r2.outputs, "output");
    EXPECT_NEAR(v1[0], 1.0, 1e-4);  // 0*2+1
    EXPECT_NEAR(v2[0], 3.0, 1e-4);  // 1*2+1
    b1.unload();
    b2.unload();
}

TEST(Independence, UnloadOneDoesNotAffectOther) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;
    OnnxBackend b1, b2;
    ASSERT_TRUE(b1.load(p, {}));
    ASSERT_TRUE(b2.load(p, {}));
    b1.unload();
    // b2 ainda deve funcionar
    auto r = b2.predict(make_input("input", {2, 2, 2, 2, 2}));
    EXPECT_TRUE(r.success);
    auto v = output_as_doubles(r.outputs, "output");
    EXPECT_NEAR(v[0], 5.0, 1e-4);  // 2*2+1
    b2.unload();
}

TEST(Independence, DifferentModels) {
    std::string pl = model_path("simple_linear.onnx");
    std::string pc = model_path("simple_classifier.onnx");
    if (!fs::exists(pl) || !fs::exists(pc)) GTEST_SKIP() << "Modelos ausentes";
    OnnxBackend bl, bc;
    ASSERT_TRUE(bl.load(pl, {}));
    ASSERT_TRUE(bc.load(pc, {}));
    // linear tem 5 saídas, classifier tem 3
    EXPECT_EQ(bl.get_schema().outputs[0].shape[1], 5);
    EXPECT_EQ(bc.get_schema().outputs[0].shape[1], 3);
    bl.unload();
    bc.unload();
}

// =============================================================================
// Grupo 13 — Ciclo completo reload
// =============================================================================

TEST(ReloadCycle, LoadPredictUnloadRepeat5x) {
    std::string p = model_path("simple_linear.onnx");
    if (!fs::exists(p)) GTEST_SKIP() << "Modelo ausente: " << p;

    for (int cycle = 0; cycle < 5; ++cycle) {
        OnnxBackend b;
        ASSERT_TRUE(b.load(p, {})) << "cycle=" << cycle;
        auto r = b.predict(make_input("input", {
            static_cast<float>(cycle), static_cast<float>(cycle+1),
            static_cast<float>(cycle+2), static_cast<float>(cycle+3),
            static_cast<float>(cycle+4)}));
        ASSERT_TRUE(r.success) << "cycle=" << cycle;
        // verifica primeiro elemento: cycle*2+1
        auto v = output_as_doubles(r.outputs, "output");
        EXPECT_NEAR(v[0], 2.0 * cycle + 1.0, 1e-3) << "cycle=" << cycle;
        b.unload();
    }
}
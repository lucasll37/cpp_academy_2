// =============================================================================
// tests/unit/test_inference_engine.cpp
//
// Testes unitários — InferenceEngine
//
// Testa o InferenceEngine diretamente, sem camada gRPC nem InferenceClient.
// Cada grupo isola um aspecto do comportamento do motor.
//
// GRUPO 1  — Construção e EngineInfo
// GRUPO 2  — load_model(): happy path
// GRUPO 3  — load_model(): falhas
// GRUPO 4  — unload_model()
// GRUPO 5  — is_model_loaded()
// GRUPO 6  — get_loaded_model_ids()
// GRUPO 7  — predict(): happy path e valores
// GRUPO 8  — predict(): falhas de entrada
// GRUPO 9  — batch_predict() (via predict em loop — InferenceEngine não expõe batch)
// GRUPO 10 — get_model_info()
// GRUPO 11 — get_model_metrics()
// GRUPO 12 — validate_model()
// GRUPO 13 — warmup_model()
// GRUPO 14 — Múltiplos modelos simultâneos
// GRUPO 15 — Ciclos load → predict → unload repetidos
// GRUPO 16 — Destrutor descarrega modelos ativos
// GRUPO 17 — Concorrência: leituras simultâneas
// GRUPO 18 — Ciclo completo end-to-end
//
// Pré-requisito:
//   $MODELS_DIR deve conter simple_linear.onnx e simple_classifier.onnx.
//   Testes que dependem dos arquivos emitem GTEST_SKIP() se o modelo faltar.
//
// Modelos usados:
//   simple_linear.onnx     : output = input * 2 + 1   (input/output [1,5])
//   simple_classifier.onnx : Linear(4→3) + Softmax    (output soma ≈ 1)
//
// Entrada no tests/unit/meson.build:
//   test('unit_inference_engine',
//       executable('test_unit_inference_engine',
//           'test_inference_engine.cpp',
//           include_directories: [inference_inc, client_inc],
//           dependencies: [worker_lib_dep, proto_dep, gtest_main_dep, gtest_dep],
//           install: false,
//       ),
//       suite:       'unit',
//       timeout:     120,
//       is_parallel: false,
//       env:         _integration_env,
//   )
// =============================================================================

#include <gtest/gtest.h>

#include "inference/inference_engine.hpp"
#include "client/inference_client.hpp"
#include "common.pb.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using miia::inference::InferenceEngine;
using miia::inference::InferenceResult;
using miia::client::Value;
using miia::client::Array;
using miia::client::Object;

// =============================================================================
// Helpers globais
// =============================================================================

static std::string models_dir() {
    const char* e = std::getenv("MODELS_DIR");
    return e ? std::string(e) : std::string("models");
}

static std::string model_path(const std::string& filename) {
    return (fs::path(models_dir()) / filename).string();
}

static std::string linear_path() {
    return model_path("simple_linear.onnx");
}

static std::string classifier_path() {
    return model_path("simple_classifier.onnx");
}

// Monta Object com array de floats para o tensor "input".
static Object make_linear_input(const std::vector<float>& vals) {
    Array arr;
    arr.reserve(vals.size());
    for (float v : vals)
        arr.push_back(Value{static_cast<double>(v)});
    Object obj;
    obj["input"] = Value{std::move(arr)};
    return obj;
}

static Object make_classifier_input(const std::vector<float>& vals) {
    return make_linear_input(vals);  // mesmo formato, tensor 'input'
}

// Extrai vetor de doubles do output nomeado.
static std::vector<double> get_output(const Object& outputs,
                                      const std::string& name) {
    const auto& arr = outputs.at(name).as_array();
    std::vector<double> out;
    out.reserve(arr.size());
    for (const auto& e : arr) out.push_back(e.as_number());
    return out;
}

// =============================================================================
// Fixture base — engine padrão, sem GPU, 4 threads
// =============================================================================

class EngineFixture : public ::testing::Test {
protected:
    InferenceEngine engine;  // construído com defaults

    void SetUp() override {}
    void TearDown() override {}

    bool load_linear(const std::string& id = "linear") {
        return engine.load_model(id, linear_path());
    }

    bool load_classifier(const std::string& id = "classifier") {
        return engine.load_model(id, classifier_path());
    }
};

// =============================================================================
// GRUPO 1 — Construção e EngineInfo
// =============================================================================

TEST(EngineConstruct, DefaultConstructorSucceeds) {
    EXPECT_NO_FATAL_FAILURE({ InferenceEngine e; });
}

TEST(EngineConstruct, GpuFalseConstructorSucceeds) {
    EXPECT_NO_FATAL_FAILURE({ InferenceEngine e(false, 0, 4); });
}

TEST(EngineConstruct, SingleThreadConstructorSucceeds) {
    EXPECT_NO_FATAL_FAILURE({ InferenceEngine e(false, 0, 1); });
}

TEST(EngineConstruct, EngineInfoGpuFalse) {
    InferenceEngine e(false, 0, 4);
    EXPECT_FALSE(e.get_engine_info().gpu_enabled);
}

TEST(EngineConstruct, EngineInfoNumThreads) {
    InferenceEngine e(false, 0, 8);
    EXPECT_EQ(e.get_engine_info().num_threads, 8u);
}

TEST(EngineConstruct, EngineInfoSupportedBackendsNotEmpty) {
    InferenceEngine e;
    EXPECT_FALSE(e.get_engine_info().supported_backends.empty());
}

TEST(EngineConstruct, EngineInfoContainsOnnxBackend) {
    InferenceEngine e;
    const auto& backends = e.get_engine_info().supported_backends;
    bool found = std::any_of(backends.begin(), backends.end(),
                             [](const std::string& s){ return s.find("onnx") != std::string::npos; });
    EXPECT_TRUE(found) << "backend 'onnx' não encontrado em supported_backends";
}

TEST(EngineConstruct, EngineInfoContainsPythonBackend) {
    InferenceEngine e;
    const auto& backends = e.get_engine_info().supported_backends;
    bool found = std::any_of(backends.begin(), backends.end(),
                             [](const std::string& s){ return s.find("py") != std::string::npos ||
                                                               s.find("python") != std::string::npos; });
    EXPECT_TRUE(found) << "backend 'python' não encontrado em supported_backends";
}

TEST(EngineConstruct, InitiallyNoModelsLoaded) {
    InferenceEngine e;
    EXPECT_TRUE(e.get_loaded_model_ids().empty());
}

// =============================================================================
// GRUPO 2 — load_model(): happy path
// =============================================================================

TEST_F(EngineFixture, LoadLinearReturnsTrue) {
    if (!fs::exists(linear_path())) GTEST_SKIP() << "Modelo ausente: " << linear_path();
    EXPECT_TRUE(load_linear());
}

TEST_F(EngineFixture, LoadLinearSetsIsLoaded) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    EXPECT_TRUE(engine.is_model_loaded("linear"));
}

TEST_F(EngineFixture, LoadClassifierReturnsTrue) {
    if (!fs::exists(classifier_path())) GTEST_SKIP();
    EXPECT_TRUE(load_classifier());
}

TEST_F(EngineFixture, LoadTwoDistinctModels) {
    if (!fs::exists(linear_path()) || !fs::exists(classifier_path())) GTEST_SKIP();
    EXPECT_TRUE(load_linear());
    EXPECT_TRUE(load_classifier());
    EXPECT_EQ(engine.get_loaded_model_ids().size(), 2u);
}

TEST_F(EngineFixture, LoadSameFileWithDifferentIds) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    EXPECT_TRUE(engine.load_model("inst_a", linear_path()));
    EXPECT_TRUE(engine.load_model("inst_b", linear_path()));
    EXPECT_EQ(engine.get_loaded_model_ids().size(), 2u);
}

TEST_F(EngineFixture, LoadSameFileWithDifferentIdsProducesIndependentInstances) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(engine.load_model("a", linear_path()));
    ASSERT_TRUE(engine.load_model("b", linear_path()));

    // Ambas devem produzir o mesmo resultado para o mesmo input
    auto ra = engine.predict("a", make_linear_input({1, 2, 3, 4, 5}));
    auto rb = engine.predict("b", make_linear_input({1, 2, 3, 4, 5}));
    ASSERT_TRUE(ra.success && rb.success);
    auto va = get_output(ra.outputs, "output");
    auto vb = get_output(rb.outputs, "output");
    for (size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(va[i], vb[i]);
}

// =============================================================================
// GRUPO 3 — load_model(): falhas
// =============================================================================

TEST_F(EngineFixture, LoadNonExistentFileReturnsFalse) {
    EXPECT_FALSE(engine.load_model("ghost", "/tmp/___nonexistent_xyzzy.onnx"));
}

TEST_F(EngineFixture, LoadNonExistentDoesNotAddToLoadedIds) {
    engine.load_model("ghost", "/tmp/___nonexistent.onnx");
    EXPECT_FALSE(engine.is_model_loaded("ghost"));
}

TEST_F(EngineFixture, LoadDuplicateIdReturnsFalse) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    EXPECT_FALSE(load_linear());  // mesmo ID
}

TEST_F(EngineFixture, LoadDuplicateDoesNotDoubleCount) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    load_linear();  // segunda tentativa com mesmo ID
    EXPECT_EQ(engine.get_loaded_model_ids().size(), 1u);
}

TEST_F(EngineFixture, LoadCorruptFileReturnsFalse) {
    std::string tmp = "/tmp/corrupt_engine_test.onnx";
    { std::ofstream f(tmp, std::ios::binary); f << "garbage"; }
    EXPECT_FALSE(engine.load_model("bad", tmp));
    std::remove(tmp.c_str());
}

TEST_F(EngineFixture, LoadUnknownExtensionReturnsFalse) {
    // Extensão desconhecida — BackendRegistry não tem fábrica para ela
    std::string tmp = "/tmp/unknown_ext_test.xyz";
    { std::ofstream f(tmp); f << "data"; }
    EXPECT_FALSE(engine.load_model("xyz", tmp));
    std::remove(tmp.c_str());
}

// =============================================================================
// GRUPO 4 — unload_model()
// =============================================================================

TEST_F(EngineFixture, UnloadLoadedModelReturnsTrue) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    EXPECT_TRUE(engine.unload_model("linear"));
}

TEST_F(EngineFixture, UnloadClearsIsLoaded) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.unload_model("linear");
    EXPECT_FALSE(engine.is_model_loaded("linear"));
}

TEST_F(EngineFixture, UnloadNonExistentReturnsFalse) {
    EXPECT_FALSE(engine.unload_model("nao_existe"));
}

TEST_F(EngineFixture, UnloadReducesLoadedCount) {
    if (!fs::exists(linear_path()) || !fs::exists(classifier_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    ASSERT_TRUE(load_classifier());
    EXPECT_EQ(engine.get_loaded_model_ids().size(), 2u);
    engine.unload_model("linear");
    EXPECT_EQ(engine.get_loaded_model_ids().size(), 1u);
}

TEST_F(EngineFixture, UnloadThenReloadSucceeds) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    ASSERT_TRUE(engine.unload_model("linear"));
    EXPECT_TRUE(load_linear());
}

TEST_F(EngineFixture, UnloadThenPredictFails) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.unload_model("linear");
    auto r = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    EXPECT_FALSE(r.success);
}

TEST_F(EngineFixture, DoubleUnloadSecondReturnsFalse) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    EXPECT_TRUE(engine.unload_model("linear"));
    EXPECT_FALSE(engine.unload_model("linear"));
}

// =============================================================================
// GRUPO 5 — is_model_loaded()
// =============================================================================

TEST_F(EngineFixture, IsLoadedFalseBeforeLoad) {
    EXPECT_FALSE(engine.is_model_loaded("linear"));
}

TEST_F(EngineFixture, IsLoadedTrueAfterLoad) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    EXPECT_TRUE(engine.is_model_loaded("linear"));
}

TEST_F(EngineFixture, IsLoadedFalseAfterUnload) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.unload_model("linear");
    EXPECT_FALSE(engine.is_model_loaded("linear"));
}

TEST_F(EngineFixture, IsLoadedFalseForArbitraryId) {
    EXPECT_FALSE(engine.is_model_loaded("nonexistent_id_xyz"));
}

// =============================================================================
// GRUPO 6 — get_loaded_model_ids()
// =============================================================================

TEST_F(EngineFixture, LoadedIdsEmptyInitially) {
    EXPECT_TRUE(engine.get_loaded_model_ids().empty());
}

TEST_F(EngineFixture, LoadedIdsContainsLoadedId) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto ids = engine.get_loaded_model_ids();
    EXPECT_NE(std::find(ids.begin(), ids.end(), "linear"), ids.end());
}

TEST_F(EngineFixture, LoadedIdsDoesNotContainUnloadedId) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.unload_model("linear");
    auto ids = engine.get_loaded_model_ids();
    EXPECT_EQ(std::find(ids.begin(), ids.end(), "linear"), ids.end());
}

TEST_F(EngineFixture, LoadedIdsCountMatchesLoadedModels) {
    if (!fs::exists(linear_path()) || !fs::exists(classifier_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    ASSERT_TRUE(load_classifier());
    EXPECT_EQ(engine.get_loaded_model_ids().size(), 2u);
}

TEST_F(EngineFixture, LoadedIdsContainsBothLoadedModels) {
    if (!fs::exists(linear_path()) || !fs::exists(classifier_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    ASSERT_TRUE(load_classifier());
    auto ids = engine.get_loaded_model_ids();
    bool has_linear     = std::find(ids.begin(), ids.end(), "linear")     != ids.end();
    bool has_classifier = std::find(ids.begin(), ids.end(), "classifier") != ids.end();
    EXPECT_TRUE(has_linear);
    EXPECT_TRUE(has_classifier);
}

// =============================================================================
// GRUPO 7 — predict(): happy path e valores
// =============================================================================

TEST_F(EngineFixture, PredictLinearReturnsSuccess) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    EXPECT_TRUE(r.success) << r.error_message;
}

TEST_F(EngineFixture, PredictLinearOutputPresent) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    EXPECT_GT(r.outputs.count("output"), 0u);
}

TEST_F(EngineFixture, PredictLinearZeroInputGivesOnes) {
    // output = 0 * 2 + 1 = 1
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    for (double x : get_output(r.outputs, "output"))
        EXPECT_NEAR(x, 1.0, 1e-4);
}

TEST_F(EngineFixture, PredictLinearOnesInputGivesThrees) {
    // output = 1 * 2 + 1 = 3
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.predict("linear", make_linear_input({1, 1, 1, 1, 1}));
    ASSERT_TRUE(r.success);
    for (double x : get_output(r.outputs, "output"))
        EXPECT_NEAR(x, 3.0, 1e-4);
}

TEST_F(EngineFixture, PredictLinearMixedValues) {
    // [0,1,2,3,4] → [1,3,5,7,9]
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.predict("linear", make_linear_input({0, 1, 2, 3, 4}));
    ASSERT_TRUE(r.success);
    auto v = get_output(r.outputs, "output");
    ASSERT_EQ(v.size(), 5u);
    const std::vector<double> expected = {1, 3, 5, 7, 9};
    for (size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(v[i], expected[i], 1e-4);
}

TEST_F(EngineFixture, PredictLinearNegativeInput) {
    // -2 * 2 + 1 = -3
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.predict("linear", make_linear_input({-2, -2, -2, -2, -2}));
    ASSERT_TRUE(r.success);
    for (double x : get_output(r.outputs, "output"))
        EXPECT_NEAR(x, -3.0, 1e-4);
}

TEST_F(EngineFixture, PredictLinearIsDeterministic) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r1 = engine.predict("linear", make_linear_input({1.5f, 2.5f, 3.5f, 4.5f, 5.5f}));
    auto r2 = engine.predict("linear", make_linear_input({1.5f, 2.5f, 3.5f, 4.5f, 5.5f}));
    ASSERT_TRUE(r1.success && r2.success);
    auto v1 = get_output(r1.outputs, "output");
    auto v2 = get_output(r2.outputs, "output");
    for (size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(v1[i], v2[i]);
}

TEST_F(EngineFixture, PredictLinearLatencyPositive) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    EXPECT_GT(r.inference_time_ms, 0.0);
}

TEST_F(EngineFixture, PredictLinearLatencyReasonable) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    EXPECT_LT(r.inference_time_ms, 500.0);
}

TEST_F(EngineFixture, PredictClassifierSoftmaxSumsToOne) {
    if (!fs::exists(classifier_path())) GTEST_SKIP();
    ASSERT_TRUE(load_classifier());
    auto r = engine.predict("classifier", make_classifier_input({1, 2, 3, 4}));
    ASSERT_TRUE(r.success);
    auto v = get_output(r.outputs, "output");
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-5);
}

TEST_F(EngineFixture, PredictClassifierAllOutputsNonNegative) {
    if (!fs::exists(classifier_path())) GTEST_SKIP();
    ASSERT_TRUE(load_classifier());
    auto r = engine.predict("classifier", make_classifier_input({0, 0, 0, 0}));
    ASSERT_TRUE(r.success);
    for (double x : get_output(r.outputs, "output"))
        EXPECT_GE(x, 0.0);
}

TEST_F(EngineFixture, Predict50SequentialCallsAllSucceed) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    for (int i = 0; i < 50; ++i) {
        auto r = engine.predict("linear", make_linear_input({
            static_cast<float>(i), static_cast<float>(i+1),
            static_cast<float>(i+2), static_cast<float>(i+3),
            static_cast<float>(i+4)}));
        ASSERT_TRUE(r.success) << "falhou na chamada " << i;
    }
}

// =============================================================================
// GRUPO 8 — predict(): falhas de entrada
// =============================================================================

TEST_F(EngineFixture, PredictUnknownIdFails) {
    auto r = engine.predict("nao_existe", make_linear_input({0, 0, 0, 0, 0}));
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(EngineFixture, PredictUnknownIdErrorMessageNotEmpty) {
    auto r = engine.predict("ghost_model", {});
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(EngineFixture, PredictMissingTensorFails) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    Object empty;
    auto r = engine.predict("linear", empty);
    EXPECT_FALSE(r.success);
}

TEST_F(EngineFixture, PredictWrongTensorNameFails) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    Object obj;
    obj["wrong"] = Value{Array{Value{1.0}}};
    auto r = engine.predict("linear", obj);
    EXPECT_FALSE(r.success);
}

TEST_F(EngineFixture, PredictAfterUnloadFails) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.unload_model("linear");
    auto r = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    EXPECT_FALSE(r.success);
}

TEST_F(EngineFixture, PredictFailLatencyNonNegative) {
    auto r = engine.predict("nao_existe", make_linear_input({0, 0, 0, 0, 0}));
    EXPECT_FALSE(r.success);
    EXPECT_GE(r.inference_time_ms, 0.0);
}

// =============================================================================
// GRUPO 9 — sequência de predições em lote (loop)
// =============================================================================

TEST_F(EngineFixture, BatchOf10PredictionsAllSucceed) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    for (int i = 0; i < 10; ++i) {
        auto r = engine.predict("linear", make_linear_input({
            static_cast<float>(i), static_cast<float>(i),
            static_cast<float>(i), static_cast<float>(i),
            static_cast<float>(i)}));
        ASSERT_TRUE(r.success) << "item " << i;
        for (double x : get_output(r.outputs, "output"))
            EXPECT_NEAR(x, 2.0 * i + 1.0, 1e-3) << "item " << i;
    }
}

TEST_F(EngineFixture, BatchResultsAreOrderPreserving) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    // Garante que outputs correspondem ao input correto
    std::vector<std::vector<double>> outputs;
    for (int i = 0; i < 5; ++i) {
        auto r = engine.predict("linear", make_linear_input({
            static_cast<float>(i), 0, 0, 0, 0}));
        ASSERT_TRUE(r.success);
        outputs.push_back(get_output(r.outputs, "output"));
    }
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(outputs[i][0], 2.0 * i + 1.0, 1e-3);
}

// =============================================================================
// GRUPO 10 — get_model_info()
// =============================================================================

TEST_F(EngineFixture, GetModelInfoLoadedModelReturnsId) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto info = engine.get_model_info("linear");
    EXPECT_EQ(info.model_id(), "linear");
}

TEST_F(EngineFixture, GetModelInfoBackendIsOnnx) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto info = engine.get_model_info("linear");
    EXPECT_EQ(info.backend(), miia::common::BACKEND_ONNX);
}

TEST_F(EngineFixture, GetModelInfoHasInputs) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto info = engine.get_model_info("linear");
    EXPECT_GT(info.inputs_size(), 0);
}

TEST_F(EngineFixture, GetModelInfoHasOutputs) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto info = engine.get_model_info("linear");
    EXPECT_GT(info.outputs_size(), 0);
}

TEST_F(EngineFixture, GetModelInfoInputNameIsInput) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto info = engine.get_model_info("linear");
    ASSERT_GT(info.inputs_size(), 0);
    EXPECT_EQ(info.inputs(0).name(), "input");
}

TEST_F(EngineFixture, GetModelInfoOutputNameIsOutput) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto info = engine.get_model_info("linear");
    ASSERT_GT(info.outputs_size(), 0);
    EXPECT_EQ(info.outputs(0).name(), "output");
}

TEST_F(EngineFixture, GetModelInfoModelPathSet) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto info = engine.get_model_info("linear");
    EXPECT_FALSE(info.model_path().empty());
}

TEST_F(EngineFixture, GetModelInfoMemoryUsagePositive) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto info = engine.get_model_info("linear");
    EXPECT_GT(info.memory_usage_bytes(), 0);
}

TEST_F(EngineFixture, GetModelInfoLoadedAtUnixPositive) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto info = engine.get_model_info("linear");
    EXPECT_GT(info.loaded_at_unix(), 0);
}

TEST_F(EngineFixture, GetModelInfoUnknownIdReturnsEmpty) {
    auto info = engine.get_model_info("nao_existe");
    // model_id vazio indica "não encontrado"
    EXPECT_TRUE(info.model_id().empty());
}

TEST_F(EngineFixture, GetModelInfoIsIdempotent) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto i1 = engine.get_model_info("linear");
    auto i2 = engine.get_model_info("linear");
    EXPECT_EQ(i1.model_id(),  i2.model_id());
    EXPECT_EQ(i1.inputs_size(),  i2.inputs_size());
    EXPECT_EQ(i1.outputs_size(), i2.outputs_size());
}

// =============================================================================
// GRUPO 11 — get_model_metrics()
// =============================================================================

TEST_F(EngineFixture, GetMetricsUnknownIdReturnsNullptr) {
    EXPECT_EQ(engine.get_model_metrics("nao_existe"), nullptr);
}

TEST_F(EngineFixture, GetMetricsLoadedModelNotNull) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    EXPECT_NE(engine.get_model_metrics("linear"), nullptr);
}

TEST_F(EngineFixture, GetMetricsInitiallyZeroInferences) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto* m = engine.get_model_metrics("linear");
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->total_inferences, 0u);
}

TEST_F(EngineFixture, GetMetricsTotalIncreasesAfterPredict) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    engine.predict("linear", make_linear_input({1, 1, 1, 1, 1}));
    auto* m = engine.get_model_metrics("linear");
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->total_inferences, 2u);
}

TEST_F(EngineFixture, GetMetricsFailedCounterZeroOnSuccess) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    auto* m = engine.get_model_metrics("linear");
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->failed_inferences, 0u);
}

TEST_F(EngineFixture, GetMetricsFailedCounterIncreasesOnFailure) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    // Array vazio para o tensor correto: OnnxBackend entra no loop,
    // detecta data.empty() e chama metrics_.record(ms, false)
    Object bad;
    bad["input"] = Value{Array{}};
    engine.predict("linear", bad);
    auto* m = engine.get_model_metrics("linear");
    ASSERT_NE(m, nullptr);
    EXPECT_GE(m->failed_inferences, 1u);
}

TEST_F(EngineFixture, GetMetricsAvgTimeMsPositiveAfterPredict) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.predict("linear", make_linear_input({1, 1, 1, 1, 1}));
    auto* m = engine.get_model_metrics("linear");
    ASSERT_NE(m, nullptr);
    EXPECT_GE(m->avg_time_ms(), 0.0);
}

TEST_F(EngineFixture, GetMetrics10PredictsTotalIs10) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    for (int i = 0; i < 10; ++i)
        engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    auto* m = engine.get_model_metrics("linear");
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->total_inferences, 10u);
}

// =============================================================================
// GRUPO 12 — validate_model()
// =============================================================================

TEST_F(EngineFixture, ValidateLinearModelValid) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    auto r = engine.validate_model(linear_path());
    EXPECT_TRUE(r.valid) << r.error_message;
}

TEST_F(EngineFixture, ValidateClassifierModelValid) {
    if (!fs::exists(classifier_path())) GTEST_SKIP();
    auto r = engine.validate_model(classifier_path());
    EXPECT_TRUE(r.valid) << r.error_message;
}

TEST_F(EngineFixture, ValidateNonExistentFileInvalid) {
    auto r = engine.validate_model("/tmp/___nonexistent_engine.onnx");
    EXPECT_FALSE(r.valid);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(EngineFixture, ValidateCorruptFileInvalid) {
    std::string tmp = "/tmp/corrupt_validate_engine.onnx";
    { std::ofstream f(tmp, std::ios::binary); f << "garbage"; }
    auto r = engine.validate_model(tmp);
    EXPECT_FALSE(r.valid);
    std::remove(tmp.c_str());
}

TEST_F(EngineFixture, ValidateUnknownExtensionInvalid) {
    std::string tmp = "/tmp/unknown_ext_validate.xyz";
    { std::ofstream f(tmp); f << "data"; }
    auto r = engine.validate_model(tmp);
    EXPECT_FALSE(r.valid);
    std::remove(tmp.c_str());
}

TEST_F(EngineFixture, ValidateReturnsInputsForLinear) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    auto r = engine.validate_model(linear_path());
    ASSERT_TRUE(r.valid);
    EXPECT_FALSE(r.inputs.empty());
}

TEST_F(EngineFixture, ValidateReturnsOutputsForLinear) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    auto r = engine.validate_model(linear_path());
    ASSERT_TRUE(r.valid);
    EXPECT_FALSE(r.outputs.empty());
}

TEST_F(EngineFixture, ValidateDetectsBackendOnnx) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    auto r = engine.validate_model(linear_path());
    ASSERT_TRUE(r.valid);
    EXPECT_EQ(r.backend, miia::common::BACKEND_ONNX);
}

TEST_F(EngineFixture, ValidateDoesNotLoadModel) {
    // validate não deve alterar o estado do engine
    if (!fs::exists(linear_path())) GTEST_SKIP();
    engine.validate_model(linear_path());
    EXPECT_FALSE(engine.is_model_loaded("linear"));
    EXPECT_TRUE(engine.get_loaded_model_ids().empty());
}

TEST_F(EngineFixture, ValidateWhileModelLoaded) {
    // validate pode ser chamado enquanto o modelo está carregado
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    EXPECT_NO_FATAL_FAILURE({ engine.validate_model(linear_path()); });
    // modelo ainda deve funcionar
    auto r = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    EXPECT_TRUE(r.success);
}

// =============================================================================
// GRUPO 13 — warmup_model()
// =============================================================================

TEST_F(EngineFixture, WarmupLoadedModelSucceeds) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.warmup_model("linear", 3);
    EXPECT_TRUE(r.success) << r.error_message;
}

TEST_F(EngineFixture, WarmupRunsCompletedMatchesRequest) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.warmup_model("linear", 5);
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.runs_completed, 5u);
}

TEST_F(EngineFixture, WarmupZeroRunsDefaultsToFive) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.warmup_model("linear", 0);
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.runs_completed, 5u);  // motor ajusta 0 → 5
}

TEST_F(EngineFixture, WarmupAvgTimeMsPositive) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.warmup_model("linear", 3);
    ASSERT_TRUE(r.success);
    EXPECT_GE(r.avg_time_ms, 0.0);
}

TEST_F(EngineFixture, WarmupMinTimeMsLeqAvg) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.warmup_model("linear", 5);
    ASSERT_TRUE(r.success);
    EXPECT_LE(r.min_time_ms, r.avg_time_ms + 1e-9);
}

TEST_F(EngineFixture, WarmupMaxTimeMsGeqAvg) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    auto r = engine.warmup_model("linear", 5);
    ASSERT_TRUE(r.success);
    EXPECT_GE(r.max_time_ms, r.avg_time_ms - 1e-9);
}

TEST_F(EngineFixture, WarmupUnknownIdFails) {
    auto r = engine.warmup_model("nao_existe", 3);
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(EngineFixture, WarmupUpdatesMetrics) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.warmup_model("linear", 5);
    auto* m = engine.get_model_metrics("linear");
    ASSERT_NE(m, nullptr);
    EXPECT_GE(m->total_inferences, 5u);
}

TEST_F(EngineFixture, WarmupModelStillFunctionalAfterWarmup) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    engine.warmup_model("linear", 3);
    auto r = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    EXPECT_TRUE(r.success);
    for (double x : get_output(r.outputs, "output"))
        EXPECT_NEAR(x, 1.0, 1e-4);
}

// =============================================================================
// GRUPO 14 — Múltiplos modelos simultâneos
// =============================================================================

TEST_F(EngineFixture, TwoModelsIndependentPredictions) {
    if (!fs::exists(linear_path()) || !fs::exists(classifier_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    ASSERT_TRUE(load_classifier());

    auto rl = engine.predict("linear", make_linear_input({1, 1, 1, 1, 1}));
    auto rc = engine.predict("classifier", make_classifier_input({1, 2, 3, 4}));

    ASSERT_TRUE(rl.success);
    ASSERT_TRUE(rc.success);

    // linear: [3,3,3,3,3]
    for (double x : get_output(rl.outputs, "output"))
        EXPECT_NEAR(x, 3.0, 1e-4);

    // classifier: softmax soma 1
    auto vc = get_output(rc.outputs, "output");
    double sum = std::accumulate(vc.begin(), vc.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-5);
}

TEST_F(EngineFixture, UnloadOneDoesNotAffectOther) {
    if (!fs::exists(linear_path()) || !fs::exists(classifier_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());
    ASSERT_TRUE(load_classifier());

    engine.unload_model("linear");

    // classifier ainda deve funcionar
    auto r = engine.predict("classifier", make_classifier_input({0, 0, 0, 0}));
    EXPECT_TRUE(r.success);

    // linear deve falhar
    auto r2 = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    EXPECT_FALSE(r2.success);
}

TEST_F(EngineFixture, FiveModelsWithSameFile) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    for (int i = 0; i < 5; ++i)
        ASSERT_TRUE(engine.load_model("m" + std::to_string(i), linear_path()));
    EXPECT_EQ(engine.get_loaded_model_ids().size(), 5u);

    for (int i = 0; i < 5; ++i) {
        auto r = engine.predict("m" + std::to_string(i), make_linear_input({0, 0, 0, 0, 0}));
        EXPECT_TRUE(r.success) << "falhou para m" << i;
    }
}

TEST_F(EngineFixture, MetricsAreIndependentPerModel) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(engine.load_model("a", linear_path()));
    ASSERT_TRUE(engine.load_model("b", linear_path()));

    // 3 predições em "a", 7 em "b"
    for (int i = 0; i < 3; ++i) engine.predict("a", make_linear_input({0, 0, 0, 0, 0}));
    for (int i = 0; i < 7; ++i) engine.predict("b", make_linear_input({0, 0, 0, 0, 0}));

    EXPECT_EQ(engine.get_model_metrics("a")->total_inferences, 3u);
    EXPECT_EQ(engine.get_model_metrics("b")->total_inferences, 7u);
}

// =============================================================================
// GRUPO 15 — Ciclos load → predict → unload repetidos
// =============================================================================

TEST_F(EngineFixture, LoadPredictUnload5Cycles) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    for (int cycle = 0; cycle < 5; ++cycle) {
        ASSERT_TRUE(engine.load_model("cycle_model", linear_path())) << "cycle=" << cycle;
        auto r = engine.predict("cycle_model", make_linear_input({
            static_cast<float>(cycle), 0, 0, 0, 0}));
        ASSERT_TRUE(r.success) << "cycle=" << cycle;
        auto v = get_output(r.outputs, "output");
        EXPECT_NEAR(v[0], 2.0 * cycle + 1.0, 1e-3) << "cycle=" << cycle;
        ASSERT_TRUE(engine.unload_model("cycle_model")) << "cycle=" << cycle;
    }
}

TEST_F(EngineFixture, AfterAllUnloadsEngineIsEmpty) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(engine.load_model("a", linear_path()));
    ASSERT_TRUE(engine.load_model("b", linear_path()));
    engine.unload_model("a");
    engine.unload_model("b");
    EXPECT_TRUE(engine.get_loaded_model_ids().empty());
}

// =============================================================================
// GRUPO 16 — Destrutor descarrega modelos ativos
// =============================================================================

TEST(EngineDestructor, DestroyWithLoadedModelsNoCrash) {
    if (!fs::exists(model_path("simple_linear.onnx"))) GTEST_SKIP();
    EXPECT_NO_FATAL_FAILURE({
        InferenceEngine e;
        e.load_model("m1", model_path("simple_linear.onnx"));
        e.load_model("m2", model_path("simple_linear.onnx"));
        // destrutor deve descarregar ambos sem crash
    });
}

TEST(EngineDestructor, DestroyWithoutLoadedModelsNoCrash) {
    EXPECT_NO_FATAL_FAILURE({
        InferenceEngine e;
        // destrutor com mapa vazio
    });
}

// =============================================================================
// GRUPO 17 — Concorrência: leituras simultâneas via is_model_loaded / get_ids
// =============================================================================

TEST_F(EngineFixture, ConcurrentIsModelLoadedCallsAreSafe) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());

    constexpr int N_THREADS = 8;
    constexpr int N_ITERS   = 100;
    std::atomic<int> errors{0};

    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < N_ITERS; ++i) {
                if (!engine.is_model_loaded("linear")) ++errors;
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(errors.load(), 0);
}

TEST_F(EngineFixture, ConcurrentPredictCallsAreSafe) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());

    constexpr int N_THREADS = 4;
    constexpr int N_ITERS   = 20;
    std::atomic<int> failures{0};

    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < N_ITERS; ++i) {
                auto r = engine.predict("linear", make_linear_input({
                    static_cast<float>(t), static_cast<float>(i),
                    0, 0, 0}));
                if (!r.success) ++failures;
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(failures.load(), 0);
}

TEST_F(EngineFixture, ConcurrentGetLoadedIdsIsSafe) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load_linear());

    constexpr int N_THREADS = 8;
    std::atomic<int> errors{0};

    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 50; ++i) {
                auto ids = engine.get_loaded_model_ids();
                if (ids.size() != 1u) ++errors;
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(errors.load(), 0);
}

// =============================================================================
// GRUPO 18 — Ciclo completo end-to-end
// =============================================================================

TEST_F(EngineFixture, EndToEnd_LinearFullLifecycle) {
    if (!fs::exists(linear_path())) GTEST_SKIP();

    // 1. Valida antes de carregar
    auto vr = engine.validate_model(linear_path());
    ASSERT_TRUE(vr.valid) << vr.error_message;
    EXPECT_FALSE(vr.inputs.empty());
    EXPECT_FALSE(vr.outputs.empty());

    // 2. Carrega
    ASSERT_TRUE(load_linear());
    EXPECT_TRUE(engine.is_model_loaded("linear"));

    // 3. Warmup
    auto wr = engine.warmup_model("linear", 3);
    EXPECT_TRUE(wr.success);
    EXPECT_EQ(wr.runs_completed, 3u);

    // 4. Introspecção
    auto info = engine.get_model_info("linear");
    EXPECT_EQ(info.model_id(), "linear");
    EXPECT_EQ(info.backend(), miia::common::BACKEND_ONNX);
    ASSERT_GT(info.inputs_size(), 0);
    EXPECT_EQ(info.inputs(0).name(), "input");

    // 5. Predições com verificação matemática
    for (int i = 0; i < 5; ++i) {
        float v = static_cast<float>(i);
        auto r = engine.predict("linear", make_linear_input({v, v, v, v, v}));
        ASSERT_TRUE(r.success) << "i=" << i;
        for (double x : get_output(r.outputs, "output"))
            EXPECT_NEAR(x, 2.0 * i + 1.0, 1e-3) << "i=" << i;
    }

    // 6. Métricas: 3 warmups + 5 predições = 8 total
    auto* m = engine.get_model_metrics("linear");
    ASSERT_NE(m, nullptr);
    EXPECT_EQ(m->total_inferences, 8u);
    EXPECT_EQ(m->failed_inferences, 0u);

    // 7. Descarrega
    EXPECT_TRUE(engine.unload_model("linear"));
    EXPECT_FALSE(engine.is_model_loaded("linear"));

    // 8. Predição após unload deve falhar
    auto fail = engine.predict("linear", make_linear_input({0, 0, 0, 0, 0}));
    EXPECT_FALSE(fail.success);
}

TEST_F(EngineFixture, EndToEnd_TwoModelsInterleaved) {
    if (!fs::exists(linear_path()) || !fs::exists(classifier_path())) GTEST_SKIP();

    ASSERT_TRUE(load_linear());
    ASSERT_TRUE(load_classifier());

    // Intercala predições entre os dois modelos
    for (int i = 0; i < 5; ++i) {
        auto rl = engine.predict("linear", make_linear_input({
            static_cast<float>(i), 0, 0, 0, 0}));
        auto rc = engine.predict("classifier", make_classifier_input({
            static_cast<float>(i), 1, 2, 3}));
        ASSERT_TRUE(rl.success) << "linear falhou em i=" << i;
        ASSERT_TRUE(rc.success) << "classifier falhou em i=" << i;

        auto vl = get_output(rl.outputs, "output");
        EXPECT_NEAR(vl[0], 2.0 * i + 1.0, 1e-3);

        auto vc = get_output(rc.outputs, "output");
        double sum = std::accumulate(vc.begin(), vc.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-5);
    }

    // Métricas independentes: 5 cada
    EXPECT_EQ(engine.get_model_metrics("linear")->total_inferences, 5u);
    EXPECT_EQ(engine.get_model_metrics("classifier")->total_inferences, 5u);

    engine.unload_model("linear");
    engine.unload_model("classifier");
    EXPECT_TRUE(engine.get_loaded_model_ids().empty());
}
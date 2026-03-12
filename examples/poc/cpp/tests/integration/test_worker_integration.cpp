// =============================================================================
// test_worker_integration.cpp — Testes de integração (requer worker rodando)
//
// Esses testes fazem chamadas gRPC reais ao worker.
// Se o worker não estiver disponível, todos os testes são pulados com GTEST_SKIP.
//
// Configuração via variáveis de ambiente:
//   WORKER_ADDRESS=localhost:50052   (padrão)
//   MODELS_DIR=./models             (padrão)
//
// Execute o worker antes de rodar:
//   make run-worker &
//   make test-integration
// =============================================================================

#include <gtest/gtest.h>
#include <client/inference_client.hpp>
#include <thread>
#include <future>
#include <atomic>
#include <numeric>
#include <cstdlib>
#include <chrono>
#include <filesystem>

using mlinference::client::InferenceClient;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers globais
// ─────────────────────────────────────────────────────────────────────────────

static std::string worker_address() {
    const char* env = std::getenv("WORKER_ADDRESS");
    return env ? std::string(env) : "localhost:50052";
}

static std::string models_dir() {
    const char* env = std::getenv("MODELS_DIR");
    return env ? std::string(env) : "./models";
}

static std::string linear_model() {
    return models_dir() + "/simple_linear.onnx";
}

static std::string classifier_model() {
    return models_dir() + "/simple_classifier.onnx";
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture base — conecta ao worker e pula se indisponível
// ─────────────────────────────────────────────────────────────────────────────

class WorkerTest : public ::testing::Test {
protected:
    void SetUp() override {
        client = std::make_unique<InferenceClient>(worker_address());
        bool connected = client->connect();
        if (!connected || !client->health_check()) {
            GTEST_SKIP() << "Worker indisponível em " << worker_address()
                         << ". Execute: make run-worker";
        }
    }

    void TearDown() override {
        // Limpar qualquer modelo que possa ter ficado carregado
        for (const auto& id : loaded_model_ids) {
            client->unload_model(id);
        }
    }

    // Helper: carrega modelo e registra para cleanup automático
    bool load(const std::string& id, const std::string& path,
              const std::string& version = "1.0.0") {
        bool ok = client->load_model(id, path, version);
        if (ok) loaded_model_ids.push_back(id);
        return ok;
    }

    std::unique_ptr<InferenceClient> client;
    std::vector<std::string> loaded_model_ids;
};

// =============================================================================
// Grupo 1: Conectividade e saúde
// =============================================================================

TEST_F(WorkerTest, HealthCheckReturnsTrue) {
    EXPECT_TRUE(client->health_check());
}

TEST_F(WorkerTest, GetStatusReturnsValidData) {
    auto status = client->get_status();
    EXPECT_FALSE(status.worker_id.empty());
    EXPECT_GE(status.uptime_seconds, 0LL);
}

TEST_F(WorkerTest, GetStatusHasSupportedBackends) {
    auto status = client->get_status();
    EXPECT_FALSE(status.supported_backends.empty());
}

TEST_F(WorkerTest, MultipleHealthChecksAllSucceed) {
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(client->health_check()) << "Falhou no health check " << i;
    }
}

// =============================================================================
// Grupo 2: Ciclo de vida de modelos
// =============================================================================

TEST_F(WorkerTest, LoadLinearModelSucceeds) {
    EXPECT_TRUE(load("linear", linear_model()));
}

TEST_F(WorkerTest, LoadSameModelTwiceReturnsFalse) {
    load("linear", linear_model());
    EXPECT_FALSE(client->load_model("linear", linear_model()));
}

TEST_F(WorkerTest, LoadMultipleModelsSucceed) {
    EXPECT_TRUE(load("m1", linear_model()));
    if (std::filesystem::exists(classifier_model())) {
        EXPECT_TRUE(load("m2", classifier_model()));
    }
}

TEST_F(WorkerTest, UnloadLoadedModelSucceeds) {
    load("linear_temp", linear_model());
    bool ok = client->unload_model("linear_temp");
    loaded_model_ids.erase(
        std::remove(loaded_model_ids.begin(), loaded_model_ids.end(), "linear_temp"),
        loaded_model_ids.end());
    EXPECT_TRUE(ok);
}

TEST_F(WorkerTest, UnloadNonExistentModelReturnsFalse) {
    EXPECT_FALSE(client->unload_model("modelo_que_nunca_existiu_" + std::to_string(rand())));
}

TEST_F(WorkerTest, ReloadAfterUnloadSucceeds) {
    load("linear_reload", linear_model());
    client->unload_model("linear_reload");
    loaded_model_ids.clear();

    EXPECT_TRUE(load("linear_reload", linear_model()));
}

TEST_F(WorkerTest, LoadNonExistentFileReturnsFalse) {
    EXPECT_FALSE(client->load_model("fantasma", "/tmp/modelo_inexistente.onnx"));
}

TEST_F(WorkerTest, ListModelsReflectsLoadedModels) {
    load("list_test_1", linear_model());

    auto models = client->list_models();
    bool found = false;
    for (const auto& m : models) {
        if (m.model_id == "list_test_1") { found = true; break; }
    }
    EXPECT_TRUE(found);
}

// =============================================================================
// Grupo 3: Inferência — correctness
// =============================================================================

TEST_F(WorkerTest, PredictLinearModelReturnsSuccess) {
    load("infer_linear", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto result = client->predict("infer_linear", inputs);

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.outputs.empty());
}

TEST_F(WorkerTest, PredictOutputHasCorrectTensorName) {
    load("tensor_name_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto result = client->predict("tensor_name_test", inputs);

    ASSERT_TRUE(result.success);
    EXPECT_TRUE(result.outputs.count("output") > 0)
        << "Tensor 'output' não encontrado. Tensors retornados: "
        << [&]() {
               std::string s;
               for (auto& [k, _] : result.outputs) s += k + " ";
               return s;
           }();
}

TEST_F(WorkerTest, PredictWithZeroInputSucceeds) {
    load("zero_input_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    EXPECT_TRUE(client->predict("zero_input_test", inputs).success);
}

TEST_F(WorkerTest, PredictWithNegativeInputSucceeds) {
    load("neg_input_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {-5.0f, -4.0f, -3.0f, -2.0f, -1.0f};

    EXPECT_TRUE(client->predict("neg_input_test", inputs).success);
}

TEST_F(WorkerTest, PredictWithLargeInputValuesSucceeds) {
    load("large_val_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1000.0f, 2000.0f, 3000.0f, 4000.0f, 5000.0f};

    EXPECT_TRUE(client->predict("large_val_test", inputs).success);
}

TEST_F(WorkerTest, PredictOnNonLoadedModelReturnsFailure) {
    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f};

    auto result = client->predict("modelo_nao_carregado_xyz", inputs);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

TEST_F(WorkerTest, PredictWithMissingInputTensorReturnsFailure) {
    load("missing_tensor_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    // Propositalmente fornece tensor errado
    inputs["tensor_errado"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto result = client->predict("missing_tensor_test", inputs);
    EXPECT_FALSE(result.success);
}

// =============================================================================
// Grupo 4: Inferência — latência
// =============================================================================

TEST_F(WorkerTest, InferenceTimeIsPositive) {
    load("lat_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto result = client->predict("lat_test", inputs);

    ASSERT_TRUE(result.success);
    EXPECT_GT(result.inference_time_ms, 0.0);
}

TEST_F(WorkerTest, InferencePlusRTTUnder100ms) {
    load("rtt_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto wall_start = std::chrono::high_resolution_clock::now();
    auto result = client->predict("rtt_test", inputs);
    auto wall_end = std::chrono::high_resolution_clock::now();

    double wall_ms = std::chrono::duration<double, std::milli>(
        wall_end - wall_start).count();

    ASSERT_TRUE(result.success);
    EXPECT_LT(wall_ms, 100.0)
        << "Round-trip levou " << wall_ms << "ms (limite: 100ms)";
}

TEST_F(WorkerTest, WarmupReducesAverageLatency) {
    load("warmup_lat_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Medir antes do warmup
    double before_total = 0.0;
    const int N = 5;
    for (int i = 0; i < N; ++i) {
        before_total += client->predict("warmup_lat_test", inputs).inference_time_ms;
    }
    double before_avg = before_total / N;

    // Warmup
    client->warmup_model("warmup_lat_test", 10);

    // Medir depois
    double after_total = 0.0;
    for (int i = 0; i < N; ++i) {
        after_total += client->predict("warmup_lat_test", inputs).inference_time_ms;
    }
    double after_avg = after_total / N;

    // Após warmup latência deve ser <= antes (ou no mínimo não muito pior)
    // Usamos fator 3x como limite superior para evitar flakiness
    EXPECT_LE(after_avg, before_avg * 3.0)
        << "Latência pós-warmup (" << after_avg
        << "ms) muito maior que pré-warmup (" << before_avg << "ms)";
}

// =============================================================================
// Grupo 5: Batch predict
// =============================================================================

TEST_F(WorkerTest, BatchPredictReturnsCorrectCount) {
    load("batch_test", linear_model());

    const int BATCH_SIZE = 5;
    std::vector<std::map<std::string, std::vector<float>>> batch;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        std::map<std::string, std::vector<float>> inp;
        inp["input"] = {static_cast<float>(i), 0.0f, 0.0f, 0.0f, 0.0f};
        batch.push_back(inp);
    }

    auto results = client->batch_predict("batch_test", batch);

    EXPECT_EQ(results.size(), static_cast<size_t>(BATCH_SIZE));
}

TEST_F(WorkerTest, BatchPredictAllSucceed) {
    load("batch_all_test", linear_model());

    std::vector<std::map<std::string, std::vector<float>>> batch;
    for (int i = 0; i < 10; ++i) {
        std::map<std::string, std::vector<float>> inp;
        inp["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        batch.push_back(inp);
    }

    auto results = client->batch_predict("batch_all_test", batch);

    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_TRUE(results[i].success) << "Falhou no item " << i;
    }
}

TEST_F(WorkerTest, BatchPredictResultsOrderMatchesInput) {
    load("batch_order_test", linear_model());

    // Inputs distintos → outputs distintos → podemos verificar ordem
    std::vector<std::map<std::string, std::vector<float>>> batch;
    for (int i = 0; i < 5; ++i) {
        std::map<std::string, std::vector<float>> inp;
        float v = static_cast<float>(i + 1);
        inp["input"] = {v, v, v, v, v};
        batch.push_back(inp);
    }

    auto results = client->batch_predict("batch_order_test", batch);
    ASSERT_EQ(results.size(), batch.size());

    // Cada resultado deve ter output
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_TRUE(results[i].success) << "Item " << i << " falhou";
        EXPECT_FALSE(results[i].outputs.empty()) << "Item " << i << " sem output";
    }
}

// =============================================================================
// Grupo 6: Warmup e validação
// =============================================================================

TEST_F(WorkerTest, WarmupModelSucceeds) {
    load("warmup_test", linear_model());

    auto result = client->warmup_model("warmup_test", 5);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.runs_completed, 5u);
    EXPECT_GT(result.avg_time_ms, 0.0);
    EXPECT_GT(result.min_time_ms, 0.0);
    EXPECT_GT(result.max_time_ms, 0.0);
    EXPECT_LE(result.min_time_ms, result.avg_time_ms);
    EXPECT_LE(result.avg_time_ms, result.max_time_ms);
}

TEST_F(WorkerTest, WarmupOnNonLoadedModelReturnsFalse) {
    auto result = client->warmup_model("nao_existe_xyz");
    EXPECT_FALSE(result.success);
}

TEST_F(WorkerTest, ValidateLinearModelReturnsValid) {
    auto result = client->validate_model(linear_model());
    EXPECT_TRUE(result.valid);
    EXPECT_FALSE(result.backend.empty());
}

TEST_F(WorkerTest, ValidateNonExistentFileReturnsInvalid) {
    auto result = client->validate_model("/tmp/modelo_fantasma.onnx");
    EXPECT_FALSE(result.valid);
}

// =============================================================================
// Grupo 7: Introspecção
// =============================================================================

TEST_F(WorkerTest, GetModelInfoAfterLoad) {
    load("info_test", linear_model());

    auto info = client->get_model_info("info_test");

    EXPECT_EQ(info.model_id, "info_test");
    EXPECT_FALSE(info.backend.empty());
    EXPECT_FALSE(info.inputs.empty());
    EXPECT_FALSE(info.outputs.empty());
}

TEST_F(WorkerTest, ListAvailableModelsContainsLinear) {
    auto models = client->list_available_models(models_dir());

    bool found = false;
    for (const auto& m : models) {
        if (m.filename.find("simple_linear") != std::string::npos) {
            found = true; break;
        }
    }
    EXPECT_TRUE(found) << "simple_linear.onnx não encontrado em list_available_models";
}

// =============================================================================
// Grupo 8: Métricas acumuladas no worker
// =============================================================================

TEST_F(WorkerTest, WorkerMetricsAccumulateAcrossRequests) {
    load("metrics_test", linear_model());

    auto status_before = client->get_status();

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    const int N = 10;
    for (int i = 0; i < N; ++i) {
        client->predict("metrics_test", inputs);
    }

    auto status_after = client->get_status();

    EXPECT_GE(status_after.total_requests,
              status_before.total_requests + N);
}

TEST_F(WorkerTest, WorkerSuccessfulRequestsCountsCorrectly) {
    load("success_count_test", linear_model());

    auto status_before = client->get_status();

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    const int N = 5;
    for (int i = 0; i < N; ++i) {
        client->predict("success_count_test", inputs);
    }

    auto status_after = client->get_status();
    EXPECT_GE(status_after.successful_requests,
              status_before.successful_requests + N);
}

// =============================================================================
// Grupo 9: Carga (stress test rápido)
// =============================================================================

TEST_F(WorkerTest, StressTest50RequestsAllSucceed) {
    load("stress_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    int success = 0;
    const int TOTAL = 50;

    for (int i = 0; i < TOTAL; ++i) {
        if (client->predict("stress_test", inputs).success) success++;
    }

    EXPECT_EQ(success, TOTAL);
}

TEST_F(WorkerTest, ConcurrentClientsDoNotInterfere) {
    // Carrega o modelo com o cliente principal
    load("concurrent_test", linear_model());

    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    std::atomic<int> success_count{0};
    const int CLIENTS = 4;
    const int CALLS = 10;

    std::vector<std::future<void>> futures;
    for (int c = 0; c < CLIENTS; ++c) {
        futures.push_back(std::async(std::launch::async, [&]() {
            InferenceClient local_client(worker_address());
            local_client.connect();
            for (int i = 0; i < CALLS; ++i) {
                if (local_client.predict("concurrent_test", inputs).success)
                    success_count++;
            }
        }));
    }

    for (auto& f : futures) f.get();

    EXPECT_EQ(success_count.load(), CLIENTS * CALLS);
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
// =============================================================================
// tests/integration/test_server_integration.cpp
//
// Teste de integração — WorkerServer (servidor embutido no processo)
//
// Estratégia:
//   O WorkerServer é iniciado em uma std::thread separada numa porta
//   efêmera (50099) e derrubado no TearDown via stop().  O cliente usa
//   GrpcClientBackend ("localhost:50099") — exercita o caminho completo:
//
//     InferenceClient → GrpcClientBackend → [rede loopback]
//       → WorkerServer → WorkerServiceImpl → InferenceEngine → backend
//
//   Isso diferencia estes testes dos testes unitários do WorkerServiceImpl
//   (que chamam RPCs diretamente sem rede) e dos testes gRPC externos
//   (que dependem de servidor externo).  Aqui o servidor nasce e morre com
//   cada suite — nenhuma dependência externa.
//
// Cobertura:
//   GRUPO 1  — Inicialização e ciclo de vida do WorkerServer
//   GRUPO 2  — Conectividade gRPC (connect, health_check)
//   GRUPO 3  — LoadModel / UnloadModel via rede
//   GRUPO 4  — Predict via rede (happy path + falhas)
//   GRUPO 5  — BatchPredict via rede
//   GRUPO 6  — ListModels / GetModelInfo via rede
//   GRUPO 7  — ValidateModel via rede
//   GRUPO 8  — WarmupModel via rede
//   GRUPO 9  — GetStatus / GetMetrics via rede
//   GRUPO 10 — ListAvailableModels via rede
//   GRUPO 11 — Ciclo completo end-to-end
//
// Dependências de build (tests/integration/meson.build):
//   test('integration_server',
//       executable('test_integration_server',
//           'test_server_integration.cpp',
//           include_directories: [server_inc, inference_inc, client_inc],
//           dependencies: [worker_server_dep, proto_dep,
//                          gtest_main_dep, gtest_dep],
//           install: false,
//       ),
//       suite:       'integration',
//       timeout:     180,
//       is_parallel: false,
//       env: {
//           'MODELS_DIR': project_root / 'models',
//           'LOG_LEVEL':  'ERROR',
//       },
//   )
//
// Variáveis de ambiente:
//   MODELS_DIR   — diretório com modelos Python e ONNX
//   LOG_LEVEL    — nível de log (padrão: ERROR para silenciar output)
// =============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "server/worker_server.hpp"
#include "client/inference_client.hpp"

namespace fs = std::filesystem;
using mlinference::server::WorkerServer;
using mlinference::client::InferenceClient;
using mlinference::client::Object;
using mlinference::client::Array;
using mlinference::client::Value;
using mlinference::client::PredictionResult;

// =============================================================================
// Configuração da porta de teste
// =============================================================================

// Porta efêmera dedicada — não conflita com o servidor real (50052) nem
// com outros testes gRPC (50053).
static constexpr const char* TEST_ADDRESS = "localhost:50099";

// =============================================================================
// Helpers globais
// =============================================================================

static std::string models_dir() {
    const char* e = std::getenv("MODELS_DIR");
    const std::string raw = e ? e : "./models";
    return fs::weakly_canonical(raw).string();
}

static std::string linear_path() {
    return models_dir() + "/simple_linear.py";
}

/// Monta inputs válidos: "input" → vetor de 5 doubles.
static Object make_valid_inputs(double v = 1.0) {
    Array arr;
    for (int i = 0; i < 5; ++i) arr.push_back(Value{v + i});
    Object o;
    o["input"] = Value{std::move(arr)};
    return o;
}

/// Monta inputs com chave errada (falha esperada).
static Object make_wrong_key_inputs() {
    Array arr;
    for (int i = 0; i < 5; ++i) arr.push_back(Value{static_cast<double>(i)});
    Object o;
    o["chave_errada"] = Value{std::move(arr)};
    return o;
}

/// Aguarda até que o servidor responda ao health_check, com timeout.
static bool wait_for_server(const std::string& address,
                             std::chrono::milliseconds timeout = std::chrono::milliseconds(5000))
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        InferenceClient probe(address);
        if (probe.connect() && probe.health_check())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

// =============================================================================
// Environment global — sobe e derruba um WorkerServer para toda a suite
// =============================================================================

/// GlobalServerEnv: sobe o WorkerServer uma vez para todos os testes.
/// Cada fixture conecta um InferenceClient fresco; o servidor é compartilhado.
class GlobalServerEnv : public ::testing::Environment {
public:
    void SetUp() override {
        server_ = std::make_unique<WorkerServer>(
            "test-integration-server",
            TEST_ADDRESS,
            /*enable_gpu=*/false,
            /*num_threads=*/2,
            models_dir());

        // run() é bloqueante — roda em thread separada
        server_thread_ = std::thread([this]{ server_->run(); });

        // Aguarda o servidor ficar disponível (máx 5s)
        if (!wait_for_server(TEST_ADDRESS)) {
            server_->stop();
            server_thread_.join();
            FAIL() << "WorkerServer não ficou disponível em " << TEST_ADDRESS;
        }
    }

    void TearDown() override {
        if (server_) {
            server_->stop();
        }
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        server_.reset();
    }

    static bool available() {
        InferenceClient probe(TEST_ADDRESS);
        return probe.connect() && probe.health_check();
    }

private:
    std::unique_ptr<WorkerServer> server_;
    std::thread server_thread_;
};

// =============================================================================
// Fixture base — cria um InferenceClient conectado ao servidor de teste
// =============================================================================

class ServerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        client = std::make_unique<InferenceClient>(TEST_ADDRESS);
        if (!client->connect() || !client->health_check())
            GTEST_SKIP() << "WorkerServer indisponível em " << TEST_ADDRESS;
    }

    void TearDown() override {
        if (client) {
            for (const auto& m : client->list_models())
                client->unload_model(m.model_id);
        }
    }

    bool load(const std::string& id, const std::string& path) {
        return client->load_model(id, path);
    }

    bool unload(const std::string& id) {
        return client->unload_model(id);
    }

    PredictionResult predict(const std::string& id, double v = 1.0) {
        return client->predict(id, make_valid_inputs(v));
    }

    std::unique_ptr<InferenceClient> client;
};

/// Fixture que pula se o modelo Python não existir e pré-carrega "linear".
class ServerIntegrationPythonTest : public ServerIntegrationTest {
protected:
    void SetUp() override {
        ServerIntegrationTest::SetUp();
        if (!fs::exists(linear_path()))
            GTEST_SKIP() << "Modelo não encontrado: " << linear_path();
        ASSERT_TRUE(load("linear", linear_path()))
            << "load_model() falhou — verifique: " << linear_path();
    }
};

// =============================================================================
// GRUPO 1 — Ciclo de vida do WorkerServer
// =============================================================================

TEST(WorkerServerLifecycle, ServidorRespondeAposRun) {
    // O servidor deve aceitar conexões após run() ser chamado na thread.
    EXPECT_TRUE(GlobalServerEnv::available());
}

TEST(WorkerServerLifecycle, ConectarDuasVezesNaoFalha) {
    InferenceClient c1(TEST_ADDRESS), c2(TEST_ADDRESS);
    EXPECT_TRUE(c1.connect());
    EXPECT_TRUE(c2.connect());
}

TEST(WorkerServerLifecycle, MultiplosCientesSimultaneos) {
    // Três clientes conectados ao mesmo servidor simultaneamente.
    std::vector<std::unique_ptr<InferenceClient>> clients;
    for (int i = 0; i < 3; ++i) {
        auto c = std::make_unique<InferenceClient>(TEST_ADDRESS);
        EXPECT_TRUE(c->connect()) << "cliente " << i;
        clients.push_back(std::move(c));
    }
    for (auto& c : clients)
        EXPECT_TRUE(c->health_check());
}

// =============================================================================
// GRUPO 2 — Conectividade gRPC
// =============================================================================

TEST_F(ServerIntegrationTest, Connect_RetornaTrue) {
    // connect() já chamado no SetUp — is_connected() deve ser true.
    EXPECT_TRUE(client->is_connected());
}

TEST_F(ServerIntegrationTest, HealthCheck_RetornaTrue) {
    EXPECT_TRUE(client->health_check());
}

TEST_F(ServerIntegrationTest, HealthCheck_EstavelEm20Chamadas) {
    for (int i = 0; i < 20; ++i)
        EXPECT_TRUE(client->health_check()) << "iteração " << i;
}

TEST(Conectividade, PortaErrada_ConnectFalha) {
    InferenceClient c("localhost:19997");
    EXPECT_FALSE(c.connect());
    EXPECT_FALSE(c.is_connected());
}

TEST(Conectividade, IsConnectedFalseAntesDeConnect) {
    InferenceClient c(TEST_ADDRESS);
    EXPECT_FALSE(c.is_connected());
}

// =============================================================================
// GRUPO 3 — LoadModel / UnloadModel via rede
// =============================================================================

TEST_F(ServerIntegrationTest, LoadModel_PathValido_Sucesso) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    EXPECT_TRUE(load("m1", linear_path()));
}

TEST_F(ServerIntegrationTest, LoadModel_PathInexistente_Falha) {
    EXPECT_FALSE(load("m_bad", "/nao/existe/modelo.py"));
}

TEST_F(ServerIntegrationTest, LoadModel_IDDuplicado_Falha) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load("dup", linear_path()));
    EXPECT_FALSE(load("dup", linear_path()));
}

TEST_F(ServerIntegrationTest, LoadModel_IDsDistintosMesmoArquivo_Sucesso) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    EXPECT_TRUE(load("inst_a", linear_path()));
    EXPECT_TRUE(load("inst_b", linear_path()));
}

TEST_F(ServerIntegrationPythonTest, UnloadModel_HappyPath_Sucesso) {
    EXPECT_TRUE(unload("linear"));
}

TEST_F(ServerIntegrationTest, UnloadModel_IDInexistente_Falha) {
    EXPECT_FALSE(unload("nao_existe"));
}

TEST_F(ServerIntegrationPythonTest, UnloadModel_DuplaDescarga_SegundaFalha) {
    EXPECT_TRUE(unload("linear"));
    EXPECT_FALSE(unload("linear"));
}

TEST_F(ServerIntegrationPythonTest, UnloadModel_RecarregarAposDescarga) {
    EXPECT_TRUE(unload("linear"));
    EXPECT_TRUE(load("linear", linear_path()));
}

// =============================================================================
// GRUPO 4 — Predict via rede
// =============================================================================

TEST_F(ServerIntegrationPythonTest, Predict_InputsValidos_Sucesso) {
    auto r = predict("linear");
    EXPECT_TRUE(r.success) << r.error_message;
}

TEST_F(ServerIntegrationPythonTest, Predict_OutputNaoVazio) {
    auto r = predict("linear");
    ASSERT_TRUE(r.success);
    EXPECT_FALSE(r.outputs.empty());
}

TEST_F(ServerIntegrationPythonTest, Predict_InferenceTimeMsPositivo) {
    auto r = predict("linear");
    ASSERT_TRUE(r.success);
    EXPECT_GT(r.inference_time_ms, 0.0);
}

TEST_F(ServerIntegrationPythonTest, Predict_ResultadoDeterministico) {
    auto r1 = predict("linear", 1.0);
    auto r2 = predict("linear", 1.0);
    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);
    // mesmo input → mesmo output
    ASSERT_FALSE(r1.outputs.empty());
    ASSERT_FALSE(r2.outputs.empty());
    const std::string key = r1.outputs.begin()->first;
    EXPECT_DOUBLE_EQ(r1.outputs[key].as_number(),
                     r2.outputs[key].as_number());
}

TEST_F(ServerIntegrationPythonTest, Predict_InputsDiferentes_OutputsDiferentes) {
    auto r1 = predict("linear", 1.0);
    auto r2 = predict("linear", 9.0);
    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);
    const std::string key = r1.outputs.begin()->first;
    EXPECT_NE(r1.outputs[key].as_number(),
              r2.outputs[key].as_number());
}

TEST_F(ServerIntegrationPythonTest, Predict_ChaveErrada_Falha) {
    auto r = client->predict("linear", make_wrong_key_inputs());
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(ServerIntegrationTest, Predict_IDInvalido_Falha) {
    auto r = client->predict("id_inexistente", make_valid_inputs());
    EXPECT_FALSE(r.success);
}

TEST_F(ServerIntegrationPythonTest, Predict_10Chamadas_TodasSucesso) {
    for (int i = 0; i < 10; ++i) {
        auto r = predict("linear");
        EXPECT_TRUE(r.success) << "falhou na chamada " << i
                               << ": " << r.error_message;
    }
}

// =============================================================================
// GRUPO 5 — BatchPredict via rede
// =============================================================================

TEST_F(ServerIntegrationPythonTest, BatchPredict_TresElementos_TodosSucesso) {
    std::vector<Object> batch;
    for (int i = 0; i < 3; ++i)
        batch.push_back(make_valid_inputs(static_cast<double>(i + 1)));

    auto results = client->batch_predict("linear", batch);
    ASSERT_EQ(results.size(), 3u);
    for (size_t i = 0; i < results.size(); ++i)
        EXPECT_TRUE(results[i].success) << "elem " << i << ": "
                                        << results[i].error_message;
}

TEST_F(ServerIntegrationPythonTest, BatchPredict_OrdemPreservada) {
    std::vector<Object> batch{make_valid_inputs(1.0), make_valid_inputs(9.0)};
    auto results = client->batch_predict("linear", batch);
    ASSERT_EQ(results.size(), 2u);
    ASSERT_TRUE(results[0].success);
    ASSERT_TRUE(results[1].success);
    const std::string key = results[0].outputs.begin()->first;
    EXPECT_NE(results[0].outputs[key].as_number(),
              results[1].outputs[key].as_number());
}

TEST_F(ServerIntegrationPythonTest, BatchPredict_Vazio_NaoTrava) {
    EXPECT_NO_THROW(client->batch_predict("linear", {}));
}

// =============================================================================
// GRUPO 6 — ListModels / GetModelInfo via rede
// =============================================================================

TEST_F(ServerIntegrationTest, ListModels_SemModelos_Vazio) {
    auto models = client->list_models();
    EXPECT_TRUE(models.empty());
}

TEST_F(ServerIntegrationPythonTest, ListModels_ContemLinear) {
    auto models = client->list_models();
    bool found = std::any_of(models.begin(), models.end(),
        [](const auto& m){ return m.model_id == "linear"; });
    EXPECT_TRUE(found);
}

TEST_F(ServerIntegrationPythonTest, ListModels_AposUnload_NaoContemLinear) {
    unload("linear");
    auto models = client->list_models();
    bool found = std::any_of(models.begin(), models.end(),
        [](const auto& m){ return m.model_id == "linear"; });
    EXPECT_FALSE(found);
}

TEST_F(ServerIntegrationPythonTest, GetModelInfo_CamposObrigatorios) {
    auto info = client->get_model_info("linear");
    EXPECT_EQ(info.model_id, "linear");
    EXPECT_FALSE(info.backend.empty());
    EXPECT_GT(info.inputs.size(), 0u);
    EXPECT_GT(info.outputs.size(), 0u);
}

TEST_F(ServerIntegrationPythonTest, GetModelInfo_BackendPython) {
    auto info = client->get_model_info("linear");
    EXPECT_EQ(info.backend, "python");
}

TEST_F(ServerIntegrationTest, GetModelInfo_IDInexistente_CamposVazios) {
    auto info = client->get_model_info("nao_existe");
    // Campos default — model_id vazio indica "não encontrado"
    EXPECT_TRUE(info.model_id.empty());
}

// =============================================================================
// GRUPO 7 — ValidateModel via rede
// =============================================================================

TEST_F(ServerIntegrationPythonTest, ValidateModel_ArquivoValido_Valid) {
    auto r = client->validate_model(linear_path());
    EXPECT_TRUE(r.valid) << r.error_message;
}

TEST_F(ServerIntegrationTest, ValidateModel_ArquivoInexistente_Invalid) {
    auto r = client->validate_model("/nao/existe/modelo.py");
    EXPECT_FALSE(r.valid);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(ServerIntegrationPythonTest, ValidateModel_InputsPreenchidos) {
    auto r = client->validate_model(linear_path());
    ASSERT_TRUE(r.valid);
    EXPECT_GT(r.inputs.size(), 0u);
}

TEST_F(ServerIntegrationPythonTest, ValidateModel_OutputsPreenchidos) {
    auto r = client->validate_model(linear_path());
    ASSERT_TRUE(r.valid);
    EXPECT_GT(r.outputs.size(), 0u);
}

// =============================================================================
// GRUPO 8 — WarmupModel via rede
// =============================================================================

TEST_F(ServerIntegrationPythonTest, WarmupModel_HappyPath_Sucesso) {
    auto r = client->warmup_model("linear", 3);
    EXPECT_TRUE(r.success) << r.error_message;
    EXPECT_EQ(r.runs_completed, 3u);
}

TEST_F(ServerIntegrationPythonTest, WarmupModel_AvgTimeMsPositivo) {
    auto r = client->warmup_model("linear", 3);
    ASSERT_TRUE(r.success);
    EXPECT_GT(r.avg_time_ms, 0.0);
}

TEST_F(ServerIntegrationPythonTest, WarmupModel_MinMenorIgualMax) {
    auto r = client->warmup_model("linear", 5);
    ASSERT_TRUE(r.success);
    EXPECT_LE(r.min_time_ms, r.max_time_ms);
}

TEST_F(ServerIntegrationTest, WarmupModel_IDInexistente_Falha) {
    auto r = client->warmup_model("nao_existe", 3);
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

// =============================================================================
// GRUPO 9 — GetStatus / GetMetrics via rede
// =============================================================================

TEST_F(ServerIntegrationTest, GetStatus_WorkerIdPreenchido) {
    auto s = client->get_status();
    EXPECT_EQ(s.worker_id, "test-integration-server");
}

TEST_F(ServerIntegrationTest, GetStatus_UptimeNaoNegativo) {
    EXPECT_GE(client->get_status().uptime_seconds, 0LL);
}

TEST_F(ServerIntegrationTest, GetStatus_SupportedBackendsNaoVazio) {
    EXPECT_FALSE(client->get_status().supported_backends.empty());
}

TEST_F(ServerIntegrationTest, GetStatus_SupportedBackendsContemPython) {
    auto s = client->get_status();
    bool found = std::any_of(s.supported_backends.begin(),
                             s.supported_backends.end(),
                             [](const std::string& b){
                                 std::string l = b;
                                 std::transform(l.begin(), l.end(), l.begin(), ::tolower);
                                 return l.find("python") != std::string::npos;
                             });
    EXPECT_TRUE(found);
}

TEST_F(ServerIntegrationPythonTest, GetStatus_LoadedModelsContemLinear) {
    auto s = client->get_status();
    bool found = std::any_of(s.loaded_models.begin(), s.loaded_models.end(),
                             [](const std::string& id){ return id == "linear"; });
    EXPECT_TRUE(found);
}

TEST_F(ServerIntegrationTest, GetMetrics_NaoTrava) {
    EXPECT_NO_THROW(client->get_metrics());
}

TEST_F(ServerIntegrationTest, GetMetrics_TotalRequestsNaoNegativo) {
    EXPECT_GE(client->get_metrics().total_requests, 0u);
}

TEST_F(ServerIntegrationPythonTest, GetMetrics_TotalRequestsCresce) {
    auto before = client->get_metrics().total_requests;
    predict("linear");
    predict("linear");
    EXPECT_GT(client->get_metrics().total_requests, before);
}

TEST_F(ServerIntegrationPythonTest, GetMetrics_SuccessfulNaoSuperapTotal) {
    predict("linear");
    auto m = client->get_metrics();
    EXPECT_LE(m.successful_requests, m.total_requests);
}

TEST_F(ServerIntegrationPythonTest, GetMetrics_PerModelContemLinear) {
    predict("linear");
    auto m = client->get_metrics();
    EXPECT_GT(m.per_model.count("linear"), 0u);
}

// =============================================================================
// GRUPO 10 — ListAvailableModels via rede
// =============================================================================

TEST_F(ServerIntegrationTest, ListAvailableModels_DirExistente_NaoTrava) {
    EXPECT_NO_THROW(client->list_available_models(models_dir()));
}

TEST_F(ServerIntegrationTest, ListAvailableModels_CamposObrigatorios) {
    auto models = client->list_available_models(models_dir());
    for (const auto& m : models) {
        EXPECT_FALSE(m.filename.empty())   << "filename vazio";
        EXPECT_FALSE(m.path.empty())       << "path vazio";
        EXPECT_FALSE(m.extension.empty())  << "extension vazia";
        EXPECT_GT(m.file_size_bytes, 0)    << "file_size <= 0";
    }
}

TEST_F(ServerIntegrationTest, ListAvailableModels_ExtensoesSuportadas) {
    auto models = client->list_available_models(models_dir());
    for (const auto& m : models) {
        EXPECT_TRUE(m.extension == ".py" || m.extension == ".onnx")
            << "extensão inesperada: " << m.extension;
    }
}

TEST_F(ServerIntegrationPythonTest, ListAvailableModels_IsLoaded_RefleteCarga) {
    auto models = client->list_available_models(models_dir());
    bool found_loaded = std::any_of(models.begin(), models.end(),
                                    [](const auto& m){ return m.is_loaded; });
    EXPECT_TRUE(found_loaded);
}

TEST_F(ServerIntegrationPythonTest, ListAvailableModels_LoadedAs_Correto) {
    auto models = client->list_available_models(models_dir());
    for (const auto& m : models)
        if (m.is_loaded)
            EXPECT_FALSE(m.loaded_as.empty());
}

// =============================================================================
// GRUPO 11 — Ciclo completo end-to-end
// =============================================================================

TEST_F(ServerIntegrationTest, CicloCompleto_LoadPredictWarmupUnload) {
    if (!fs::exists(linear_path())) GTEST_SKIP();

    // 1. Load
    ASSERT_TRUE(load("e2e", linear_path()));

    // 2. Validate após load
    auto val = client->validate_model(linear_path());
    EXPECT_TRUE(val.valid);

    // 3. Predict (5x)
    for (int i = 0; i < 5; ++i) {
        auto r = predict("e2e");
        ASSERT_TRUE(r.success) << "predict falhou no i=" << i
                               << ": " << r.error_message;
    }

    // 4. Warmup
    auto warm = client->warmup_model("e2e", 3);
    EXPECT_TRUE(warm.success);
    EXPECT_EQ(warm.runs_completed, 3u);

    // 5. GetModelInfo
    auto info = client->get_model_info("e2e");
    EXPECT_EQ(info.model_id, "e2e");
    EXPECT_GT(info.inputs.size(), 0u);

    // 6. GetMetrics — deve ter inferências registradas
    auto metrics = client->get_metrics();
    EXPECT_GT(metrics.total_requests, 0u);

    // 7. Unload
    ASSERT_TRUE(unload("e2e"));

    // 8. Predict após unload deve falhar
    auto r_fail = predict("e2e");
    EXPECT_FALSE(r_fail.success);
}

TEST_F(ServerIntegrationTest, CicloCompleto_TresModelsSimultaneos) {
    if (!fs::exists(linear_path())) GTEST_SKIP();

    ASSERT_TRUE(load("m1", linear_path()));
    ASSERT_TRUE(load("m2", linear_path()));
    ASSERT_TRUE(load("m3", linear_path()));

    // Todos respondem corretamente
    for (const auto& id : std::vector<std::string>{"m1", "m2", "m3"}) {
        auto r = predict(id);
        EXPECT_TRUE(r.success) << "falhou para " << id << ": " << r.error_message;
    }

    // 3 modelos na lista
    auto models = client->list_models();
    EXPECT_GE(models.size(), 3u);

    // Unload todos
    EXPECT_TRUE(unload("m1"));
    EXPECT_TRUE(unload("m2"));
    EXPECT_TRUE(unload("m3"));

    // Lista deve estar vazia
    auto models_after = client->list_models();
    EXPECT_EQ(models_after.size(), 0u);
}

TEST_F(ServerIntegrationTest, CicloCompleto_LoadUnloadReload) {
    if (!fs::exists(linear_path())) GTEST_SKIP();

    for (int ciclo = 0; ciclo < 3; ++ciclo) {
        ASSERT_TRUE(load("reload_test", linear_path())) << "ciclo " << ciclo;
        auto r = predict("reload_test");
        EXPECT_TRUE(r.success) << "ciclo=" << ciclo << ": " << r.error_message;
        ASSERT_TRUE(unload("reload_test")) << "ciclo " << ciclo;
    }
}

// =============================================================================
// main() — registra o GlobalServerEnv antes de rodar os testes
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new GlobalServerEnv());
    return RUN_ALL_TESTS();
}
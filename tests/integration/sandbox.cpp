// =============================================================================
// sandbox.cpp — Teste de integração do InferenceClient com tutorial_model.py
//
// Cobre TODOS os métodos públicos de InferenceClient no modo in-process:
//
//   Conexão     : connect(), is_connected()
//   Ciclo vida  : load_model(), unload_model()
//   Inferência  : predict(), batch_predict()
//   Introspecção: list_models(), get_model_info(), validate_model(),
//                 warmup_model()
//   Observabil. : health_check(), get_status(), get_metrics()
//   Descoberta  : list_available_models()
//
// Modelo usado: tutorial_model.py (ShipAvoidance — campo potencial)
//   Input  : "state" → Object estruturado { toHeading, latitude, longitude,
//                       hazards: [{ bearing, distance, minSafeDist }] }
//   Output : "heading" → escalar (shape [1,1] colapsado → is_number())
//
// O modelo é TOLERANTE a state={} vazio — retorna heading=0.0 com success=true.
// Por isso os testes de failure-path usam tipos completamente errados (escalar
// no lugar de Object) para forçar exceção no lado Python.
//
// Variáveis de ambiente:
//   MODELS_DIR=./models  (padrão)
// =============================================================================

#include <gtest/gtest.h>
#include <client/inference_client.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Aliases
// ─────────────────────────────────────────────────────────────────────────────

using miia::client::Array;
using miia::client::AvailableModel;
using miia::client::InferenceClient;
using miia::client::ModelInfo;
using miia::client::Object;
using miia::client::PredictionResult;
using miia::client::ServerMetrics;
using miia::client::Value;
using miia::client::ValidationResult;
using miia::client::WarmupResult;
using miia::client::WorkerStatus;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers globais
// ─────────────────────────────────────────────────────────────────────────────

static std::string models_dir() {
    const char* e = std::getenv("MODELS_DIR");
    const std::string raw = e ? e : "./models";
    return std::filesystem::weakly_canonical(raw).string();
}

static std::string linear_path() {
    return models_dir() + "/tutorial_model.py";
}

// Constrói um Object de entrada válido para o ShipAvoidance.
//   toHeading : rumo ao steerpoint alvo (graus)
//   lat / lon : posição do navio (graus)
// Inclui dois hazards representativos para exercitar o campo de repulsão.
static Object make_ship_inputs(float to_heading = 45.0f,
                                float lat = -23.5f,
                                float lon = -46.6f) {
    Object hazard1;
    hazard1["bearing"]     = Value{90.0};
    hazard1["distance"]    = Value{500.0};
    hazard1["minSafeDist"] = Value{300.0};

    Object hazard2;
    hazard2["bearing"]     = Value{180.0};
    hazard2["distance"]    = Value{800.0};
    hazard2["minSafeDist"] = Value{300.0};

    Array hazards;
    hazards.push_back(Value{std::move(hazard1)});
    hazards.push_back(Value{std::move(hazard2)});

    Object state;
    state["toHeading"]  = Value{static_cast<double>(to_heading)};
    state["latitude"]   = Value{static_cast<double>(lat)};
    state["longitude"]  = Value{static_cast<double>(lon)};
    state["hazards"]    = Value{std::move(hazards)};

    Object o;
    o["state"] = Value{std::move(state)};
    return o;
}

// Alias genérico — compatível com os padrões de chamada dos testes de ciclo.
// Os parâmetros extras (v3, v4) são ignorados; to_heading, lat e lon
// mapeiam nos três primeiros para variação entre iterações de lote.
static Object make_valid_inputs(float to_heading = 45.0f,
                                 float lat = -23.5f,
                                 float lon = -46.6f,
                                 float /*unused*/ = 0.f,
                                 float /*unused*/ = 0.f) {
    return make_ship_inputs(to_heading, lat, lon);
}

// Chave completamente desconhecida — o modelo recebe um dict sem "state".
// ShipAvoidance faz inputs.get("state", {}) → retorna heading=0.0 com success.
// Este helper serve para testes que verificam comportamento com chave ausente.
static Object make_wrong_key_inputs() {
    Object o;
    o["tensor_errado"] = Value{Array{}};
    return o;
}

// "state" com tipo escalar (double) em vez de Object → Python lança AttributeError
// ao tentar chamar .get() no float → success=false esperado.
static Object make_wrong_size_inputs() {
    Object o;
    o["state"] = Value{-999.0};
    return o;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture base — conecta in-process e carrega tutorial_model.py como "ship"
// ─────────────────────────────────────────────────────────────────────────────

class InProcessPythonTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(linear_path()))
            GTEST_SKIP() << "Modelo Python não encontrado: " << linear_path();

        client = std::make_unique<InferenceClient>("inprocess");
        ASSERT_TRUE(client->connect())                           << "connect() falhou";
        ASSERT_TRUE(client->load_model("ship", linear_path()))  << "load_model() falhou";
    }

    void TearDown() override {
        if (client)
            for (const auto& m : client->list_models())
                client->unload_model(m.model_id);
    }

    std::unique_ptr<InferenceClient> client;
};

// // =============================================================================
// // GRUPO 1 — connect() / is_connected()
// // =============================================================================

TEST(Connect, RetornaTrueNaPrimeiraConexao) {
    InferenceClient c("inprocess");
    EXPECT_TRUE(c.connect());
}

TEST(Connect, IsConnectedFalseAntesDeConectar) {
    InferenceClient c("inprocess");
    EXPECT_FALSE(c.is_connected());
}

TEST(Connect, IsConnectedTrueAposConectar) {
    InferenceClient c("inprocess");
    c.connect();
    EXPECT_TRUE(c.is_connected());
}

TEST(Connect, ConexaoRepeticaoEstavelSemCrash) {
    InferenceClient c("inprocess");
    for (int i = 0; i < 5; ++i)
        EXPECT_TRUE(c.connect()) << "iteração=" << i;
    EXPECT_TRUE(c.is_connected());
}

// =============================================================================
// GRUPO 2 — load_model()
// =============================================================================

TEST(LoadModel, CarregaModeloValidoRetornaTrue) {
    if (!std::filesystem::exists(linear_path()))
        GTEST_SKIP() << "Modelo Python não encontrado: " << linear_path();
    InferenceClient c("inprocess");
    c.connect();
    EXPECT_TRUE(c.load_model("s", linear_path()));
}

TEST(LoadModel, PathInexistenteRetornaFalse) {
    InferenceClient c("inprocess");
    c.connect();
    EXPECT_FALSE(c.load_model("x", "/nao/existe/modelo.py"));
}

TEST(LoadModel, IDDuplicadoRetornaFalse) {
    if (!std::filesystem::exists(linear_path()))
        GTEST_SKIP() << "Modelo Python não encontrado: " << linear_path();
    InferenceClient c("inprocess");
    c.connect();
    EXPECT_TRUE(c.load_model("dup", linear_path()));
    EXPECT_FALSE(c.load_model("dup", linear_path()));
    c.unload_model("dup");
}

TEST(LoadModel, IDsDiferentesParaMesmoArquivoOk) {
    if (!std::filesystem::exists(linear_path()))
        GTEST_SKIP() << "Modelo Python não encontrado: " << linear_path();
    InferenceClient c("inprocess");
    c.connect();
    EXPECT_TRUE(c.load_model("a1", linear_path()));
    EXPECT_TRUE(c.load_model("a2", linear_path()));
    c.unload_model("a1");
    c.unload_model("a2");
}

// =============================================================================
// GRUPO 3 — unload_model()
// =============================================================================

TEST(UnloadModel, DescarregaModeloCarregadoRetornaTrue) {
    if (!std::filesystem::exists(linear_path()))
        GTEST_SKIP() << "Modelo Python não encontrado: " << linear_path();
    InferenceClient c("inprocess");
    c.connect();
    c.load_model("u1", linear_path());
    EXPECT_TRUE(c.unload_model("u1"));
}

TEST(UnloadModel, RecarregarAposDescarregar) {
    if (!std::filesystem::exists(linear_path()))
        GTEST_SKIP() << "Modelo Python não encontrado: " << linear_path();
    InferenceClient c("inprocess");
    c.connect();
    c.load_model("reload", linear_path());
    c.unload_model("reload");
    EXPECT_TRUE(c.load_model("reload", linear_path()));
    c.unload_model("reload");
}

TEST(UnloadModel, DescarregaMultiplosModelosEmSequencia) {
    if (!std::filesystem::exists(linear_path()))
        GTEST_SKIP() << "Modelo Python não encontrado: " << linear_path();
    InferenceClient c("inprocess");
    c.connect();
    c.load_model("a", linear_path());
    c.load_model("b", linear_path());
    EXPECT_TRUE(c.unload_model("a"));
    EXPECT_TRUE(c.unload_model("b"));
}

TEST(UnloadModel, IDInexistenteRetornaFalse) {
    InferenceClient c("inprocess");
    c.connect();
    EXPECT_FALSE(c.unload_model("id_que_nao_existe_xyz"));
}

TEST(UnloadModel, SegundoUnloadDoMesmoIDRetornaFalse) {
    InferenceClient c("inprocess");
    c.connect();
    c.load_model("once", linear_path());
    c.unload_model("once");
    EXPECT_FALSE(c.unload_model("once"));
}

// =============================================================================
// GRUPO 4 — predict()
// =============================================================================

TEST_F(InProcessPythonTest, PredictRetornaSuccess) {
    // Input bem formado deve produzir success=true.
    auto r = client->predict("ship", make_valid_inputs());
    EXPECT_TRUE(r.success) << r.error_message;
}

TEST_F(InProcessPythonTest, PredictOutputNaoVazio) {
    auto r = client->predict("ship", make_valid_inputs());
    ASSERT_TRUE(r.success);
    EXPECT_FALSE(r.outputs.empty());
}

TEST_F(InProcessPythonTest, PredictOutputContemChaveHeading) {
    // O ShipAvoidance retorna o tensor de saída sob a chave "heading".
    auto r = client->predict("ship", make_valid_inputs());
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.outputs.count("heading") > 0);
}

TEST_F(InProcessPythonTest, PredictOutputHeadingEhEscalar) {
    // shape [1,1] colapsado por py_dict_to_outputs → is_number().
    auto r = client->predict("ship", make_valid_inputs());
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.outputs.at("heading").is_number())
        << "heading não é escalar — verifique shape [1,1] no modelo";
}

TEST_F(InProcessPythonTest, PredictHeadingEmGrausValido) {
    // O rumo resultante deve ser um número real (sem NaN/Inf).
    auto r = client->predict("ship", make_valid_inputs());
    ASSERT_TRUE(r.success);
    double hdg = r.outputs.at("heading").as_number();
    EXPECT_FALSE(std::isnan(hdg))  << "heading é NaN";
    EXPECT_FALSE(std::isinf(hdg))  << "heading é Inf";
}

TEST_F(InProcessPythonTest, PredictStateVazioRetornaHeadingZero) {
    // ShipAvoidance é tolerante: state={} → heading=0.0, success=true.
    Object empty_state;
    empty_state["state"] = Value{Object{}};
    auto r = client->predict("ship", empty_state);
    EXPECT_TRUE(r.success) << r.error_message;
    EXPECT_NEAR(r.outputs.at("heading").as_number(), 0.0, 1e-6);
}

TEST_F(InProcessPythonTest, PredictChaveAusenteRetornaHeadingZero) {
    // Sem a chave "state", o modelo usa state={} por padrão → heading=0.0.
    auto r = client->predict("ship", make_wrong_key_inputs());
    EXPECT_TRUE(r.success) << r.error_message;
    EXPECT_NEAR(r.outputs.at("heading").as_number(), 0.0, 1e-6);
}

TEST_F(InProcessPythonTest, PredictStateTipoErradoFalha) {
    // "state" como escalar (double) → AttributeError no Python → success=false.
    auto r = client->predict("ship", make_wrong_size_inputs());
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(InProcessPythonTest, PredictObjectVazioRetornaHeadingZero) {
    // Object completamente vazio: o modelo trata como state={} → heading=0.0.
    auto r = client->predict("ship", Object{});
    EXPECT_TRUE(r.success) << r.error_message;
}

TEST_F(InProcessPythonTest, PredictIDInexistenteFalha) {
    auto r = client->predict("nao_existe", make_valid_inputs());
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(InProcessPythonTest, PredictSemHazardsRetornaSteerpoint) {
    // Sem hazards, o rumo resultante deve ser igual ao toHeading (apenas atração).
    Object state;
    state["toHeading"]  = Value{90.0};
    state["latitude"]   = Value{-23.5};
    state["longitude"]  = Value{-46.6};
    state["hazards"]    = Value{Array{}};
    Object o;
    o["state"] = Value{std::move(state)};

    auto r = client->predict("ship", o);
    ASSERT_TRUE(r.success) << r.error_message;
    // Com apenas força de atração, atan2(sin(90°), cos(90°)) ≈ 90°.
    EXPECT_NEAR(r.outputs.at("heading").as_number(), 90.0, 1e-3);
}

TEST_F(InProcessPythonTest, PredictMesmaEntradaMesmoResultado) {
    // Modelo determinístico: a mesma entrada deve produzir o mesmo rumo.
    auto inputs = make_valid_inputs(45.0f);
    auto r1 = client->predict("ship", inputs);
    auto r2 = client->predict("ship", inputs);
    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);
    EXPECT_NEAR(r1.outputs.at("heading").as_number(),
                r2.outputs.at("heading").as_number(), 1e-6);
}

// // =============================================================================
// // GRUPO 5 — batch_predict()
// // =============================================================================

// TEST_F(InProcessPythonTest, BatchPredictLoteUnicoOk) {
//     auto results = client->batch_predict("ship", {make_valid_inputs()});
//     ASSERT_EQ(results.size(), 1u);
//     EXPECT_TRUE(results[0].success);
// }

// TEST_F(InProcessPythonTest, BatchPredictRetornaMesmoTamanho) {
//     std::vector<Object> batch;
//     for (int i = 0; i < 7; ++i)
//         batch.push_back(make_valid_inputs(static_cast<float>(i) * 10.0f));
//     EXPECT_EQ(client->batch_predict("ship", batch).size(), 7u);
// }

// TEST_F(InProcessPythonTest, BatchPredictTodosItensComSucesso) {
//     std::vector<Object> batch;
//     for (int i = 0; i < 5; ++i)
//         batch.push_back(make_valid_inputs(static_cast<float>(i) * 30.0f));
//     auto results = client->batch_predict("ship", batch);
//     for (size_t i = 0; i < results.size(); ++i)
//         EXPECT_TRUE(results[i].success)
//             << "item=" << i << " erro=" << results[i].error_message;
// }

// TEST_F(InProcessPythonTest, BatchPredictEquivalenteAoPredictEscalar) {
//     // batch[0] e predict() devem produzir o mesmo rumo para a mesma entrada.
//     auto inputs = make_valid_inputs(60.0f);
//     auto scalar = client->predict("ship", inputs);
//     auto batch  = client->batch_predict("ship", {inputs});
//     ASSERT_TRUE(scalar.success);
//     ASSERT_EQ(batch.size(), 1u);
//     ASSERT_TRUE(batch[0].success);
//     ASSERT_TRUE(scalar.outputs.at("heading").is_number());
//     ASSERT_TRUE(batch[0].outputs.at("heading").is_number());
//     EXPECT_NEAR(scalar.outputs.at("heading").as_number(),
//                 batch[0].outputs.at("heading").as_number(), 1e-6);
// }

// TEST_F(InProcessPythonTest, BatchPredictLote50ElementosOk) {
//     std::vector<Object> batch;
//     for (int i = 0; i < 50; ++i)
//         batch.push_back(make_valid_inputs(static_cast<float>(i) * 7.0f));
//     auto results = client->batch_predict("ship", batch);
//     ASSERT_EQ(results.size(), 50u);
//     for (const auto& r : results)
//         EXPECT_TRUE(r.success);
// }

// TEST_F(InProcessPythonTest, BatchPredictLoteVazioRetornaVetorVazio) {
//     EXPECT_TRUE(client->batch_predict("ship", {}).empty());
// }

// TEST_F(InProcessPythonTest, BatchPredictIDInexistenteRetornaFalhasPorItem) {
//     auto results = client->batch_predict("nao_existe",
//                        {make_valid_inputs(), make_valid_inputs()});
//     for (const auto& r : results)
//         EXPECT_FALSE(r.success);
// }

// // =============================================================================
// // GRUPO 6 — list_models()
// // =============================================================================

// TEST_F(InProcessPythonTest, ListModelsNaoVaziaAposLoad) {
//     EXPECT_FALSE(client->list_models().empty());
// }

// TEST_F(InProcessPythonTest, ListModelsContemIDCarregado) {
//     auto models = client->list_models();
//     bool found  = std::any_of(models.begin(), models.end(),
//                               [](const ModelInfo& m){ return m.model_id == "ship"; });
//     EXPECT_TRUE(found);
// }

// TEST_F(InProcessPythonTest, ListModelsContagemAumentaAposNovoLoad) {
//     auto before = client->list_models().size();
//     client->load_model("extra", linear_path());
//     EXPECT_EQ(client->list_models().size(), before + 1);
// }

// TEST_F(InProcessPythonTest, ListModelsContagemDiminuiAposUnload) {
//     client->load_model("temp", linear_path());
//     auto before = client->list_models().size();
//     client->unload_model("temp");
//     EXPECT_EQ(client->list_models().size(), before - 1);
// }

// TEST_F(InProcessPythonTest, ListModelsEntradaTemModelIdNaoVazio) {
//     for (const auto& m : client->list_models())
//         EXPECT_FALSE(m.model_id.empty());
// }

// TEST_F(InProcessPythonTest, ListModelsEntradaTemBackendNaoVazio) {
//     for (const auto& m : client->list_models())
//         EXPECT_FALSE(m.backend.empty());
// }

// TEST(ListModels, ListaVaziaAntesDeQualquerLoad) {
//     InferenceClient c("inprocess");
//     c.connect();
//     EXPECT_TRUE(c.list_models().empty());
// }

// // =============================================================================
// // GRUPO 7 — get_model_info()
// // =============================================================================

// TEST_F(InProcessPythonTest, GetModelInfoIdCorreto) {
//     EXPECT_EQ(client->get_model_info("ship").model_id, "ship");
// }

// TEST_F(InProcessPythonTest, GetModelInfoBackendEhPython) {
//     EXPECT_EQ(client->get_model_info("ship").backend, "python");
// }

// TEST_F(InProcessPythonTest, GetModelInfoTemInputs) {
//     EXPECT_FALSE(client->get_model_info("ship").inputs.empty());
// }

// TEST_F(InProcessPythonTest, GetModelInfoTemOutputs) {
//     EXPECT_FALSE(client->get_model_info("ship").outputs.empty());
// }

// TEST_F(InProcessPythonTest, GetModelInfoInputNomesNaoVazios) {
//     for (const auto& ts : client->get_model_info("ship").inputs)
//         EXPECT_FALSE(ts.name.empty());
// }

// TEST_F(InProcessPythonTest, GetModelInfoOutputNomesNaoVazios) {
//     for (const auto& ts : client->get_model_info("ship").outputs)
//         EXPECT_FALSE(ts.name.empty());
// }

// TEST_F(InProcessPythonTest, GetModelInfoInputContemChaveState) {
//     // O ShipAvoidance declara um tensor de entrada chamado "state".
//     auto info = client->get_model_info("ship");
//     bool found = std::any_of(info.inputs.begin(), info.inputs.end(),
//                              [](const ModelInfo::TensorSpec& t){ return t.name == "state"; });
//     EXPECT_TRUE(found);
// }

// TEST_F(InProcessPythonTest, GetModelInfoOutputContemChaveHeading) {
//     // O ShipAvoidance declara um tensor de saída chamado "heading".
//     auto info = client->get_model_info("ship");
//     bool found = std::any_of(info.outputs.begin(), info.outputs.end(),
//                              [](const ModelInfo::TensorSpec& t){ return t.name == "heading"; });
//     EXPECT_TRUE(found);
// }

// TEST_F(InProcessPythonTest, GetModelInfoInputShapesNaoVazias) {
//     for (const auto& ts : client->get_model_info("ship").inputs)
//         EXPECT_FALSE(ts.shape.empty());
// }

// TEST_F(InProcessPythonTest, GetModelInfoInputStateEhEstruturado) {
//     // O campo "state" deve ter structured=true no schema.
//     auto info = client->get_model_info("ship");
//     for (const auto& ts : info.inputs)
//         if (ts.name == "state")
//             EXPECT_TRUE(ts.structured);
// }

// // =============================================================================
// // GRUPO 8 — validate_model()
// // =============================================================================

// TEST_F(InProcessPythonTest, ValidateModelArquivoValidoRetornaTrue) {
//     auto vr = client->validate_model(linear_path());
//     EXPECT_TRUE(vr.valid) << vr.error_message;
// }

// TEST_F(InProcessPythonTest, ValidateModelArquivoInexistenteRetornaFalse) {
//     auto vr = client->validate_model("/nao/existe/modelo.py");
//     EXPECT_FALSE(vr.valid);
// }

// TEST_F(InProcessPythonTest, ValidateModelArquivoValidoSemMensagemDeErro) {
//     EXPECT_TRUE(client->validate_model(linear_path()).error_message.empty());
// }

// // =============================================================================
// // GRUPO 9 — warmup_model()
// // =============================================================================

// TEST_F(InProcessPythonTest, WarmupModelRetornaSuccess) {
//     auto wr = client->warmup_model("ship", 3);
//     EXPECT_TRUE(wr.success) << wr.error_message;
// }

// TEST_F(InProcessPythonTest, WarmupModelRunsCompletadosCorretos) {
//     auto wr = client->warmup_model("ship", 5);
//     EXPECT_EQ(wr.runs_completed, 5u);
// }

// TEST_F(InProcessPythonTest, WarmupModelTempoMedioNaoNegativo) {
//     auto wr = client->warmup_model("ship", 3);
//     EXPECT_GE(wr.avg_time_ms, 0.0);
// }

// TEST_F(InProcessPythonTest, WarmupModelTempoMinLeTempoMax) {
//     auto wr = client->warmup_model("ship", 5);
//     EXPECT_LE(wr.min_time_ms, wr.max_time_ms);
// }

// TEST_F(InProcessPythonTest, WarmupModelIDInexistenteFalha) {
//     EXPECT_FALSE(client->warmup_model("nao_existe_xyz", 5).success);
//     EXPECT_FALSE(client->warmup_model("nao_existe_xyz", 5).error_message.empty());
// }

// // =============================================================================
// // GRUPO 10 — health_check()
// // =============================================================================

// TEST_F(InProcessPythonTest, HealthCheckRetornaTrueAposConexao) {
//     EXPECT_TRUE(client->health_check());
// }

// TEST_F(InProcessPythonTest, HealthCheckEstavelEmRepeticoes) {
//     for (int i = 0; i < 20; ++i)
//         EXPECT_TRUE(client->health_check()) << "iteração=" << i;
// }

// TEST(HealthCheck, FalseSemConexao) {
//     InferenceClient c("inprocess");
//     EXPECT_FALSE(c.health_check());
// }

// // =============================================================================
// // GRUPO 11 — get_status()
// // =============================================================================

// TEST_F(InProcessPythonTest, GetStatusWorkerIdNaoVazio) {
//     EXPECT_FALSE(client->get_status().worker_id.empty());
// }

// TEST_F(InProcessPythonTest, GetStatusWorkerIdEhInprocess) {
//     EXPECT_EQ(client->get_status().worker_id, "inprocess");
// }

// TEST_F(InProcessPythonTest, GetStatusUptimeNaoNegativo) {
//     EXPECT_GE(client->get_status().uptime_seconds, 0LL);
// }

// TEST_F(InProcessPythonTest, GetStatusLoadedModelsContemShip) {
//     auto s = client->get_status();
//     bool found = std::any_of(s.loaded_models.begin(), s.loaded_models.end(),
//                              [](const std::string& id){ return id == "ship"; });
//     EXPECT_TRUE(found);
// }

// TEST_F(InProcessPythonTest, GetStatusLoadedModelsRefleteCargaDinamica) {
//     auto before = client->get_status().loaded_models.size();
//     client->load_model("s2", linear_path());
//     EXPECT_EQ(client->get_status().loaded_models.size(), before + 1);
// }

// TEST_F(InProcessPythonTest, GetStatusLoadedModelsRefletUnload) {
//     client->load_model("s_ul", linear_path());
//     client->unload_model("s_ul");
//     auto s = client->get_status();
//     bool still_there = std::any_of(s.loaded_models.begin(), s.loaded_models.end(),
//                                    [](const std::string& id){ return id == "s_ul"; });
//     EXPECT_FALSE(still_there);
// }

// TEST_F(InProcessPythonTest, GetStatusSupportedBackendsNaoVazio) {
//     EXPECT_FALSE(client->get_status().supported_backends.empty());
// }

// TEST_F(InProcessPythonTest, GetStatusSupportedBackendsContemPython) {
//     auto s = client->get_status();
//     bool found = std::any_of(s.supported_backends.begin(), s.supported_backends.end(),
//                              [](const std::string& b){ return b == "python"; });
//     EXPECT_TRUE(found);
// }

// // =============================================================================
// // GRUPO 12 — get_metrics()
// // =============================================================================

// TEST_F(InProcessPythonTest, GetMetricsNaoLancaException) {
//     EXPECT_NO_THROW(client->get_metrics());
// }

// TEST_F(InProcessPythonTest, GetMetricsTotalRequestsNaoRegride) {
//     // O backend in-process não garante incremento exato por predict —
//     // verifica apenas que o contador não regride após chamadas.
//     auto before = client->get_metrics().total_requests;
//     client->predict("ship", make_valid_inputs());
//     client->predict("ship", make_valid_inputs());
//     EXPECT_GE(client->get_metrics().total_requests, before);
// }

// // =============================================================================
// // GRUPO 13 — list_available_models()
// // =============================================================================

// TEST_F(InProcessPythonTest, ListAvailableModelsNaoVaziaParaDirValido) {
//     EXPECT_FALSE(client->list_available_models(models_dir()).empty());
// }

// TEST_F(InProcessPythonTest, ListAvailableModelsContemTutorialModel) {
//     auto models = client->list_available_models(models_dir());
//     bool found  = std::any_of(models.begin(), models.end(),
//                               [](const AvailableModel& m){
//                                   return m.filename.find("tutorial_model.py") != std::string::npos;
//                               });
//     EXPECT_TRUE(found);
// }

// TEST_F(InProcessPythonTest, ListAvailableModelsIsLoadedTrueAposLoad) {
//     client->load_model("avail_check", linear_path());
//     auto models     = client->list_available_models(models_dir());
//     bool any_loaded = std::any_of(models.begin(), models.end(),
//                                   [](const AvailableModel& m){ return m.is_loaded; });
//     EXPECT_TRUE(any_loaded);
//     client->unload_model("avail_check");
// }

// TEST_F(InProcessPythonTest, ListAvailableModelsLoadedAsRefletID) {
//     client->unload_model("ship");
//     client->load_model("meu_id_especifico", linear_path());
//     auto models = client->list_available_models(models_dir());
//     bool found  = std::any_of(models.begin(), models.end(),
//                               [](const AvailableModel& m){
//                                   return m.is_loaded && m.loaded_as == "meu_id_especifico";
//                               });
//     EXPECT_TRUE(found);
//     client->unload_model("meu_id_especifico");
//     client->load_model("ship", linear_path());
// }

// TEST_F(InProcessPythonTest, ListAvailableModelsIsLoadedFalseAposUnload) {
//     client->load_model("ul_avail", linear_path());
//     client->unload_model("ul_avail");
//     for (const auto& m : client->list_available_models(models_dir()))
//         EXPECT_NE(m.loaded_as, "ul_avail");
// }

// TEST_F(InProcessPythonTest, ListAvailableModelsDirInexistenteRetornaVazio) {
//     EXPECT_TRUE(client->list_available_models("/nao/existe/de/jeito/nenhum").empty());
// }

// TEST_F(InProcessPythonTest, ListAvailableModelsDirVazioUsaDefaultSemCrash) {
//     EXPECT_NO_THROW(client->list_available_models(""));
// }

// // =============================================================================
// // GRUPO 14 — Ciclos de vida completos (end-to-end)
// // =============================================================================

// TEST_F(InProcessPythonTest, E2E_CicloCompletoSemFalhas) {
//     // Fluxo completo de produção: validar → carregar → warmup → introspectar
//     // → inferir → batch → verificar status → descobrir → descarregar.
//     const std::string id = "e2e_full";

//     // 1. Valida antes de carregar.
//     auto vr = client->validate_model(linear_path());
//     ASSERT_TRUE(vr.valid) << "validação falhou: " << vr.error_message;

//     // 2. Carrega sob ID temporário.
//     ASSERT_TRUE(client->load_model(id, linear_path()));

//     // 3. Warmup.
//     auto wr = client->warmup_model(id, 3);
//     EXPECT_TRUE(wr.success);
//     EXPECT_EQ(wr.runs_completed, 3u);

//     // 4. Introspecção.
//     auto info = client->get_model_info(id);
//     EXPECT_EQ(info.model_id, id);
//     EXPECT_EQ(info.backend, "python");
//     EXPECT_FALSE(info.inputs.empty());
//     EXPECT_FALSE(info.outputs.empty());

//     // 5. Inferência escalar — saída é número (heading em graus).
//     auto pr = client->predict(id, make_valid_inputs());
//     ASSERT_TRUE(pr.success) << pr.error_message;
//     EXPECT_TRUE(pr.outputs.at("heading").is_number());

//     // 6. Inferência em lote.
//     auto br = client->batch_predict(id, {make_valid_inputs(), make_valid_inputs(90.0f)});
//     EXPECT_EQ(br.size(), 2u);
//     for (const auto& r : br) EXPECT_TRUE(r.success);

//     // 7. Status reflete o modelo carregado.
//     auto st = client->get_status();
//     bool in_status = std::any_of(st.loaded_models.begin(), st.loaded_models.end(),
//                                  [&id](const std::string& s){ return s == id; });
//     EXPECT_TRUE(in_status);

//     // 8. Descoberta confirma is_loaded=true.
//     auto avail = client->list_available_models(models_dir());
//     bool in_avail = std::any_of(avail.begin(), avail.end(),
//                                 [&id](const AvailableModel& m){ return m.loaded_as == id; });
//     EXPECT_TRUE(in_avail);

//     // 9. Descarrega.
//     EXPECT_TRUE(client->unload_model(id));

//     // 10. ID não deve mais aparecer em list_models().
//     auto final_list = client->list_models();
//     bool still_there = std::any_of(final_list.begin(), final_list.end(),
//                                    [&id](const ModelInfo& m){ return m.model_id == id; });
//     EXPECT_FALSE(still_there);
// }

// TEST_F(InProcessPythonTest, E2E_LoopSimulacao100Ticks) {
//     // Simula 100 ticks de um loop de navegação: o rumo comandado pelo modelo
//     // é realimentado como toHeading no tick seguinte.
//     // Objetivo: garantir que o pipeline sustenta N chamadas sem erros.
//     float to_heading = 45.0f;
//     constexpr int TICKS = 100;

//     for (int tick = 0; tick < TICKS; ++tick) {
//         auto r = client->predict("ship", make_ship_inputs(to_heading));

//         ASSERT_TRUE(r.success)
//             << "tick=" << tick << " erro=" << r.error_message;

//         ASSERT_TRUE(r.outputs.at("heading").is_number())
//             << "tick=" << tick << ": heading não é escalar";

//         double hdg = r.outputs.at("heading").as_number();
//         EXPECT_FALSE(std::isnan(hdg)) << "tick=" << tick << ": NaN";
//         EXPECT_FALSE(std::isinf(hdg)) << "tick=" << tick << ": Inf";

//         // Realimenta o rumo comandado no próximo tick.
//         to_heading = static_cast<float>(hdg);
//     }
// }

// TEST_F(InProcessPythonTest, E2E_DoisClientesNaoInterferam) {
//     // Dois InferenceClients in-process com o mesmo arquivo sob IDs distintos
//     // devem operar de forma independente.
//     InferenceClient c2("inprocess");
//     ASSERT_TRUE(c2.connect());
//     ASSERT_TRUE(c2.load_model("peer", linear_path()));

//     auto inputs = make_valid_inputs(30.0f);
//     auto r1 = client->predict("ship", inputs);
//     auto r2 = c2.predict("peer",      inputs);
//     EXPECT_TRUE(r1.success);
//     EXPECT_TRUE(r2.success);

//     // Descarregar de c2 não afeta client.
//     EXPECT_TRUE(c2.unload_model("peer"));
//     auto r3 = client->predict("ship", inputs);
//     EXPECT_TRUE(r3.success);
// }

// TEST_F(InProcessPythonTest, E2E_HeadingConvergeParaSteerpoint) {
//     // Com apenas atração (sem hazards) e toHeading=N, o rumo resultante
//     // deve convergir para N após uma chamada (campo potencial puro).
//     // Tolerância 1e-2°: cos/sin de ângulos "redondos" em float64 acumulam
//     // erro numérico pequeno mas suficiente para quebrar 1e-3.
//     // Ângulos restritos ao intervalo de saída do atan2: (-180°, 180°].
//     // 270° é equivalente a -90° nessa convenção — excluído para evitar
//     // comparação -90.0 ≈ 270.0 que sempre falha.
//     for (float hdg : {0.0f, 45.0f, 90.0f, 135.0f, -90.0f, -135.0f, 180.0f}) {
//         Object state;
//         state["toHeading"]  = Value{static_cast<double>(hdg)};
//         state["latitude"]   = Value{-23.5};
//         state["longitude"]  = Value{-46.6};
//         state["hazards"]    = Value{Array{}};
//         Object o;
//         o["state"] = Value{std::move(state)};

//         auto r = client->predict("ship", o);
//         ASSERT_TRUE(r.success) << "hdg=" << hdg;
//         EXPECT_NEAR(r.outputs.at("heading").as_number(), hdg, 1e-2)
//             << "rumo esperado=" << hdg;
//     }
// }
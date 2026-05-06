// =============================================================================
// tests/unit/test_worker_server.cpp
//
// Testes unitários — WorkerServiceImpl
//
// Estratégia:
//   Instancia WorkerServiceImpl diretamente (sem gRPC Server) e chama
//   cada RPC via a interface de serviço, passando contexto nulo.
//   Não requer rede — todos os testes rodam in-process.
//
// Cobertura:
//   GRUPO 1  — Construção e destruição
//   GRUPO 2  — LoadModel (happy path + falhas)
//   GRUPO 3  — UnloadModel (happy path + falhas)
//   GRUPO 4  — Predict (happy path + tensores ausentes + ID inválido)
//   GRUPO 5  — BatchPredict (size 0, 1, N + ordem + contadores)
//   GRUPO 6  — ListModels (antes/depois de load/unload)
//   GRUPO 7  — GetModelInfo (campos obrigatórios + modelo ausente)
//   GRUPO 8  — ValidateModel (arquivo válido + inexistente + ext errada)
//   GRUPO 9  — WarmupModel (happy path + ID inexistente + runs=0)
//   GRUPO 10 — HealthCheck (sempre true)
//   GRUPO 11 — GetStatus (worker_id, uptime, loaded_models, backends)
//   GRUPO 12 — GetMetrics (contadores + crescimento após predict)
//   GRUPO 13 — ListAvailableModels (campos + is_loaded + dir inexistente)
//   GRUPO 14 — PredictStream (leitura/escrita em loop via mock)
//   GRUPO 15 — Contadores atômicos (total, sucesso, falha)
//   GRUPO 16 — Ciclo completo (load → predict → warmup → unload)
//
// Nomes reais dos campos proto (server.proto / common.proto):
//   UnloadModelResponse    → message (não error_message)
//   GetMetricsResponse     → worker_metrics (não metrics)
//   ModelRuntimeMetrics    → avg_inference_time_ms (não avg_time_ms)
//   GetStatusResponse      → metrics (WorkerMetrics), capabilities, loaded_model_ids
//   WorkerMetrics          → uptime_seconds, total_requests, ...
//   WorkerCapabilities     → supported_backends
//
// Dependências de build (tests/unit/meson.build):
//   test('unit_worker_server',
//       executable('test_unit_worker_server',
//           'test_worker_server.cpp',
//           include_directories: [server_inc, inference_inc, client_inc],
//           dependencies: [worker_server_dep, proto_dep,
//                          gtest_main_dep, gtest_dep],
//           install: false,
//       ),
//       suite:       'unit',
//       timeout:     120,
//       is_parallel: false,
//       env: {
//           'MODELS_DIR': project_root / 'models',
//           'LOG_LEVEL':  'ERROR',
//       },
//   )
// =============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "server/worker_server.hpp"
#include "client/value_convert.hpp"
#include "server.grpc.pb.h"
#include "common.pb.h"

namespace fs = std::filesystem;
using namespace mlinference::server;
using namespace mlinference::client;
using namespace mlinference;

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

/// Constrói PredictRequest com inputs válidos para simple_linear.py
/// (chave "input", vetor de 5 floats).
static PredictRequest make_predict_request(
    const std::string& model_id,
    const std::vector<double>& vals = {1.0, 2.0, 3.0, 4.0, 5.0})
{
    Array arr;
    for (double v : vals) arr.push_back(Value{v});
    Object obj;
    obj["input"] = Value{std::move(arr)};

    PredictRequest req;
    req.set_model_id(model_id);
    *req.mutable_inputs() = to_proto_struct(obj);
    return req;
}

/// Constrói PredictRequest com chave de tensor errada (falha esperada).
static PredictRequest make_wrong_key_request(const std::string& model_id) {
    Array arr;
    for (int i = 0; i < 5; ++i) arr.push_back(Value{static_cast<double>(i)});
    Object obj;
    obj["chave_errada"] = Value{std::move(arr)};

    PredictRequest req;
    req.set_model_id(model_id);
    *req.mutable_inputs() = to_proto_struct(obj);
    return req;
}

/// Case-insensitive substring search.
static bool contains_ci(const std::string& haystack, const std::string& needle) {
    auto it = std::search(
        haystack.begin(), haystack.end(),
        needle.begin(),   needle.end(),
        [](char a, char b){ return std::tolower(a) == std::tolower(b); });
    return it != haystack.end();
}

// =============================================================================
// Fixture base
// =============================================================================

class WorkerServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        svc = std::make_unique<WorkerServiceImpl>(
            "test-worker",
            /*enable_gpu=*/false,
            /*num_threads=*/2,
            models_dir());
    }

    void TearDown() override { svc.reset(); }

    bool load(const std::string& id, const std::string& path) {
        LoadModelRequest  req;
        LoadModelResponse resp;
        req.set_model_id(id);
        req.set_model_path(path);
        return svc->LoadModel(nullptr, &req, &resp).ok() && resp.success();
    }

    bool unload(const std::string& id) {
        UnloadModelRequest  req;
        UnloadModelResponse resp;
        req.set_model_id(id);
        return svc->UnloadModel(nullptr, &req, &resp).ok() && resp.success();
    }

    PredictResponse predict(const std::string& id,
                            const std::vector<double>& vals = {1,2,3,4,5}) {
        auto req = make_predict_request(id, vals);
        PredictResponse resp;
        svc->Predict(nullptr, &req, &resp);
        return resp;
    }

    std::unique_ptr<WorkerServiceImpl> svc;
};

/// Fixture que pula se o modelo Python não existir e pré-carrega "linear".
class WorkerServicePythonTest : public WorkerServiceTest {
protected:
    void SetUp() override {
        WorkerServiceTest::SetUp();
        if (!fs::exists(linear_path()))
            GTEST_SKIP() << "Modelo Python não encontrado: " << linear_path();
        ASSERT_TRUE(load("linear", linear_path())) << "load() falhou no SetUp()";
    }

    void TearDown() override {
        // Descarrega tudo antes de destruir
        ListModelsRequest  lreq;
        ListModelsResponse lresp;
        svc->ListModels(nullptr, &lreq, &lresp);
        for (const auto& m : lresp.models()) {
            UnloadModelRequest req;
            req.set_model_id(m.model_id());
            UnloadModelResponse resp;
            svc->UnloadModel(nullptr, &req, &resp);
        }
        WorkerServiceTest::TearDown();
    }
};

// =============================================================================
// GRUPO 1 — Construção e destruição
// =============================================================================

TEST(WorkerServiceConstruct, CriaComValoresPadrao) {
    EXPECT_NO_THROW({ WorkerServiceImpl svc("worker-padrao"); });
}

TEST(WorkerServiceConstruct, CriaComWorkerIdCustomizado) {
    WorkerServiceImpl svc("meu-worker-42", false, 4, "./models");
    GetStatusRequest  req;
    GetStatusResponse resp;
    svc.GetStatus(nullptr, &req, &resp);
    EXPECT_EQ(resp.worker_id(), "meu-worker-42");
}

TEST(WorkerServiceConstruct, DestrucaoSemCrash) {
    EXPECT_NO_THROW({
        auto s = std::make_unique<WorkerServiceImpl>("test-destroy");
        s.reset();
    });
}

TEST(WorkerServiceConstruct, DestrucaoComModeloCarregado) {
    auto path = fs::weakly_canonical("./models/simple_linear.py");
    if (!fs::exists(path)) GTEST_SKIP() << "Modelo não encontrado";

    EXPECT_NO_THROW({
        WorkerServiceImpl svc("test-destroy-loaded");
        LoadModelRequest  req;
        LoadModelResponse resp;
        req.set_model_id("m1");
        req.set_model_path(path.string());
        svc.LoadModel(nullptr, &req, &resp);
        // destrutor deve descarregar sem crash
    });
}

TEST(WorkerServiceConstruct, StatusRetornaImediatamente) {
    WorkerServiceImpl svc("test-imediato");
    GetStatusRequest  req;
    GetStatusResponse resp;
    auto st = svc.GetStatus(nullptr, &req, &resp);
    EXPECT_TRUE(st.ok());
    EXPECT_EQ(resp.worker_id(), "test-imediato");
    EXPECT_EQ(resp.loaded_model_ids_size(), 0);
}

// =============================================================================
// GRUPO 2 — LoadModel
// =============================================================================

TEST_F(WorkerServiceTest, LoadModel_PathValido_Sucesso) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    EXPECT_TRUE(load("m1", linear_path()));
}

TEST_F(WorkerServiceTest, LoadModel_RetornaStatusOk) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    LoadModelRequest  req;
    LoadModelResponse resp;
    req.set_model_id("m1");
    req.set_model_path(linear_path());
    EXPECT_TRUE(svc->LoadModel(nullptr, &req, &resp).ok());
}

TEST_F(WorkerServiceTest, LoadModel_PathInexistente_Falha) {
    EXPECT_FALSE(load("m_bad", "/nao/existe/modelo.py"));
}

TEST_F(WorkerServiceTest, LoadModel_PathInexistente_ErrorMessagePreenchido) {
    LoadModelRequest  req;
    LoadModelResponse resp;
    req.set_model_id("m_bad");
    req.set_model_path("/nao/existe/modelo.py");
    svc->LoadModel(nullptr, &req, &resp);
    EXPECT_FALSE(resp.error_message().empty());
}

TEST_F(WorkerServiceTest, LoadModel_IDDuplicado_Falha) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    ASSERT_TRUE(load("dup", linear_path()));
    EXPECT_FALSE(load("dup", linear_path()));
}

TEST_F(WorkerServiceTest, LoadModel_IDsDistintosArquivoIgual_DuasInstancias) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    EXPECT_TRUE(load("inst_a", linear_path()));
    EXPECT_TRUE(load("inst_b", linear_path()));
    ListModelsRequest  lreq;
    ListModelsResponse lresp;
    svc->ListModels(nullptr, &lreq, &lresp);
    EXPECT_GE(lresp.models_size(), 2);
}

TEST_F(WorkerServiceTest, LoadModel_ExtensaoDesconhecida_Falha) {
    auto tmp = fs::temp_directory_path() / "modelo_test.xyz";
    { std::ofstream f(tmp); f << "dummy"; }
    EXPECT_FALSE(load("m_xyz", tmp.string()));
    fs::remove(tmp);
}

TEST_F(WorkerServiceTest, LoadModel_IDVazio_Aceito) {
    // O engine não valida ID vazio — aceita e carrega normalmente.
    // Teste documenta o comportamento real (sem rejeição de ID vazio).
    if (!fs::exists(linear_path())) GTEST_SKIP();
    EXPECT_TRUE(load("", linear_path()));
}

// =============================================================================
// GRUPO 3 — UnloadModel
// =============================================================================

TEST_F(WorkerServicePythonTest, UnloadModel_HappyPath_Sucesso) {
    EXPECT_TRUE(unload("linear"));
}

TEST_F(WorkerServicePythonTest, UnloadModel_RetornaStatusOk) {
    UnloadModelRequest  req;
    UnloadModelResponse resp;
    req.set_model_id("linear");
    EXPECT_TRUE(svc->UnloadModel(nullptr, &req, &resp).ok());
}

TEST_F(WorkerServiceTest, UnloadModel_IDInexistente_Falha) {
    EXPECT_FALSE(unload("nao_existe"));
}

TEST_F(WorkerServiceTest, UnloadModel_IDInexistente_MessagePreenchido) {
    // campo correto: message (não error_message) — UnloadModelResponse.message
    UnloadModelRequest  req;
    UnloadModelResponse resp;
    req.set_model_id("nao_existe");
    svc->UnloadModel(nullptr, &req, &resp);
    EXPECT_FALSE(resp.message().empty());
}

TEST_F(WorkerServicePythonTest, UnloadModel_PosUnload_ModeloSumeDaLista) {
    unload("linear");
    ListModelsRequest  lreq;
    ListModelsResponse lresp;
    svc->ListModels(nullptr, &lreq, &lresp);
    for (const auto& m : lresp.models())
        EXPECT_NE(m.model_id(), "linear");
}

TEST_F(WorkerServicePythonTest, UnloadModel_DuplaDescarga_SegundaFalha) {
    EXPECT_TRUE(unload("linear"));
    EXPECT_FALSE(unload("linear"));
}

// =============================================================================
// GRUPO 4 — Predict
// =============================================================================

TEST_F(WorkerServicePythonTest, Predict_InputsValidos_Sucesso) {
    EXPECT_TRUE(predict("linear").success());
}

TEST_F(WorkerServicePythonTest, Predict_RetornaStatusOk) {
    auto req = make_predict_request("linear");
    PredictResponse resp;
    EXPECT_TRUE(svc->Predict(nullptr, &req, &resp).ok());
}

TEST_F(WorkerServicePythonTest, Predict_OutputNaoVazio) {
    auto resp = predict("linear");
    ASSERT_TRUE(resp.success());
    EXPECT_GT(resp.outputs().fields_size(), 0);
}

TEST_F(WorkerServicePythonTest, Predict_InferenceTimeMsPositivo) {
    auto resp = predict("linear");
    ASSERT_TRUE(resp.success());
    EXPECT_GT(resp.inference_time_ms(), 0.0);
}

TEST_F(WorkerServicePythonTest, Predict_ResultadoDeterministico) {
    auto r1 = predict("linear", {1,2,3,4,5});
    auto r2 = predict("linear", {1,2,3,4,5});
    ASSERT_TRUE(r1.success());
    ASSERT_TRUE(r2.success());
    const auto& f1 = r1.outputs().fields();
    const auto& f2 = r2.outputs().fields();
    ASSERT_EQ(f1.size(), f2.size());
    for (auto it = f1.begin(); it != f1.end(); ++it) {
        auto jt = f2.find(it->first);
        ASSERT_NE(jt, f2.end());
        EXPECT_DOUBLE_EQ(it->second.number_value(), jt->second.number_value());
    }
}

TEST_F(WorkerServicePythonTest, Predict_ChaveErrada_Falha) {
    auto req = make_wrong_key_request("linear");
    PredictResponse resp;
    svc->Predict(nullptr, &req, &resp);
    EXPECT_FALSE(resp.success());
}

TEST_F(WorkerServicePythonTest, Predict_ChaveErrada_ErrorMessagePreenchido) {
    auto req = make_wrong_key_request("linear");
    PredictResponse resp;
    svc->Predict(nullptr, &req, &resp);
    EXPECT_FALSE(resp.error_message().empty());
}

TEST_F(WorkerServiceTest, Predict_IDInvalido_Falha) {
    auto req = make_predict_request("modelo_que_nao_existe");
    PredictResponse resp;
    svc->Predict(nullptr, &req, &resp);
    EXPECT_FALSE(resp.success());
}

TEST_F(WorkerServiceTest, Predict_IDInvalido_RetornaStatusOk) {
    // Erros de inferência ficam no payload, não no Status gRPC
    auto req = make_predict_request("modelo_que_nao_existe");
    PredictResponse resp;
    EXPECT_TRUE(svc->Predict(nullptr, &req, &resp).ok());
    EXPECT_FALSE(resp.success());
}

TEST_F(WorkerServicePythonTest, Predict_InputsVazios_Falha) {
    PredictRequest  req;
    PredictResponse resp;
    req.set_model_id("linear");
    svc->Predict(nullptr, &req, &resp);
    EXPECT_FALSE(resp.success());
}

TEST_F(WorkerServicePythonTest, Predict_10Chamadas_TodasSucesso) {
    for (int i = 0; i < 10; ++i)
        EXPECT_TRUE(predict("linear").success()) << "falhou na chamada " << i;
}

// =============================================================================
// GRUPO 5 — BatchPredict
// =============================================================================

TEST_F(WorkerServicePythonTest, BatchPredict_Vazio_Sucesso) {
    BatchPredictRequest  req;
    BatchPredictResponse resp;
    EXPECT_TRUE(svc->BatchPredict(nullptr, &req, &resp).ok());
    EXPECT_TRUE(resp.success());
    EXPECT_EQ(resp.responses_size(), 0);
}

TEST_F(WorkerServicePythonTest, BatchPredict_UmElemento_Sucesso) {
    BatchPredictRequest req;
    *req.add_requests() = make_predict_request("linear");
    BatchPredictResponse resp;
    svc->BatchPredict(nullptr, &req, &resp);
    ASSERT_EQ(resp.responses_size(), 1);
    EXPECT_TRUE(resp.responses(0).success());
}

TEST_F(WorkerServicePythonTest, BatchPredict_CincoElementos_TodosSucesso) {
    BatchPredictRequest req;
    for (int i = 0; i < 5; ++i)
        *req.add_requests() = make_predict_request("linear",
            {(double)i, (double)(i+1), (double)(i+2),
             (double)(i+3), (double)(i+4)});
    BatchPredictResponse resp;
    svc->BatchPredict(nullptr, &req, &resp);
    ASSERT_EQ(resp.responses_size(), 5);
    for (int i = 0; i < 5; ++i)
        EXPECT_TRUE(resp.responses(i).success()) << "elem " << i;
}

TEST_F(WorkerServicePythonTest, BatchPredict_OrdemPreservada) {
    BatchPredictRequest req;
    *req.add_requests() = make_predict_request("linear", {1,1,1,1,1});
    *req.add_requests() = make_predict_request("linear", {9,9,9,9,9});
    BatchPredictResponse resp;
    svc->BatchPredict(nullptr, &req, &resp);
    ASSERT_EQ(resp.responses_size(), 2);
    double out0 = resp.responses(0).outputs().fields().begin()->second.number_value();
    double out1 = resp.responses(1).outputs().fields().begin()->second.number_value();
    EXPECT_NE(out0, out1) << "Inputs distintos devem gerar outputs distintos";
}

TEST_F(WorkerServicePythonTest, BatchPredict_UmFalhando_SucessoFalse) {
    BatchPredictRequest req;
    *req.add_requests() = make_predict_request("linear");
    *req.add_requests() = make_wrong_key_request("linear");
    BatchPredictResponse resp;
    svc->BatchPredict(nullptr, &req, &resp);
    ASSERT_EQ(resp.responses_size(), 2);
    EXPECT_TRUE(resp.responses(0).success());
    EXPECT_FALSE(resp.responses(1).success());
    EXPECT_FALSE(resp.success());
}

TEST_F(WorkerServicePythonTest, BatchPredict_TotalTimeMsPositivo) {
    BatchPredictRequest req;
    *req.add_requests() = make_predict_request("linear");
    BatchPredictResponse resp;
    svc->BatchPredict(nullptr, &req, &resp);
    EXPECT_GT(resp.total_time_ms(), 0.0);
}

// =============================================================================
// GRUPO 6 — ListModels
// =============================================================================

TEST_F(WorkerServiceTest, ListModels_SemModelos_RetornaVazio) {
    ListModelsRequest  req;
    ListModelsResponse resp;
    EXPECT_TRUE(svc->ListModels(nullptr, &req, &resp).ok());
    EXPECT_EQ(resp.models_size(), 0);
}

TEST_F(WorkerServicePythonTest, ListModels_AposLoad_ContemLinear) {
    ListModelsRequest  req;
    ListModelsResponse resp;
    svc->ListModels(nullptr, &req, &resp);
    ASSERT_GE(resp.models_size(), 1);
    bool found = false;
    for (const auto& m : resp.models())
        if (m.model_id() == "linear") { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(WorkerServicePythonTest, ListModels_AposUnload_NaoContemLinear) {
    unload("linear");
    ListModelsRequest  req;
    ListModelsResponse resp;
    svc->ListModels(nullptr, &req, &resp);
    for (const auto& m : resp.models())
        EXPECT_NE(m.model_id(), "linear");
}

TEST_F(WorkerServicePythonTest, ListModels_CamposObrigatoriosPreenchidos) {
    ListModelsRequest  req;
    ListModelsResponse resp;
    svc->ListModels(nullptr, &req, &resp);
    ASSERT_GE(resp.models_size(), 1);
    EXPECT_FALSE(resp.models(0).model_id().empty());
    EXPECT_FALSE(resp.models(0).model_path().empty());
}

TEST_F(WorkerServicePythonTest, ListModels_DoisModelos_ContagemCorreta) {
    load("extra", linear_path());
    ListModelsRequest  req;
    ListModelsResponse resp;
    svc->ListModels(nullptr, &req, &resp);
    EXPECT_GE(resp.models_size(), 2);
}

// =============================================================================
// GRUPO 7 — GetModelInfo
// =============================================================================

TEST_F(WorkerServicePythonTest, GetModelInfo_ModeloCarregado_Sucesso) {
    GetModelInfoRequest  req;
    GetModelInfoResponse resp;
    req.set_model_id("linear");
    EXPECT_TRUE(svc->GetModelInfo(nullptr, &req, &resp).ok());
    EXPECT_TRUE(resp.success());
}

TEST_F(WorkerServicePythonTest, GetModelInfo_ModelIDCorreto) {
    GetModelInfoRequest  req;
    GetModelInfoResponse resp;
    req.set_model_id("linear");
    svc->GetModelInfo(nullptr, &req, &resp);
    EXPECT_EQ(resp.model_info().model_id(), "linear");
}

TEST_F(WorkerServicePythonTest, GetModelInfo_BackendPython) {
    GetModelInfoRequest  req;
    GetModelInfoResponse resp;
    req.set_model_id("linear");
    svc->GetModelInfo(nullptr, &req, &resp);
    // common::BackendType enum — BACKEND_PYTHON = 2 (verificar common.proto)
    EXPECT_EQ(resp.model_info().backend(), common::BACKEND_PYTHON);
}

TEST_F(WorkerServicePythonTest, GetModelInfo_InputsNaoVazios) {
    GetModelInfoRequest  req;
    GetModelInfoResponse resp;
    req.set_model_id("linear");
    svc->GetModelInfo(nullptr, &req, &resp);
    EXPECT_GT(resp.model_info().inputs_size(), 0);
}

TEST_F(WorkerServicePythonTest, GetModelInfo_OutputsNaoVazios) {
    GetModelInfoRequest  req;
    GetModelInfoResponse resp;
    req.set_model_id("linear");
    svc->GetModelInfo(nullptr, &req, &resp);
    EXPECT_GT(resp.model_info().outputs_size(), 0);
}

TEST_F(WorkerServicePythonTest, GetModelInfo_PathPreenchido) {
    GetModelInfoRequest  req;
    GetModelInfoResponse resp;
    req.set_model_id("linear");
    svc->GetModelInfo(nullptr, &req, &resp);
    EXPECT_FALSE(resp.model_info().model_path().empty());
}

TEST_F(WorkerServiceTest, GetModelInfo_ModeloAusente_SuccessFalse) {
    GetModelInfoRequest  req;
    GetModelInfoResponse resp;
    req.set_model_id("nao_carregado");
    EXPECT_TRUE(svc->GetModelInfo(nullptr, &req, &resp).ok());
    EXPECT_FALSE(resp.success());
}

TEST_F(WorkerServiceTest, GetModelInfo_ModeloAusente_ErrorMessagePreenchido) {
    GetModelInfoRequest  req;
    GetModelInfoResponse resp;
    req.set_model_id("nao_carregado");
    svc->GetModelInfo(nullptr, &req, &resp);
    EXPECT_FALSE(resp.error_message().empty());
}

// =============================================================================
// GRUPO 8 — ValidateModel
// =============================================================================

TEST_F(WorkerServiceTest, ValidateModel_RetornaStatusOk) {
    ValidateModelRequest  req;
    ValidateModelResponse resp;
    req.set_model_path("/qualquer/coisa.py");
    EXPECT_TRUE(svc->ValidateModel(nullptr, &req, &resp).ok());
}

TEST_F(WorkerServicePythonTest, ValidateModel_ArquivoValido_Valid) {
    ValidateModelRequest  req;
    ValidateModelResponse resp;
    req.set_model_path(linear_path());
    svc->ValidateModel(nullptr, &req, &resp);
    EXPECT_TRUE(resp.valid());
}

TEST_F(WorkerServiceTest, ValidateModel_ArquivoInexistente_Invalid) {
    ValidateModelRequest  req;
    ValidateModelResponse resp;
    req.set_model_path("/nao/existe/modelo.py");
    svc->ValidateModel(nullptr, &req, &resp);
    EXPECT_FALSE(resp.valid());
}

TEST_F(WorkerServiceTest, ValidateModel_ArquivoInexistente_ErrorMessagePreenchido) {
    ValidateModelRequest  req;
    ValidateModelResponse resp;
    req.set_model_path("/nao/existe/modelo.py");
    svc->ValidateModel(nullptr, &req, &resp);
    EXPECT_FALSE(resp.error_message().empty());
}

TEST_F(WorkerServiceTest, ValidateModel_ExtensaoDesconhecida_Invalid) {
    auto tmp = fs::temp_directory_path() / "modelo_test_ws.xyz";
    { std::ofstream f(tmp); f << "dummy"; }
    ValidateModelRequest  req;
    ValidateModelResponse resp;
    req.set_model_path(tmp.string());
    svc->ValidateModel(nullptr, &req, &resp);
    EXPECT_FALSE(resp.valid());
    fs::remove(tmp);
}

TEST_F(WorkerServicePythonTest, ValidateModel_BackendPreenchido) {
    ValidateModelRequest  req;
    ValidateModelResponse resp;
    req.set_model_path(linear_path());
    svc->ValidateModel(nullptr, &req, &resp);
    EXPECT_NE(resp.backend(), common::BACKEND_UNKNOWN);
}

TEST_F(WorkerServicePythonTest, ValidateModel_InputsPreenchidos) {
    ValidateModelRequest  req;
    ValidateModelResponse resp;
    req.set_model_path(linear_path());
    svc->ValidateModel(nullptr, &req, &resp);
    EXPECT_GT(resp.inputs_size(), 0);
}

TEST_F(WorkerServicePythonTest, ValidateModel_OutputsPreenchidos) {
    ValidateModelRequest  req;
    ValidateModelResponse resp;
    req.set_model_path(linear_path());
    svc->ValidateModel(nullptr, &req, &resp);
    EXPECT_GT(resp.outputs_size(), 0);
}

TEST_F(WorkerServicePythonTest, ValidateModel_NomeInputNaoVazio) {
    ValidateModelRequest  req;
    ValidateModelResponse resp;
    req.set_model_path(linear_path());
    svc->ValidateModel(nullptr, &req, &resp);
    ASSERT_GT(resp.inputs_size(), 0);
    EXPECT_FALSE(resp.inputs(0).name().empty());
}

// =============================================================================
// GRUPO 9 — WarmupModel
// =============================================================================

TEST_F(WorkerServicePythonTest, WarmupModel_HappyPath_Sucesso) {
    WarmupModelRequest  req;
    WarmupModelResponse resp;
    req.set_model_id("linear");
    req.set_num_runs(3);
    EXPECT_TRUE(svc->WarmupModel(nullptr, &req, &resp).ok());
    EXPECT_TRUE(resp.success());
}

TEST_F(WorkerServicePythonTest, WarmupModel_RunsCompletados_IgualAoSolicitado) {
    WarmupModelRequest  req;
    WarmupModelResponse resp;
    req.set_model_id("linear");
    req.set_num_runs(5);
    svc->WarmupModel(nullptr, &req, &resp);
    EXPECT_EQ(resp.runs_completed(), 5u);
}

TEST_F(WorkerServicePythonTest, WarmupModel_AvgTimeMsPositivo) {
    WarmupModelRequest  req;
    WarmupModelResponse resp;
    req.set_model_id("linear");
    req.set_num_runs(3);
    svc->WarmupModel(nullptr, &req, &resp);
    EXPECT_GT(resp.avg_time_ms(), 0.0);
}

TEST_F(WorkerServicePythonTest, WarmupModel_MinMenorIgualMax) {
    WarmupModelRequest  req;
    WarmupModelResponse resp;
    req.set_model_id("linear");
    req.set_num_runs(5);
    svc->WarmupModel(nullptr, &req, &resp);
    EXPECT_LE(resp.min_time_ms(), resp.max_time_ms());
}

TEST_F(WorkerServicePythonTest, WarmupModel_NumRunsZero_UsaDefault) {
    WarmupModelRequest  req;
    WarmupModelResponse resp;
    req.set_model_id("linear");
    req.set_num_runs(0);
    svc->WarmupModel(nullptr, &req, &resp);
    EXPECT_TRUE(resp.success());
    EXPECT_GT(resp.runs_completed(), 0u);
}

TEST_F(WorkerServiceTest, WarmupModel_IDInexistente_Falha) {
    WarmupModelRequest  req;
    WarmupModelResponse resp;
    req.set_model_id("nao_existe");
    req.set_num_runs(3);
    EXPECT_TRUE(svc->WarmupModel(nullptr, &req, &resp).ok());
    EXPECT_FALSE(resp.success());
}

TEST_F(WorkerServiceTest, WarmupModel_IDInexistente_ErrorMessagePreenchido) {
    WarmupModelRequest  req;
    WarmupModelResponse resp;
    req.set_model_id("nao_existe");
    req.set_num_runs(3);
    svc->WarmupModel(nullptr, &req, &resp);
    EXPECT_FALSE(resp.error_message().empty());
}

// =============================================================================
// GRUPO 10 — HealthCheck
// =============================================================================

TEST_F(WorkerServiceTest, HealthCheck_RetornaSaudavel) {
    HealthCheckRequest  req;
    HealthCheckResponse resp;
    EXPECT_TRUE(svc->HealthCheck(nullptr, &req, &resp).ok());
    EXPECT_TRUE(resp.healthy());
}

TEST_F(WorkerServiceTest, HealthCheck_EstavelEm20Chamadas) {
    for (int i = 0; i < 20; ++i) {
        HealthCheckRequest  req;
        HealthCheckResponse resp;
        svc->HealthCheck(nullptr, &req, &resp);
        EXPECT_TRUE(resp.healthy()) << "falhou na chamada " << i;
    }
}

TEST_F(WorkerServiceTest, HealthCheck_AposLoadUnload_AindaSaudavel) {
    if (!fs::exists(linear_path())) GTEST_SKIP();
    load("hc_model", linear_path());
    unload("hc_model");
    HealthCheckRequest  req;
    HealthCheckResponse resp;
    svc->HealthCheck(nullptr, &req, &resp);
    EXPECT_TRUE(resp.healthy());
}

// =============================================================================
// GRUPO 11 — GetStatus
// GetStatusResponse.metrics → common::WorkerMetrics
// GetStatusResponse.capabilities → common::WorkerCapabilities
// =============================================================================

TEST_F(WorkerServiceTest, GetStatus_WorkerIdPreenchido) {
    GetStatusRequest  req;
    GetStatusResponse resp;
    svc->GetStatus(nullptr, &req, &resp);
    EXPECT_EQ(resp.worker_id(), "test-worker");
}

TEST_F(WorkerServiceTest, GetStatus_UptimeNaoNegativo) {
    GetStatusRequest  req;
    GetStatusResponse resp;
    svc->GetStatus(nullptr, &req, &resp);
    // metrics().uptime_seconds() — campo de common::WorkerMetrics
    EXPECT_GE(resp.metrics().uptime_seconds(), 0LL);
}

TEST_F(WorkerServiceTest, GetStatus_SemModelos_ListaVazia) {
    GetStatusRequest  req;
    GetStatusResponse resp;
    svc->GetStatus(nullptr, &req, &resp);
    EXPECT_EQ(resp.loaded_model_ids_size(), 0);
}

TEST_F(WorkerServicePythonTest, GetStatus_AposLoad_ContemLinear) {
    GetStatusRequest  req;
    GetStatusResponse resp;
    svc->GetStatus(nullptr, &req, &resp);
    bool found = false;
    for (const auto& id : resp.loaded_model_ids())
        if (id == "linear") { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(WorkerServicePythonTest, GetStatus_AposUnload_NaoContemLinear) {
    unload("linear");
    GetStatusRequest  req;
    GetStatusResponse resp;
    svc->GetStatus(nullptr, &req, &resp);
    for (const auto& id : resp.loaded_model_ids())
        EXPECT_NE(id, "linear");
}

TEST_F(WorkerServiceTest, GetStatus_UptimeCresceCom1s) {
    GetStatusRequest  req;
    GetStatusResponse resp1, resp2;
    svc->GetStatus(nullptr, &req, &resp1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    svc->GetStatus(nullptr, &req, &resp2);
    EXPECT_GT(resp2.metrics().uptime_seconds(),
              resp1.metrics().uptime_seconds());
}

TEST_F(WorkerServiceTest, GetStatus_SupportedBackendsNaoVazio) {
    GetStatusRequest  req;
    GetStatusResponse resp;
    svc->GetStatus(nullptr, &req, &resp);
    // capabilities().supported_backends() — campo de common::WorkerCapabilities
    EXPECT_GT(resp.capabilities().supported_backends_size(), 0);
}

TEST_F(WorkerServiceTest, GetStatus_SupportedBackendsContemPython) {
    GetStatusRequest  req;
    GetStatusResponse resp;
    svc->GetStatus(nullptr, &req, &resp);
    bool found = false;
    for (const auto& b : resp.capabilities().supported_backends())
        if (contains_ci(b, "python")) { found = true; break; }
    EXPECT_TRUE(found);
}

// =============================================================================
// GRUPO 12 — GetMetrics
// GetMetricsResponse.worker_metrics → common::WorkerMetrics
// GetMetricsResponse.per_model_metrics → map<string, ModelRuntimeMetrics>
// ModelRuntimeMetrics.avg_inference_time_ms (não avg_time_ms)
// =============================================================================

TEST_F(WorkerServiceTest, GetMetrics_RetornaStatusOk) {
    GetMetricsRequest  req;
    GetMetricsResponse resp;
    EXPECT_TRUE(svc->GetMetrics(nullptr, &req, &resp).ok());
}

TEST_F(WorkerServiceTest, GetMetrics_NaoTrava) {
    EXPECT_NO_THROW({
        GetMetricsRequest  req;
        GetMetricsResponse resp;
        svc->GetMetrics(nullptr, &req, &resp);
    });
}

TEST_F(WorkerServiceTest, GetMetrics_TotalRequestsInicialZero) {
    GetMetricsRequest  req;
    GetMetricsResponse resp;
    svc->GetMetrics(nullptr, &req, &resp);
    // campo correto: worker_metrics (não metrics)
    EXPECT_EQ(resp.worker_metrics().total_requests(), 0u);
}

TEST_F(WorkerServicePythonTest, GetMetrics_TotalRequestsCresce) {
    predict("linear");
    predict("linear");
    GetMetricsRequest  req;
    GetMetricsResponse resp;
    svc->GetMetrics(nullptr, &req, &resp);
    EXPECT_GE(resp.worker_metrics().total_requests(), 2u);
}

TEST_F(WorkerServicePythonTest, GetMetrics_SuccessfulNaoSuperapTotal) {
    predict("linear");
    GetMetricsRequest  req;
    GetMetricsResponse resp;
    svc->GetMetrics(nullptr, &req, &resp);
    EXPECT_LE(resp.worker_metrics().successful_requests(),
              resp.worker_metrics().total_requests());
}

TEST_F(WorkerServicePythonTest, GetMetrics_PerModelContemLinear) {
    predict("linear");
    GetMetricsRequest  req;
    GetMetricsResponse resp;
    svc->GetMetrics(nullptr, &req, &resp);
    EXPECT_GT(resp.per_model_metrics().count("linear"), 0u);
}

TEST_F(WorkerServicePythonTest, GetMetrics_PerModelAvgTimeMsPositivo) {
    predict("linear");
    GetMetricsRequest  req;
    GetMetricsResponse resp;
    svc->GetMetrics(nullptr, &req, &resp);
    auto it = resp.per_model_metrics().find("linear");
    if (it != resp.per_model_metrics().end()) {
        // campo correto: avg_inference_time_ms (não avg_time_ms)
        EXPECT_GT(it->second.avg_inference_time_ms(), 0.0);
    }
}

TEST_F(WorkerServicePythonTest, GetMetrics_FalhaIncrementa_FailedRequests) {
    auto req_bad = make_wrong_key_request("linear");
    PredictResponse presp;
    svc->Predict(nullptr, &req_bad, &presp);
    GetMetricsRequest  req;
    GetMetricsResponse resp;
    svc->GetMetrics(nullptr, &req, &resp);
    EXPECT_GE(resp.worker_metrics().failed_requests(), 1u);
}

// =============================================================================
// GRUPO 13 — ListAvailableModels
// =============================================================================

TEST_F(WorkerServiceTest, ListAvailableModels_DirExistente_RetornaStatusOk) {
    ListAvailableModelsRequest  req;
    ListAvailableModelsResponse resp;
    req.set_directory(models_dir());
    EXPECT_TRUE(svc->ListAvailableModels(nullptr, &req, &resp).ok());
}

TEST_F(WorkerServiceTest, ListAvailableModels_DirInexistente_NotFound) {
    ListAvailableModelsRequest  req;
    ListAvailableModelsResponse resp;
    req.set_directory("/dir/que/nao/existe");
    auto st = svc->ListAvailableModels(nullptr, &req, &resp);
    EXPECT_EQ(st.error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(WorkerServiceTest, ListAvailableModels_CamposObrigatorios) {
    ListAvailableModelsRequest  req;
    ListAvailableModelsResponse resp;
    req.set_directory(models_dir());
    svc->ListAvailableModels(nullptr, &req, &resp);
    for (const auto& m : resp.models()) {
        EXPECT_FALSE(m.filename().empty())  << "filename vazio";
        EXPECT_FALSE(m.path().empty())      << "path vazio";
        EXPECT_FALSE(m.extension().empty()) << "extension vazia";
        EXPECT_GT(m.file_size_bytes(), 0)   << "file_size <= 0";
    }
}

TEST_F(WorkerServiceTest, ListAvailableModels_ExtensoesSuportadas) {
    ListAvailableModelsRequest  req;
    ListAvailableModelsResponse resp;
    req.set_directory(models_dir());
    svc->ListAvailableModels(nullptr, &req, &resp);
    for (const auto& m : resp.models()) {
        const auto& ext = m.extension();
        EXPECT_TRUE(ext == ".py" || ext == ".onnx")
            << "extensão inesperada: " << ext;
    }
}

TEST_F(WorkerServicePythonTest, ListAvailableModels_IsLoaded_RefleteCarga) {
    ListAvailableModelsRequest  req;
    ListAvailableModelsResponse resp;
    req.set_directory(models_dir());
    svc->ListAvailableModels(nullptr, &req, &resp);
    bool achou_carregado = false;
    for (const auto& m : resp.models())
        if (m.is_loaded()) { achou_carregado = true; break; }
    EXPECT_TRUE(achou_carregado);
}

TEST_F(WorkerServicePythonTest, ListAvailableModels_LoadedAs_Correto) {
    ListAvailableModelsRequest  req;
    ListAvailableModelsResponse resp;
    req.set_directory(models_dir());
    svc->ListAvailableModels(nullptr, &req, &resp);
    for (const auto& m : resp.models()) {
        if (m.is_loaded()) {
            EXPECT_FALSE(m.loaded_as().empty())
                << "loaded_as vazio para modelo carregado";
        }
    }
}

TEST_F(WorkerServicePythonTest, ListAvailableModels_BackendCorreto) {
    ListAvailableModelsRequest  req;
    ListAvailableModelsResponse resp;
    req.set_directory(models_dir());
    svc->ListAvailableModels(nullptr, &req, &resp);
    for (const auto& m : resp.models()) {
        if (m.extension() == ".py") {
            EXPECT_EQ(m.backend(), common::BACKEND_PYTHON);
        } else if (m.extension() == ".onnx") {
            EXPECT_EQ(m.backend(), common::BACKEND_ONNX);
        }
    }
}

TEST_F(WorkerServiceTest, ListAvailableModels_DirVazio_UsaDefault) {
    ListAvailableModelsRequest  req;
    ListAvailableModelsResponse resp;
    // directory não setado — usa models_dir_ configurado no construtor
    EXPECT_TRUE(svc->ListAvailableModels(nullptr, &req, &resp).ok());
}

// =============================================================================
// GRUPO 14 — PredictStream (via mock de ServerReaderWriter)
//
// grpc::ServerReaderWriter herda de grpc::internal::ServerReaderWriterInterface
// via grpc::ServerReaderWriterInterface. O mock precisa herdar da classe
// concreta ServerReaderWriter usando um canal fake, o que é inviável sem
// infraestrutura gRPC. A alternativa correta é refatorar PredictStream para
// aceitar uma interface abstraída — por ora testamos o comportamento via
// BatchPredict (equivalente funcional) e documentamos a limitação.
// =============================================================================

TEST_F(WorkerServicePythonTest, PredictStream_EquivalenteViaBatch_UmItem) {
    // PredictStream processa items individualmente igual ao Predict unitário.
    // Validamos o comportamento equivalente via BatchPredict (mesmo caminho
    // de código interno) já que ServerReaderWriter não é mockável sem server.
    BatchPredictRequest  req;
    BatchPredictResponse resp;
    *req.add_requests() = make_predict_request("linear");
    svc->BatchPredict(nullptr, &req, &resp);
    ASSERT_EQ(resp.responses_size(), 1);
    EXPECT_TRUE(resp.responses(0).success());
}

TEST_F(WorkerServicePythonTest, PredictStream_EquivalenteViaBatch_TresItens) {
    BatchPredictRequest  req;
    BatchPredictResponse resp;
    for (int i = 0; i < 3; ++i)
        *req.add_requests() = make_predict_request("linear");
    svc->BatchPredict(nullptr, &req, &resp);
    ASSERT_EQ(resp.responses_size(), 3);
    for (const auto& r : resp.responses())
        EXPECT_TRUE(r.success());
}

// =============================================================================
// GRUPO 15 — Contadores atômicos
// =============================================================================

TEST_F(WorkerServicePythonTest, Contadores_AposPredict_TotalIncrementa) {
    GetMetricsRequest  req;
    GetMetricsResponse before, after;
    svc->GetMetrics(nullptr, &req, &before);
    predict("linear");
    svc->GetMetrics(nullptr, &req, &after);
    EXPECT_GT(after.worker_metrics().total_requests(),
              before.worker_metrics().total_requests());
}

TEST_F(WorkerServicePythonTest, Contadores_AposPredict_SuccessfulIncrementa) {
    GetMetricsRequest  req;
    GetMetricsResponse before, after;
    svc->GetMetrics(nullptr, &req, &before);
    predict("linear");
    svc->GetMetrics(nullptr, &req, &after);
    EXPECT_GT(after.worker_metrics().successful_requests(),
              before.worker_metrics().successful_requests());
}

TEST_F(WorkerServicePythonTest, Contadores_AposFalha_FailedIncrementa) {
    GetMetricsRequest  req;
    GetMetricsResponse before, after;
    svc->GetMetrics(nullptr, &req, &before);
    auto req_bad = make_wrong_key_request("linear");
    PredictResponse presp;
    svc->Predict(nullptr, &req_bad, &presp);
    svc->GetMetrics(nullptr, &req, &after);
    EXPECT_GT(after.worker_metrics().failed_requests(),
              before.worker_metrics().failed_requests());
}

TEST_F(WorkerServicePythonTest, Contadores_ActiveRequestsVoltaZero) {
    predict("linear");
    GetStatusRequest  req;
    GetStatusResponse resp;
    svc->GetStatus(nullptr, &req, &resp);
    EXPECT_EQ(resp.metrics().active_requests(), 0u);
}

// =============================================================================
// GRUPO 16 — Ciclo completo
// =============================================================================

TEST_F(WorkerServiceTest, CicloCompleto_LoadPredictWarmupUnload) {
    if (!fs::exists(linear_path())) GTEST_SKIP();

    // 1. Load
    ASSERT_TRUE(load("ciclo", linear_path()));

    // 2. Predict (5x)
    for (int i = 0; i < 5; ++i)
        EXPECT_TRUE(predict("ciclo", {1,2,3,4,5}).success()) << "i=" << i;

    // 3. Warmup
    {
        WarmupModelRequest  req;
        WarmupModelResponse resp;
        req.set_model_id("ciclo");
        req.set_num_runs(3);
        svc->WarmupModel(nullptr, &req, &resp);
        EXPECT_TRUE(resp.success());
    }

    // 4. GetModelInfo
    {
        GetModelInfoRequest  req;
        GetModelInfoResponse resp;
        req.set_model_id("ciclo");
        svc->GetModelInfo(nullptr, &req, &resp);
        EXPECT_TRUE(resp.success());
    }

    // 5. Unload
    EXPECT_TRUE(unload("ciclo"));

    // 6. Predict após unload deve falhar
    EXPECT_FALSE(predict("ciclo").success());
}

TEST_F(WorkerServiceTest, CicloCompleto_TresModelsSimultaneos) {
    if (!fs::exists(linear_path())) GTEST_SKIP();

    ASSERT_TRUE(load("m1", linear_path()));
    ASSERT_TRUE(load("m2", linear_path()));
    ASSERT_TRUE(load("m3", linear_path()));

    for (const auto& id : std::vector<std::string>{"m1","m2","m3"})
        EXPECT_TRUE(predict(id).success()) << "falhou para " << id;

    {
        ListModelsRequest  lreq;
        ListModelsResponse lresp;
        svc->ListModels(nullptr, &lreq, &lresp);
        EXPECT_GE(lresp.models_size(), 3);
    }

    EXPECT_TRUE(unload("m1"));
    EXPECT_TRUE(unload("m2"));
    EXPECT_TRUE(unload("m3"));

    {
        // Nova variável — proto não acumula entre chamadas separadas
        ListModelsRequest  lreq2;
        ListModelsResponse lresp2;
        svc->ListModels(nullptr, &lreq2, &lresp2);
        EXPECT_EQ(lresp2.models_size(), 0);
    }
}
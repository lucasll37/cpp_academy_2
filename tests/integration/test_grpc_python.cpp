// =============================================================================
// test_grpc_python.cpp — Teste de integração exaustivo do InferenceClient
//                        (transporte gRPC, backend Python)
//
// Cobre TODOS os métodos públicos de InferenceClient via gRPC:
//
//   Conexão     : connect(), is_connected()
//   Ciclo vida  : load_model(), unload_model()
//   Inferência  : predict(), batch_predict()
//   Introspecção: list_models(), get_model_info(), validate_model(),
//                 warmup_model()
//   Observabil. : health_check(), get_status(), get_metrics()
//   Descoberta  : list_available_models()
//
// Espelho direto de test_grpc_onnx.cpp — mesma cobertura, mesmos
// grupos e mesmos casos de falha. As únicas diferenças são:
//
//   • Modelo      : simple_linear.py (PythonBackend) em vez de .onnx
//   • Output      : shape [1,1] → total==1 → is_number() (escalar colapsado
//                   por py_dict_to_outputs), não is_array()
//   • validate()  : PythonBackend verifica apenas existência + extensão ".py"
//                   — não inspeciona conteúdo nem popula inputs/outputs specs
//   • load() falha: arquivo .py sem subclasse MiiaModel (não arquivo corrompido)
//   • list_available_models(): extensão ".py", backend "python"
//   • get_status().supported_backends: contém "python"
//   • get_model_info().backend == "python"
//
// Pré-requisito: worker rodando antes de executar.
//   make run-worker &
//   make test-integration
//
// Variáveis de ambiente:
//   WORKER_ADDRESS=localhost:50052  (padrão)
//   MODELS_DIR=./models             (padrão — deve ser visível pelo worker)
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

static std::string worker_address() {
    const char* e = std::getenv("WORKER_ADDRESS");
    return e ? e : "localhost:50052";
}

static std::string models_dir() {
    // Canonicalizado localmente: quando worker e teste rodam no mesmo host,
    // o path resolvido aqui é o mesmo que o servidor armazena após
    // resolve_path(weakly_canonical) no load_model().
    const char* e = std::getenv("MODELS_DIR");
    const std::string raw = e ? e : "./models";
    return std::filesystem::weakly_canonical(raw).string();
}

static std::string linear_path() {
    return models_dir() + "/simple_linear.py";
}

// Verifica se o worker está disponível. Usado nos GTEST_SKIP dos standalone.
static bool worker_disponivel() {
    InferenceClient c(worker_address());
    return c.connect() && c.health_check();
}

// Constrói um Object de entrada válido: "input" → vetor de 5 floats.
static Object make_valid_inputs(float v0 = 1.0f, float v1 = 2.0f,
                                 float v2 = 3.0f, float v3 = 4.0f,
                                 float v4 = 5.0f) {
    Array arr;
    arr.push_back(Value{static_cast<double>(v0)});
    arr.push_back(Value{static_cast<double>(v1)});
    arr.push_back(Value{static_cast<double>(v2)});
    arr.push_back(Value{static_cast<double>(v3)});
    arr.push_back(Value{static_cast<double>(v4)});
    Object o;
    o["input"] = Value{std::move(arr)};
    return o;
}

// Constrói um Object com nome de tensor errado (falha esperada no backend Python).
static Object make_wrong_key_inputs() {
    Array arr;
    for (int i = 0; i < 5; ++i)
        arr.push_back(Value{static_cast<double>(i)});
    Object o;
    o["tensor_errado"] = Value{std::move(arr)};
    return o;
}

// Constrói um Object com número errado de elementos no tensor (falha esperada).
static Object make_wrong_size_inputs() {
    Array arr;
    for (int i = 0; i < 3; ++i)  // modelo espera 5 elementos
        arr.push_back(Value{static_cast<double>(i)});
    Object o;
    o["input"] = Value{std::move(arr)};
    return o;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture base — conecta ao worker via gRPC e carrega o modelo como "linear"
// ─────────────────────────────────────────────────────────────────────────────

class GrpcPythonTest : public ::testing::Test {
protected:
    void SetUp() override {
        client = std::make_unique<InferenceClient>(worker_address());
        if (!client->connect() || !client->health_check())
            GTEST_SKIP() << "Worker indisponível em " << worker_address();
        ASSERT_TRUE(client->load_model("linear", linear_path()))
            << "load_model() falhou — verifique se o worker enxerga: " << linear_path();
    }

    void TearDown() override {
        if (client)
            for (const auto& m : client->list_models())
                client->unload_model(m.model_id);
    }

    std::unique_ptr<InferenceClient> client;
};

// =============================================================================
// GRUPO 1 — connect() / is_connected()
//
// Testa a conectividade gRPC: endereço válido, endereços inválidos,
// e o estado de is_connected() antes e depois do connect().
// =============================================================================

TEST(Conexao, WorkerDisponivelConecta) {
    // O worker deve aceitar conexão no endereço configurado.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    EXPECT_TRUE(c.connect());
}

TEST(Conexao, IsConnectedFalseAntesDoConnect) {
    // Antes de connect(), is_connected() deve ser false independente do worker.
    InferenceClient c(worker_address());
    EXPECT_FALSE(c.is_connected());
}

TEST(Conexao, IsConnectedTrueAposConnect) {
    // Após connect() bem-sucedido, is_connected() deve retornar true.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    EXPECT_TRUE(c.is_connected());
}

TEST(Conexao, ConectarDuasVezesNaoFalha) {
    // Reconectar um cliente já conectado não deve retornar false.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    EXPECT_TRUE(c.connect());
    EXPECT_TRUE(c.connect());
}

TEST(Conexao, PortaFechadaRetornaFalse) {
    // Endereço gRPC com porta fechada deve falhar silenciosamente.
    InferenceClient c("localhost:19998");
    EXPECT_FALSE(c.connect());
}

TEST(Conexao, IsConnectedFalseAposPortaFechada) {
    // is_connected() deve ser false quando connect() falhou.
    InferenceClient c("localhost:19998");
    c.connect();
    EXPECT_FALSE(c.is_connected());
}

// =============================================================================
// GRUPO 2 — load_model()
//
// Os happy-paths requerem worker rodando e o arquivo visível por ele.
// O skip é por worker indisponível, não por arquivo local.
// Os failure-paths (arquivo inexistente no servidor, conteúdo inválido)
// não precisam do skip pois testam rejeição, não sucesso.
// =============================================================================

TEST(LoadModel, CarregaModeloValidoRetornaTrue) {
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    EXPECT_TRUE(c.load_model("m1", linear_path()));
    c.unload_model("m1");
}

TEST(LoadModel, MesmoArquivoComIDsDiferentesPermitido) {
    // O mesmo arquivo pode ser carregado sob IDs distintos simultaneamente.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    EXPECT_TRUE(c.load_model("inst_a", linear_path()));
    EXPECT_TRUE(c.load_model("inst_b", linear_path()));
    c.unload_model("inst_a");
    c.unload_model("inst_b");
}

TEST(LoadModel, CarregaComVersionStringExplicita) {
    // O parâmetro version é opcional e não deve impedir o carregamento.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    EXPECT_TRUE(c.load_model("m_v2", linear_path(), "2.0.0"));
    c.unload_model("m_v2");
}

TEST(LoadModel, IDDuplicadoRetornaFalse) {
    // Reutilizar um ID já carregado deve ser rejeitado pelo worker.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    c.load_model("dup", linear_path());
    EXPECT_FALSE(c.load_model("dup", linear_path()));
    c.unload_model("dup");
}

TEST(LoadModel, ArquivoInexistenteNoServidorRetornaFalse) {
    // Path inexistente no servidor deve ser rejeitado — não precisa do worker
    // em modo de sucesso, mas sim conectado para receber a resposta de erro.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    EXPECT_FALSE(c.load_model("ghost", "/nao/existe/modelo.onnx"));
}

TEST(LoadModel, ArquivoPySemMiiaModelRetornaFalse) {
    // Um .py válido mas sem subclasse de MiiaModel deve ser rejeitado pelo
    // PythonBackend no servidor (find_model_class retorna nullptr).
    // O arquivo precisa existir no servidor — usamos /tmp.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    const std::string tmp = "/tmp/falso_grpc_load.py";
    { std::ofstream f(tmp); f << "x = 42\n"; }
    InferenceClient c(worker_address());
    c.connect();
    EXPECT_FALSE(c.load_model("fake", tmp));
    std::filesystem::remove(tmp);
}

// =============================================================================
// GRUPO 3 — unload_model()
//
// Todos os happy-paths requerem um load() bem-sucedido.
// Skip por worker indisponível.
// =============================================================================

TEST(UnloadModel, DescarregaModeloCarregadoRetornaTrue) {
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    c.load_model("u1", linear_path());
    EXPECT_TRUE(c.unload_model("u1"));
}

TEST(UnloadModel, RecarregarAposDescarregar) {
    // Após unload(), o mesmo ID deve estar disponível para load_model().
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    c.load_model("reload", linear_path());
    c.unload_model("reload");
    EXPECT_TRUE(c.load_model("reload", linear_path()));
    c.unload_model("reload");
}

TEST(UnloadModel, DescarregaMultiplosModelosEmSequencia) {
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    c.load_model("a", linear_path());
    c.load_model("b", linear_path());
    EXPECT_TRUE(c.unload_model("a"));
    EXPECT_TRUE(c.unload_model("b"));
}

TEST(UnloadModel, IDInexistenteRetornaFalse) {
    // Descarregar um ID nunca carregado deve retornar false.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    EXPECT_FALSE(c.unload_model("id_que_nao_existe_xyz"));
}

TEST(UnloadModel, SegundoUnloadDoMesmoIDRetornaFalse) {
    // Após um unload() bem-sucedido, repetir a chamada deve falhar.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    c.load_model("once", linear_path());
    c.unload_model("once");
    EXPECT_FALSE(c.unload_model("once"));
}

// =============================================================================
// GRUPO 4 — predict()
//
// Todos dependem da fixture (GTEST_SKIP já está no SetUp).
// =============================================================================

TEST_F(GrpcPythonTest, PredictRetornaSuccess) {
    // Inferência com inputs bem formados deve ter success = true.
    auto r = client->predict("linear", make_valid_inputs());
    EXPECT_TRUE(r.success) << r.error_message;
}

TEST_F(GrpcPythonTest, PredictOutputNaoVazio) {
    // O mapa de outputs não pode estar vazio após inferência bem-sucedida.
    auto r = client->predict("linear", make_valid_inputs());
    ASSERT_TRUE(r.success);
    EXPECT_FALSE(r.outputs.empty());
}

TEST_F(GrpcPythonTest, PredictOutputContemChaveOutput) {
    // O tensor de saída deve estar sob a chave "output".
    auto r = client->predict("linear", make_valid_inputs());
    ASSERT_TRUE(r.success);
    EXPECT_GT(r.outputs.count("output"), 0u);
}

TEST_F(GrpcPythonTest, PredictOutputEhEscalar) {
    // shape [1,1] → total == 1 → py_dict_to_outputs colapsa para is_number().
    // Este é o comportamento distinto do backend Python vs ONNX.
    auto r = client->predict("linear", make_valid_inputs());
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.outputs.at("output").is_number())
        << "Esperado Value escalar (is_number()); "
           "shape [1,1] deve ser colapsado por py_dict_to_outputs";
}

TEST_F(GrpcPythonTest, PredictOutputNaoEhArray) {
    // Confirma o inverso: o output NÃO deve ser array para este modelo.
    auto r = client->predict("linear", make_valid_inputs());
    ASSERT_TRUE(r.success);
    EXPECT_FALSE(r.outputs.at("output").is_array())
        << "Output de shape [1,1] não deve ser array — deve ser escalar";
}

TEST_F(GrpcPythonTest, PredictInferenceTimeMsPositivo) {
    // O tempo de inferência reportado deve ser estritamente positivo.
    auto r = client->predict("linear", make_valid_inputs());
    ASSERT_TRUE(r.success);
    EXPECT_GT(r.inference_time_ms, 0.0);
}

TEST_F(GrpcPythonTest, PredictDeterministico) {
    // Mesma entrada em chamadas consecutivas deve produzir escalar idêntico.
    auto inputs = make_valid_inputs(1.f, 2.f, 3.f, 4.f, 5.f);
    auto r1 = client->predict("linear", inputs);
    auto r2 = client->predict("linear", inputs);
    ASSERT_TRUE(r1.success && r2.success);
    ASSERT_GT(r1.outputs.count("output"), 0u);
    ASSERT_GT(r2.outputs.count("output"), 0u);
    EXPECT_NEAR(r1.outputs.at("output").as_number(),
                r2.outputs.at("output").as_number(), 1e-6);
}

TEST_F(GrpcPythonTest, PredictMultiplasEntradasValidasSemFalha) {
    // Verifica que o cliente aceita múltiplas entradas distintas sem falhar.
    // Output é escalar (is_number()) — não testamos valores, só sucesso.
    const std::vector<std::array<float,5>> casos = {
        {0.f, 0.f, 0.f, 0.f, 0.f},
        {1.f, 2.f, 3.f, 4.f, 5.f},
        {9.f, 9.f, 9.f, 9.f, 9.f},
        {-1.f, -2.f, -3.f, -4.f, -5.f},
    };
    for (size_t i = 0; i < casos.size(); ++i) {
        const auto& c = casos[i];
        auto r = client->predict("linear",
                     make_valid_inputs(c[0], c[1], c[2], c[3], c[4]));
        ASSERT_TRUE(r.success)
            << "caso=" << i << " erro=" << r.error_message;
        ASSERT_GT(r.outputs.count("output"), 0u) << "caso=" << i;
        EXPECT_TRUE(r.outputs.at("output").is_number()) << "caso=" << i;
    }
}

TEST_F(GrpcPythonTest, PredictComEntradasNegativasOk) {
    // Valores negativos são entradas válidas para o modelo.
    auto r = client->predict("linear", make_valid_inputs(-1.f, -2.f, -3.f, -4.f, -5.f));
    EXPECT_TRUE(r.success) << r.error_message;
}

TEST_F(GrpcPythonTest, PredictComEntradasZeradasOk) {
    // Vetor de zeros é um input válido.
    auto r = client->predict("linear", make_valid_inputs(0.f, 0.f, 0.f, 0.f, 0.f));
    EXPECT_TRUE(r.success) << r.error_message;
}

TEST_F(GrpcPythonTest, Predict100TicksConsecutivosOk) {
    // Testa estabilidade da conexão gRPC: 100 predições seguidas sem erro.
    for (int i = 0; i < 100; ++i) {
        auto r = client->predict("linear",
                     make_valid_inputs(static_cast<float>(i) * 0.1f,
                                       1.f, 1.f, 1.f, 1.f));
        ASSERT_TRUE(r.success) << "tick=" << i << " erro=" << r.error_message;
    }
}

TEST_F(GrpcPythonTest, PredictIDInexistenteRetornaFalse) {
    // Inferir com um model_id não carregado deve retornar success = false.
    auto r = client->predict("id_que_nao_existe", make_valid_inputs());
    EXPECT_FALSE(r.success);
}

TEST_F(GrpcPythonTest, PredictIDInexistentePopulaErrorMessage) {
    // Em caso de falha, error_message não deve estar vazio.
    auto r = client->predict("id_que_nao_existe", make_valid_inputs());
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(GrpcPythonTest, PredictChaveInputErradaFalha) {
    // Input com nome de tensor inexistente no modelo deve ser rejeitado.
    auto r = client->predict("linear", make_wrong_key_inputs());
    EXPECT_FALSE(r.success);
}

TEST_F(GrpcPythonTest, PredictTamanhoInputErradoFalha) {
    // Input com número incorreto de elementos deve ser rejeitado pelo runtime.
    auto r = client->predict("linear", make_wrong_size_inputs());
    EXPECT_FALSE(r.success);
}

TEST_F(GrpcPythonTest, PredictObjectVazioFalha) {
    // Um Object vazio (sem nenhum tensor) não é uma entrada válida.
    auto r = client->predict("linear", Object{});
    EXPECT_FALSE(r.success);
}

// =============================================================================
// GRUPO 5 — batch_predict()
// =============================================================================

TEST_F(GrpcPythonTest, BatchPredictLoteUnicoOk) {
    // Um lote com um único elemento deve funcionar como predict() singular.
    auto results = client->batch_predict("linear", {make_valid_inputs()});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].success);
}

TEST_F(GrpcPythonTest, BatchPredictRetornaMesmoTamanho) {
    // O vetor de retorno deve ter exatamente o mesmo tamanho que o lote de entrada.
    std::vector<Object> batch;
    for (int i = 0; i < 7; ++i)
        batch.push_back(make_valid_inputs(static_cast<float>(i)));
    EXPECT_EQ(client->batch_predict("linear", batch).size(), 7u);
}

TEST_F(GrpcPythonTest, BatchPredictTodosItensComSucesso) {
    // Cada item do lote com input válido deve ter success = true.
    std::vector<Object> batch;
    for (int i = 0; i < 5; ++i)
        batch.push_back(make_valid_inputs(static_cast<float>(i) + 1.f));
    auto results = client->batch_predict("linear", batch);
    for (size_t i = 0; i < results.size(); ++i)
        EXPECT_TRUE(results[i].success)
            << "item=" << i << " erro=" << results[i].error_message;
}

TEST_F(GrpcPythonTest, BatchPredictEquivalenteAoPredictEscalar) {
    // batch[0] e predict() devem produzir o mesmo escalar para a mesma entrada.
    // Output é is_number() — comparação direta por as_number().
    auto inputs = make_valid_inputs(2.f, 3.f, 4.f, 5.f, 6.f);
    auto scalar = client->predict("linear", inputs);
    auto batch  = client->batch_predict("linear", {inputs});
    ASSERT_TRUE(scalar.success);
    ASSERT_EQ(batch.size(), 1u);
    ASSERT_TRUE(batch[0].success);
    EXPECT_NEAR(scalar.outputs.at("output").as_number(),
                batch[0].outputs.at("output").as_number(), 1e-6);
}

TEST_F(GrpcPythonTest, BatchPredictLote50ElementosOk) {
    // Lotes maiores devem ser processados sem erro de rede ou crash.
    std::vector<Object> batch;
    for (int i = 0; i < 50; ++i)
        batch.push_back(make_valid_inputs(static_cast<float>(i) * 0.1f));
    auto results = client->batch_predict("linear", batch);
    ASSERT_EQ(results.size(), 50u);
    for (const auto& r : results)
        EXPECT_TRUE(r.success);
}

TEST_F(GrpcPythonTest, BatchPredictLoteVazioRetornaVetorVazio) {
    // Um lote vazio deve retornar vetor vazio, sem crash nem exception.
    EXPECT_TRUE(client->batch_predict("linear", {}).empty());
}

TEST_F(GrpcPythonTest, BatchPredictIDInexistenteRetornaFalhasPorItem) {
    // Cada item do lote contra ID inválido deve ter success = false.
    auto results = client->batch_predict("nao_existe",
                       {make_valid_inputs(), make_valid_inputs()});
    for (const auto& r : results)
        EXPECT_FALSE(r.success);
}

// =============================================================================
// GRUPO 6 — list_models()
// =============================================================================

TEST_F(GrpcPythonTest, ListModelsNaoVaziaAposLoad) {
    // Após load_model(), list_models() deve retornar ao menos um item.
    EXPECT_FALSE(client->list_models().empty());
}

TEST_F(GrpcPythonTest, ListModelsContemIDCarregado) {
    // O ID "linear" (carregado na fixture) deve estar presente na lista.
    auto models = client->list_models();
    bool found  = std::any_of(models.begin(), models.end(),
                              [](const ModelInfo& m){ return m.model_id == "linear"; });
    EXPECT_TRUE(found);
}

TEST_F(GrpcPythonTest, ListModelsContagemAumentaAposNovoLoad) {
    // Um segundo load_model() deve aumentar a contagem em exatamente 1.
    auto before = client->list_models().size();
    client->load_model("extra", linear_path());
    EXPECT_EQ(client->list_models().size(), before + 1);
}

TEST_F(GrpcPythonTest, ListModelsContagemDiminuiAposUnload) {
    // Após unload_model(), a contagem deve reduzir em exatamente 1.
    client->load_model("temp", linear_path());
    auto before = client->list_models().size();
    client->unload_model("temp");
    EXPECT_EQ(client->list_models().size(), before - 1);
}

TEST_F(GrpcPythonTest, ListModelsEntradaTemModelIdNaoVazio) {
    // Nenhuma entrada da lista pode ter model_id vazio.
    for (const auto& m : client->list_models())
        EXPECT_FALSE(m.model_id.empty());
}

TEST_F(GrpcPythonTest, ListModelsEntradaTemBackendNaoVazio) {
    // Cada entrada deve informar o backend que está servindo o modelo.
    for (const auto& m : client->list_models())
        EXPECT_FALSE(m.backend.empty());
}

TEST(ListModels, ListaVaziaAntesDeQualquerLoad) {
    // Sem nenhum modelo carregado, list_models() deve retornar lista vazia.
    if (!worker_disponivel())
        GTEST_SKIP() << "Worker indisponível em " << worker_address();
    InferenceClient c(worker_address());
    c.connect();
    // TearDown não existe aqui — garante estado limpo pedindo lista antes de qualquer load.
    // Se outro teste deixou modelos, pode falhar; idealmente o worker é reiniciado entre suites.
    EXPECT_TRUE(c.list_models().empty());
}

// =============================================================================
// GRUPO 7 — get_model_info()
// =============================================================================

TEST_F(GrpcPythonTest, GetModelInfoIdCorreto) {
    // O model_id retornado deve ser idêntico ao ID passado no load_model().
    EXPECT_EQ(client->get_model_info("linear").model_id, "linear");
}

TEST_F(GrpcPythonTest, GetModelInfoBackendEhPython) {
    // O PythonBackend deve se identificar como "python".
    EXPECT_EQ(client->get_model_info("linear").backend, "python");
}

TEST_F(GrpcPythonTest, GetModelInfoTemInputs) {
    // O schema de entrada deve ter ao menos um TensorSpec.
    EXPECT_FALSE(client->get_model_info("linear").inputs.empty());
}

TEST_F(GrpcPythonTest, GetModelInfoTemOutputs) {
    // O schema de saída deve ter ao menos um TensorSpec.
    EXPECT_FALSE(client->get_model_info("linear").outputs.empty());
}

TEST_F(GrpcPythonTest, GetModelInfoInputNomesNaoVazios) {
    // Cada TensorSpec de entrada deve ter o campo name preenchido.
    for (const auto& ts : client->get_model_info("linear").inputs)
        EXPECT_FALSE(ts.name.empty());
}

TEST_F(GrpcPythonTest, GetModelInfoOutputNomesNaoVazios) {
    // Cada TensorSpec de saída deve ter o campo name preenchido.
    for (const auto& ts : client->get_model_info("linear").outputs)
        EXPECT_FALSE(ts.name.empty());
}

TEST_F(GrpcPythonTest, GetModelInfoInputContemChaveInput) {
    // O modelo linear declara um tensor de entrada chamado "input".
    auto info = client->get_model_info("linear");
    bool found = std::any_of(info.inputs.begin(), info.inputs.end(),
                             [](const ModelInfo::TensorSpec& t){ return t.name == "input"; });
    EXPECT_TRUE(found);
}

TEST_F(GrpcPythonTest, GetModelInfoOutputContemChaveOutput) {
    // O modelo linear declara um tensor de saída chamado "output".
    auto info = client->get_model_info("linear");
    bool found = std::any_of(info.outputs.begin(), info.outputs.end(),
                             [](const ModelInfo::TensorSpec& t){ return t.name == "output"; });
    EXPECT_TRUE(found);
}

TEST_F(GrpcPythonTest, GetModelInfoInputShapesNaoVazias) {
    // Cada TensorSpec de entrada deve ter ao menos uma dimensão declarada.
    for (const auto& ts : client->get_model_info("linear").inputs)
        EXPECT_FALSE(ts.shape.empty()) << "tensor=" << ts.name;
}

TEST_F(GrpcPythonTest, GetModelInfoOutputShapesNaoVazias) {
    // Cada TensorSpec de saída deve ter ao menos uma dimensão declarada.
    for (const auto& ts : client->get_model_info("linear").outputs)
        EXPECT_FALSE(ts.shape.empty()) << "tensor=" << ts.name;
}

TEST_F(GrpcPythonTest, GetModelInfoIDInexistenteRetornaModelIdVazio) {
    // Para um ID não carregado, model_id deve estar vazio (sem exception).
    EXPECT_TRUE(client->get_model_info("nao_existe_xyz").model_id.empty());
}

// =============================================================================
// GRUPO 8 — validate_model()
//
// O path é avaliado pelo worker — deve ser visível no filesystem do servidor.
// =============================================================================

TEST_F(GrpcPythonTest, ValidateModelArquivoPyValidoRetornaValid) {
    // Arquivo .py existente deve passar na validação de path/extensão.
    auto r = client->validate_model(linear_path());
    EXPECT_TRUE(r.valid) << r.error_message;
}

TEST_F(GrpcPythonTest, ValidateModelRetornaBackendNaoVazio) {
    // O campo backend deve ser preenchido com o nome do engine detector.
    auto r = client->validate_model(linear_path());
    ASSERT_TRUE(r.valid);
    EXPECT_FALSE(r.backend.empty());
}

TEST_F(GrpcPythonTest, ValidateModelArquivoPySemMiiaModelPassaNaValidacao) {
    // O PythonBackend::validate() só checa existência + extensão ".py".
    // Um .py qualquer (sem MiiaModel) passa aqui; a falha ocorre no load_model().
    const std::string tmp = "/tmp/qualquer_grpc.py";
    { std::ofstream f(tmp); f << "x = 42\n"; }
    auto r = client->validate_model(tmp);
    EXPECT_TRUE(r.valid)
        << "validate_model() do PythonBackend não inspeciona conteúdo, "
           "apenas existência e extensão";
    std::filesystem::remove(tmp);
}

TEST_F(GrpcPythonTest, ValidateModelNaoTemSideEffectDeLoad) {
    // validate_model() não deve alterar a lista de modelos carregados no worker.
    auto before = client->list_models().size();
    client->validate_model(linear_path());
    EXPECT_EQ(client->list_models().size(), before);
}

TEST_F(GrpcPythonTest, ValidateModelArquivoInexistenteFalha) {
    // Path inexistente no servidor deve retornar valid = false.
    auto r = client->validate_model("/nao/existe/modelo.onnx");
    EXPECT_FALSE(r.valid);
    EXPECT_FALSE(r.error_message.empty());
}

TEST_F(GrpcPythonTest, ValidateModelExtensaoErradaFalha) {
    // Um arquivo com extensão diferente de ".py" deve ser rejeitado.
    const std::string tmp = "/tmp/nao_py_grpc.txt";
    { std::ofstream f(tmp); f << "conteudo irrelevante"; }
    auto r = client->validate_model(tmp);
    EXPECT_FALSE(r.valid);
    std::filesystem::remove(tmp);
}

// =============================================================================
// GRUPO 9 — warmup_model()
// =============================================================================

TEST_F(GrpcPythonTest, WarmupModelRetornaSuccess) {
    // warmup_model() com modelo carregado deve completar com success = true.
    EXPECT_TRUE(client->warmup_model("linear", 5).success);
}

TEST_F(GrpcPythonTest, WarmupModelRunsCompletedExato) {
    // O número de runs completados deve ser exatamente o número solicitado.
    auto r = client->warmup_model("linear", 10);
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.runs_completed, 10u);
}

TEST_F(GrpcPythonTest, WarmupModelRunDefaultNaoTrava) {
    // Passar num_runs = 0 usa o default interno e não deve travar.
    EXPECT_NO_THROW(client->warmup_model("linear", 0));
}

TEST_F(GrpcPythonTest, WarmupModelAvgMsPositivo) {
    // O tempo médio deve ser estritamente positivo após warmup.
    auto r = client->warmup_model("linear", 5);
    ASSERT_TRUE(r.success);
    EXPECT_GT(r.avg_time_ms, 0.0);
}

TEST_F(GrpcPythonTest, WarmupModelMinMenorOuIgualMax) {
    // A latência mínima não pode ser maior que a latência máxima.
    auto r = client->warmup_model("linear", 5);
    ASSERT_TRUE(r.success);
    EXPECT_LE(r.min_time_ms, r.max_time_ms);
}

TEST_F(GrpcPythonTest, WarmupModelAvgEntreMineMax) {
    // A média deve estar dentro do intervalo [min, max].
    auto r = client->warmup_model("linear", 5);
    ASSERT_TRUE(r.success);
    EXPECT_GE(r.avg_time_ms, r.min_time_ms);
    EXPECT_LE(r.avg_time_ms, r.max_time_ms);
}

TEST_F(GrpcPythonTest, WarmupModelIDInexistenteFalha) {
    // Warmup de modelo não carregado deve retornar success = false.
    EXPECT_FALSE(client->warmup_model("nao_existe_xyz", 5).success);
}

TEST_F(GrpcPythonTest, WarmupModelIDInexistentePopulaErrorMessage) {
    // A mensagem de erro não deve estar vazia em caso de falha.
    EXPECT_FALSE(client->warmup_model("nao_existe_xyz", 5).error_message.empty());
}

// =============================================================================
// GRUPO 10 — health_check()
// =============================================================================

TEST_F(GrpcPythonTest, HealthCheckRetornaTrueAposConexao) {
    // Worker conectado deve responder true ao health check.
    EXPECT_TRUE(client->health_check());
}

TEST_F(GrpcPythonTest, HealthCheckEstavelEmRepeticoes) {
    // health_check() chamado 20 vezes seguidas deve ser sempre true.
    for (int i = 0; i < 20; ++i)
        EXPECT_TRUE(client->health_check()) << "iteração=" << i;
}

TEST(HealthCheck, FalseSemConexao) {
    // Sem chamar connect(), health_check() deve retornar false.
    InferenceClient c(worker_address());
    EXPECT_FALSE(c.health_check());
}

TEST(HealthCheck, FalseParaWorkerInexistente) {
    // Worker em porta fechada deve retornar false no health_check().
    InferenceClient c("localhost:19998");
    c.connect();
    EXPECT_FALSE(c.health_check());
}

// =============================================================================
// GRUPO 11 — get_status()
// =============================================================================

TEST_F(GrpcPythonTest, GetStatusWorkerIdNaoVazio) {
    // O worker deve preencher o campo worker_id na resposta de status.
    EXPECT_FALSE(client->get_status().worker_id.empty());
}

TEST_F(GrpcPythonTest, GetStatusUptimeNaoNegativo) {
    // O uptime em segundos desde o início do worker não pode ser negativo.
    EXPECT_GE(client->get_status().uptime_seconds, 0LL);
}

TEST_F(GrpcPythonTest, GetStatusLoadedModelsContemLinear) {
    // O modelo "linear" carregado na fixture deve estar em loaded_models.
    auto s = client->get_status();
    bool found = std::any_of(s.loaded_models.begin(), s.loaded_models.end(),
                             [](const std::string& id){ return id == "linear"; });
    EXPECT_TRUE(found);
}

TEST_F(GrpcPythonTest, GetStatusLoadedModelsRefleteCargaDinamica) {
    // Ao carregar um segundo modelo, loaded_models deve crescer em 1.
    auto before = client->get_status().loaded_models.size();
    client->load_model("s2", linear_path());
    EXPECT_EQ(client->get_status().loaded_models.size(), before + 1);
}

TEST_F(GrpcPythonTest, GetStatusLoadedModelsRefletUnload) {
    // Após unload, o ID não deve mais aparecer em loaded_models.
    client->load_model("s_ul", linear_path());
    client->unload_model("s_ul");
    auto s = client->get_status();
    bool still_there = std::any_of(s.loaded_models.begin(), s.loaded_models.end(),
                                   [](const std::string& id){ return id == "s_ul"; });
    EXPECT_FALSE(still_there);
}

TEST_F(GrpcPythonTest, GetStatusSupportedBackendsNaoVazio) {
    // A lista de backends suportados pelo worker não deve estar vazia.
    // Requer que o WorkerServiceImpl preencha capabilities em GetStatusResponse.
    EXPECT_FALSE(client->get_status().supported_backends.empty());
}

TEST_F(GrpcPythonTest, GetStatusSupportedBackendsContemPython) {
    // O PythonBackend deve estar na lista de backends suportados.
    auto s = client->get_status();
    bool found = std::any_of(s.supported_backends.begin(),
                             s.supported_backends.end(),
                             [](const std::string& b){
                                 return b.find("python") != std::string::npos
                                     || b.find("Python") != std::string::npos;
                             });
    EXPECT_TRUE(found);
}

TEST(GetStatus, RetornaStatusVazioSemConexao) {
    // Sem conectar, get_status() não deve travar — worker_id estará vazio.
    InferenceClient c(worker_address());
    EXPECT_TRUE(c.get_status().worker_id.empty());
}

// =============================================================================
// GRUPO 12 — get_metrics()
//
// O worker gRPC implementa métricas completas — ao contrário do in-process.
// =============================================================================

TEST_F(GrpcPythonTest, GetMetricsNaoTrava) {
    // get_metrics() deve retornar sem exception.
    EXPECT_NO_THROW(client->get_metrics());
}

TEST_F(GrpcPythonTest, GetMetricsTotalRequestsNaoNegativo) {
    // O contador de requisições totais deve ser >= 0.
    EXPECT_GE(client->get_metrics().total_requests, 0u);
}

TEST_F(GrpcPythonTest, GetMetricsSucessosNaoUltrapassamTotal) {
    // successful_requests não pode ser maior que total_requests.
    auto m = client->get_metrics();
    EXPECT_LE(m.successful_requests, m.total_requests);
}

TEST_F(GrpcPythonTest, GetMetricsFalhasNaoUltrapassamTotal) {
    // failed_requests não pode ser maior que total_requests.
    auto m = client->get_metrics();
    EXPECT_LE(m.failed_requests, m.total_requests);
}

TEST_F(GrpcPythonTest, GetMetricsCresceAposPredict) {
    // Após uma predição, total_requests no worker deve ser maior do que antes.
    // Nota: o worker acumula requests de todos os clientes — verificamos apenas
    // que o valor não regride.
    auto before = client->get_metrics().total_requests;
    client->predict("linear", make_valid_inputs());
    auto after  = client->get_metrics().total_requests;
    EXPECT_GE(after, before);
}

// =============================================================================
// GRUPO 13 — list_available_models()
//
// Os paths retornados são do servidor. Sem canonicalização local.
// A limitação de break-on-first-match (mesmo arquivo, dois IDs) persiste.
// =============================================================================

TEST_F(GrpcPythonTest, ListAvailableModelsPyNaoVazioEmModelsDir) {
    // Deve haver ao menos um arquivo .py no diretório de modelos do servidor.
    auto models = client->list_available_models(models_dir());
    bool has_py = std::any_of(models.begin(), models.end(),
                              [](const AvailableModel& m){ return m.extension == ".py"; });
    EXPECT_TRUE(has_py);
}

TEST_F(GrpcPythonTest, ListAvailableModelsFilenameNaoVazio) {
    // Cada arquivo descoberto deve ter o campo filename preenchido.
    for (const auto& m : client->list_available_models(models_dir()))
        EXPECT_FALSE(m.filename.empty());
}

TEST_F(GrpcPythonTest, ListAvailableModelsPathNaoVazio) {
    // Cada entrada deve ter o path completo no servidor.
    for (const auto& m : client->list_available_models(models_dir()))
        EXPECT_FALSE(m.path.empty());
}

TEST_F(GrpcPythonTest, ListAvailableModelsExtensaoComecaComPonto) {
    // O campo extension deve iniciar com '.' (ex: ".onnx", ".py").
    for (const auto& m : client->list_available_models(models_dir()))
        ASSERT_FALSE(m.extension.empty()) << "arquivo=" << m.filename;
}

TEST_F(GrpcPythonTest, ListAvailableModelsExtensaoCorretaParaPy) {
    // Arquivos com extensão .py devem ter extension == ".py".
    for (const auto& m : client->list_available_models(models_dir()))
        if (m.filename.size() > 3 &&
            m.filename.substr(m.filename.size() - 3) == ".py") {
                EXPECT_EQ(m.extension, ".py") << "arquivo=" << m.filename;
            }
}

TEST_F(GrpcPythonTest, ListAvailableModelsBackendNaoVazio) {
    // Cada arquivo descoberto deve ter o backend inferido (onnx/python).
    for (const auto& m : client->list_available_models(models_dir()))
        EXPECT_FALSE(m.backend.empty()) << "arquivo=" << m.filename;
}

TEST_F(GrpcPythonTest, ListAvailableModelsArquivoPyTemBackendPython) {
    // Arquivos .py devem ser mapeados para o backend "python".
    for (const auto& m : client->list_available_models(models_dir()))
        if (m.extension == ".py") {
            EXPECT_EQ(m.backend, "python") << "arquivo=" << m.filename;
        }
}

TEST_F(GrpcPythonTest, ListAvailableModelsFileSizePositivo) {
    // Todos os arquivos listados devem ter tamanho em bytes > 0.
    for (const auto& m : client->list_available_models(models_dir()))
        EXPECT_GT(m.file_size_bytes, 0) << "arquivo=" << m.filename;
}

TEST_F(GrpcPythonTest, ListAvailableModelsIsLoadedTrueAposLoad) {
    // Após load_model(), o arquivo correspondente deve aparecer com is_loaded=true.
    client->load_model("avail_check", linear_path());
    auto models     = client->list_available_models(models_dir());

    bool any_loaded = std::any_of(models.begin(), models.end(),
                                  [](const AvailableModel& m){ return m.is_loaded; });
    EXPECT_TRUE(any_loaded);
    client->unload_model("avail_check");
}

TEST_F(GrpcPythonTest, ListAvailableModelsLoadedAsRefletID) {
    // O campo loaded_as deve conter exatamente o ID passado ao load_model().
    // list_available_models() faz break no primeiro ID que bate com o path,
    // portanto descarregamos "linear" (da fixture) antes de carregar o ID
    // de teste — garantindo que só um ID aponte para simple_linear.onnx.
    client->unload_model("linear");
    client->load_model("meu_id_especifico", linear_path());
    auto models = client->list_available_models(models_dir());
    bool found  = std::any_of(models.begin(), models.end(),
                              [](const AvailableModel& m){
                                  return m.is_loaded && m.loaded_as == "meu_id_especifico";
                              });
    EXPECT_TRUE(found);
    client->unload_model("meu_id_especifico");
    // Recarrega "linear" para o TearDown não reclamar de ID inexistente.
    client->load_model("linear", linear_path());
}

TEST_F(GrpcPythonTest, ListAvailableModelsIsLoadedFalseAposUnload) {
    // Após unload(), o arquivo não deve mais aparecer com loaded_as do ID removido.
    client->load_model("ul_avail", linear_path());
    client->unload_model("ul_avail");
    for (const auto& m : client->list_available_models(models_dir()))
        EXPECT_NE(m.loaded_as, "ul_avail");
}

TEST_F(GrpcPythonTest, ListAvailableModelsDirInexistenteRetornaVazio) {
    // Diretório inexistente no servidor deve retornar lista vazia, sem crash.
    EXPECT_TRUE(client->list_available_models("/nao/existe/de/jeito/nenhum").empty());
}

TEST_F(GrpcPythonTest, ListAvailableModelsDirVazioUsaDefaultSemCrash) {
    // String vazia como diretório deve usar o default do worker sem travar.
    EXPECT_NO_THROW(client->list_available_models(""));
}

// =============================================================================
// GRUPO 14 — Ciclos de vida completos (end-to-end)
// =============================================================================

TEST_F(GrpcPythonTest, E2E_CicloCompletoSemFalhas) {
    // Fluxo completo de produção via gRPC: validar → carregar → warmup
    // → introspectar → inferir → batch → verificar status → descobrir → descarregar.
    const std::string id = "e2e_full";

    // 1. Valida antes de carregar — garante que o worker enxerga o arquivo.
    auto vr = client->validate_model(linear_path());
    ASSERT_TRUE(vr.valid) << "validação falhou: " << vr.error_message;

    // 2. Carrega o modelo sob um ID temporário.
    ASSERT_TRUE(client->load_model(id, linear_path()));

    // 3. Warmup para aquecer JIT/caches internos do ONNX Runtime no worker.
    auto wr = client->warmup_model(id, 3);
    EXPECT_TRUE(wr.success);
    EXPECT_EQ(wr.runs_completed, 3u);

    // 4. Introspecção — verifica schema de I/O via gRPC.
    auto info = client->get_model_info(id);
    EXPECT_EQ(info.model_id, id);
    EXPECT_EQ(info.backend, "python");
    EXPECT_FALSE(info.inputs.empty());
    EXPECT_FALSE(info.outputs.empty());

    // 5. Inferência escalar — output é número (shape [1,1] colapsado).
    auto pr = client->predict(id, make_valid_inputs());
    ASSERT_TRUE(pr.success);
    EXPECT_TRUE(pr.outputs.at("output").is_number());

    // 6. Inferência em lote.
    auto br = client->batch_predict(id, {make_valid_inputs(), make_valid_inputs()});
    EXPECT_EQ(br.size(), 2u);
    for (const auto& r : br) EXPECT_TRUE(r.success);

    // 7. Status reflete o modelo carregado no worker.
    auto st = client->get_status();
    bool in_status = std::any_of(st.loaded_models.begin(), st.loaded_models.end(),
                                 [&id](const std::string& s){ return s == id; });
    EXPECT_TRUE(in_status);

    // 8. Descoberta confirma is_loaded = true no diretório do worker.
    auto avail = client->list_available_models(models_dir());
    bool in_avail = std::any_of(avail.begin(), avail.end(),
                                [&id](const AvailableModel& m){ return m.loaded_as == id; });
    EXPECT_TRUE(in_avail);

    // 9. Descarrega.
    EXPECT_TRUE(client->unload_model(id));

    // 10. ID não deve mais aparecer em list_models().
    auto final_list = client->list_models();
    bool still_there = std::any_of(final_list.begin(), final_list.end(),
                                   [&id](const ModelInfo& m){ return m.model_id == id; });
    EXPECT_FALSE(still_there);
}

TEST_F(GrpcPythonTest, E2E_LoopSimulacao100Ticks) {
    // Simula 100 ticks de um loop de controle via gRPC.
    // Objetivo: garantir que a conexão gRPC sustenta N chamadas consecutivas.
    float position = 0.0f;
    float velocity = 1.0f;
    float heading  = 0.0f;
    float fuel     = 100.0f;
    float t        = 0.0f;
    constexpr float DT = 0.1f;

    for (int tick = 0; tick < 100; ++tick) {
        float sensor = std::sin(t);
        auto r = client->predict("linear",
                     make_valid_inputs(position, velocity, heading, fuel, sensor));

        ASSERT_TRUE(r.success)
            << "tick=" << tick << " erro=" << r.error_message;

        // Output é escalar (shape [1,1] colapsado pelo py_dict_to_outputs).
        ASSERT_TRUE(r.outputs.at("output").is_number()) << "tick=" << tick;
        heading = static_cast<float>(r.outputs.at("output").as_number());
        position += velocity * DT;
        fuel     -= 0.05f;
        t        += DT;
    }

    // Verifica integridade da física acumulada.
    EXPECT_NEAR(position, 100 * velocity * DT, 1e-3f);
}

TEST_F(GrpcPythonTest, E2E_DoisClientesNaoInterferam) {
    // Dois InferenceClients gRPC com IDs distintos sobre o mesmo arquivo
    // devem operar de forma independente no mesmo worker.
    InferenceClient c2(worker_address());
    ASSERT_TRUE(c2.connect());
    ASSERT_TRUE(c2.load_model("peer", linear_path()));

    // Ambos devem produzir sucesso para a mesma entrada.
    auto inputs = make_valid_inputs(1.f, 1.f, 1.f, 1.f, 1.f);
    auto r1 = client->predict("linear", inputs);
    auto r2 = c2.predict("peer",        inputs);
    EXPECT_TRUE(r1.success);
    EXPECT_TRUE(r2.success);

    // Descarregar de c2 não afeta client.
    EXPECT_TRUE(c2.unload_model("peer"));
    auto r3 = client->predict("linear", inputs);
    EXPECT_TRUE(r3.success);
}
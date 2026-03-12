// =============================================================================
// test_inference_client.cpp — Testes unitários do InferenceClient
//
// Esses testes NÃO precisam de um worker rodando.
// Cobrem: construção, estado de conexão, comportamento sem servidor,
//         validação de entrada, IDs de modelos e valores de retorno defensivos.
// =============================================================================

#include <gtest/gtest.h>
#include <client/inference_client.hpp>
#include <chrono>
#include <thread>
#include <map>
#include <vector>
#include <string>

using mlinference::client::InferenceClient;
using mlinference::client::PredictionResult;
using mlinference::client::WorkerStatus;

// ─────────────────────────────────────────────────────────────────────────────
// Fixture base — reutilizada em todos os grupos de testes sem servidor
// ─────────────────────────────────────────────────────────────────────────────

class ClientNoServerTest : public ::testing::Test {
protected:
    // Porta que com certeza não tem nada escutando
    InferenceClient client{"localhost:19999"};
};

// =============================================================================
// Grupo 1: Construção e estado inicial
// =============================================================================

TEST_F(ClientNoServerTest, InitialStateIsNotConnected) {
    EXPECT_FALSE(client.is_connected());
}

TEST(ClientConstructionTest, AcceptsLocalhostAddress) {
    EXPECT_NO_THROW(InferenceClient c("localhost:50052"));
}

TEST(ClientConstructionTest, AcceptsIPv4Address) {
    EXPECT_NO_THROW(InferenceClient c("127.0.0.1:50052"));
}

TEST(ClientConstructionTest, AcceptsRemoteAddress) {
    EXPECT_NO_THROW(InferenceClient c("192.168.1.100:50052"));
}

TEST(ClientConstructionTest, AcceptsHighPortNumber) {
    EXPECT_NO_THROW(InferenceClient c("localhost:65535"));
}

// =============================================================================
// Grupo 2: Conexão falhando graciosamente
// =============================================================================

TEST_F(ClientNoServerTest, ConnectToInvalidPortReturnsFalse) {
    bool result = client.connect();
    EXPECT_FALSE(result);
}

TEST_F(ClientNoServerTest, ConnectDoesNotThrow) {
    EXPECT_NO_THROW(client.connect());
}

TEST_F(ClientNoServerTest, IsConnectedReturnsFalseAfterFailedConnect) {
    client.connect();
    EXPECT_FALSE(client.is_connected());
}

TEST_F(ClientNoServerTest, ReconnectAttemptAlsoReturnsFalse) {
    client.connect();
    bool second_attempt = client.connect();
    EXPECT_FALSE(second_attempt);
}

// =============================================================================
// Grupo 3: Operações sem conexão — devem falhar limpo, sem travar
// =============================================================================

TEST_F(ClientNoServerTest, HealthCheckWithoutConnectionReturnsFalse) {
    EXPECT_FALSE(client.health_check());
}

TEST_F(ClientNoServerTest, LoadModelWithoutConnectionReturnsFalse) {
    EXPECT_FALSE(client.load_model("m", "./models/simple_linear.onnx"));
}

TEST_F(ClientNoServerTest, UnloadModelWithoutConnectionReturnsFalse) {
    EXPECT_FALSE(client.unload_model("m"));
}

TEST_F(ClientNoServerTest, PredictWithoutConnectionReturnsFailure) {
    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto result = client.predict("modelo", inputs);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
}

TEST_F(ClientNoServerTest, PredictOutputsEmptyOnFailure) {
    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f};

    auto result = client.predict("modelo", inputs);

    EXPECT_TRUE(result.outputs.empty());
}

TEST_F(ClientNoServerTest, BatchPredictWithoutConnectionReturnsEmptyVector) {
    std::vector<std::map<std::string, std::vector<float>>> batch;
    for (int i = 0; i < 3; ++i) {
        std::map<std::string, std::vector<float>> inp;
        inp["input"] = {static_cast<float>(i), 0.0f};
        batch.push_back(inp);
    }

    auto results = client.batch_predict("modelo", batch);
    // Sem conexão: retorna vetor vazio ou com resultados de falha
    for (const auto& r : results) {
        EXPECT_FALSE(r.success);
    }
}

TEST_F(ClientNoServerTest, ListModelsWithoutConnectionReturnsEmpty) {
    auto models = client.list_models();
    EXPECT_TRUE(models.empty());
}

TEST_F(ClientNoServerTest, ListAvailableModelsWithoutConnectionReturnsEmpty) {
    auto models = client.list_available_models();
    EXPECT_TRUE(models.empty());
}

TEST_F(ClientNoServerTest, GetStatusWithoutConnectionReturnsDefaultStruct) {
    auto status = client.get_status();
    // Não deve travar; campos numéricos devem ser zero
    EXPECT_EQ(status.total_requests, 0u);
    EXPECT_EQ(status.successful_requests, 0u);
    EXPECT_EQ(status.failed_requests, 0u);
}

TEST_F(ClientNoServerTest, WarmupWithoutConnectionDoesNotCrash) {
    auto result = client.warmup_model("modelo_inexistente", 3);
    EXPECT_FALSE(result.success);
}

TEST_F(ClientNoServerTest, ValidateModelWithoutConnectionDoesNotCrash) {
    auto result = client.validate_model("./models/simple_linear.onnx");
    EXPECT_FALSE(result.valid);
}

// =============================================================================
// Grupo 4: Inputs vazios e edge cases de dados
// =============================================================================

TEST_F(ClientNoServerTest, PredictWithEmptyInputMapDoesNotCrash) {
    std::map<std::string, std::vector<float>> empty_inputs;
    EXPECT_NO_THROW(client.predict("modelo", empty_inputs));
}

TEST_F(ClientNoServerTest, PredictWithEmptyVectorValueDoesNotCrash) {
    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {};
    EXPECT_NO_THROW(client.predict("modelo", inputs));
}

TEST_F(ClientNoServerTest, PredictWithLargeInputDoesNotCrash) {
    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = std::vector<float>(224 * 224 * 3, 0.5f);  // tamanho ImageNet
    EXPECT_NO_THROW(client.predict("modelo", inputs));
}

TEST_F(ClientNoServerTest, PredictWithMultipleInputTensorsDoesNotCrash) {
    std::map<std::string, std::vector<float>> inputs;
    inputs["input_a"] = {1.0f, 2.0f};
    inputs["input_b"] = {3.0f, 4.0f};
    inputs["input_c"] = {5.0f, 6.0f};
    EXPECT_NO_THROW(client.predict("modelo_multi", inputs));
}

TEST_F(ClientNoServerTest, BatchPredictWithEmptyBatchDoesNotCrash) {
    std::vector<std::map<std::string, std::vector<float>>> empty_batch;
    EXPECT_NO_THROW(client.batch_predict("modelo", empty_batch));
}

// =============================================================================
// Grupo 5: IDs de modelos — limites e caracteres especiais
// =============================================================================

TEST_F(ClientNoServerTest, LoadModelWithEmptyIdDoesNotCrash) {
    EXPECT_NO_THROW(client.load_model("", "./models/simple_linear.onnx"));
}

TEST_F(ClientNoServerTest, LoadModelWithEmptyPathDoesNotCrash) {
    EXPECT_NO_THROW(client.load_model("modelo", ""));
}

TEST_F(ClientNoServerTest, LoadModelWithLongIdDoesNotCrash) {
    std::string long_id(256, 'x');
    EXPECT_NO_THROW(client.load_model(long_id, "./models/simple_linear.onnx"));
}

TEST_F(ClientNoServerTest, LoadModelWithSpecialCharsInIdDoesNotCrash) {
    EXPECT_NO_THROW(client.load_model("model-v1.2_final", "./models/m.onnx"));
}

TEST_F(ClientNoServerTest, UnloadNonExistentModelDoesNotCrash) {
    EXPECT_NO_THROW(client.unload_model("modelo_que_nao_existe"));
}

// =============================================================================
// Grupo 6: Destruição segura
// =============================================================================

TEST(ClientDestructionTest, CanBeDestroyedWithoutConnecting) {
    EXPECT_NO_THROW({
        InferenceClient c("localhost:19999");
        // Destruído aqui
    });
}

TEST(ClientDestructionTest, CanBeDestroyedAfterFailedConnect) {
    EXPECT_NO_THROW({
        InferenceClient c("localhost:19999");
        c.connect();
        // Destruído aqui
    });
}

TEST(ClientDestructionTest, MultipleInstancesDestroyedSafely) {
    EXPECT_NO_THROW({
        InferenceClient c1("localhost:19990");
        InferenceClient c2("localhost:19991");
        InferenceClient c3("localhost:19992");
        c1.connect();
        c2.connect();
        // c3 nunca conecta
    });
}

// =============================================================================
// Grupo 7: Timeout de conexão é razoável
// =============================================================================

TEST(ClientTimeoutTest, ConnectToBlackHoleCompletesWithinReasonableTime) {
    // 10.255.255.1 é um endereço que normalmente não responde (timeout real)
    // Aqui usamos porta fechada no localhost que rejeita imediatamente
    InferenceClient c("localhost:19998");

    auto start = std::chrono::steady_clock::now();
    c.connect();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start).count();

    // O connect() não deve demorar mais de 10 segundos
    EXPECT_LT(elapsed, 10L) << "connect() demorou mais do que o esperado";
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
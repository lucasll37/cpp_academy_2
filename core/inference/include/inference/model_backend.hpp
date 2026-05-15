// =============================================================================
/// @file   model_backend.hpp
/// @brief  Interface abstrata de backend de inferência e tipos de dados internos.
///
/// @details
/// Este header define três camadas do sistema de inferência:
///
/// 1. **Tipos de dados internos** — #InferenceResult, #TensorSpecData,
///    #ModelSchema e #RuntimeMetrics: estruturas usadas entre o motor
///    (#InferenceEngine) e os backends concretos.
///
/// 2. **Interface de backend** — #ModelBackend: classe puramente abstrata que
///    todo backend (ONNX, Python, ...) deve implementar.  Instâncias **não**
///    são thread-safe; a serialização é responsabilidade do #InferenceEngine.
///
/// 3. **Fábrica de backends** — #BackendFactory: interface de criação usada
///    pelo #BackendRegistry para instanciar backends por extensão de arquivo
///    ou por tipo enum.
///
/// ### Adicionando um novo backend
/// Para suportar um novo formato de modelo basta:
/// 1. Implementar #ModelBackend e #BackendFactory para o novo formato.
/// 2. Registrar no #BackendRegistry:
/// @code
/// BackendRegistry::instance().register_backend(
///     ".meu_ext", std::make_unique<MeuBackendFactory>());
/// @endcode
/// Nenhuma outra alteração no motor ou no cliente é necessária.
///
/// @see miia::inference::InferenceEngine
/// @see miia::inference::BackendRegistry
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_MODEL_BACKEND_HPP
#define ML_INFERENCE_MODEL_BACKEND_HPP

#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.pb.h"
#include "client/inference_client.hpp"  // client::Object, Value, Array

namespace miia {
namespace inference {

// =============================================================================
// Tipos de dados internos
// =============================================================================

// -----------------------------------------------------------------------------
// InferenceResult
// -----------------------------------------------------------------------------

/// @brief Resultado de uma única chamada de inferência no backend.
///
/// @details
/// Produzido por `ModelBackend::predict()` e repassado pelo #InferenceEngine
/// até o #InProcessBackend ou #GrpcClientBackend, onde é convertido para o
/// tipo público #miia::client::PredictionResult.
struct InferenceResult {
    /// @brief @c true se a inferência foi concluída sem erros.
    bool success = false;

    /// @brief Mapa de saídas do modelo (nome → #client::Value).
    ///
    /// Válido apenas quando #success é @c true.
    /// A estrutura espelha o formato JSON-like do #client::Object:
    /// escalares, arrays e objetos aninhados são suportados.
    client::Object outputs;

    /// @brief Tempo de execução da inferência, em milissegundos.
    ///
    /// Medido com `std::chrono::high_resolution_clock` internamente em cada
    /// backend.  Inclui apenas o tempo de computação — não inclui serialização
    /// ou transporte gRPC.
    double inference_time_ms = 0.0;

    /// @brief Mensagem de erro descritiva quando #success é @c false.
    std::string error_message;
};

// -----------------------------------------------------------------------------
// TensorSpecData
// -----------------------------------------------------------------------------

/// @brief Especificação de uma porta de tensor (entrada ou saída) de um modelo.
///
/// @details
/// Populada por `ModelBackend::get_schema()` após a carga do modelo.
/// Usada pelo #InferenceEngine para gerar dados sintéticos de warmup e para
/// construir o protobuf `common::ModelInfo` retornado ao cliente.
struct TensorSpecData {
    /// @brief Nome do tensor — deve corresponder às chaves do #client::Object.
    std::string name;

    /// @brief Shape do tensor; use @c -1 para dimensões dinâmicas.
    ///
    /// Exemplos:
    /// - `[1, 3]` — batch de 1, 3 features
    /// - `[-1, 128]` — batch dinâmico, 128 features
    /// - `[-1]` — vetor de tamanho arbitrário
    std::vector<int64_t> shape;

    /// @brief Tipo de dado do tensor (padrão: `FLOAT32`).
    common::DataType dtype = common::FLOAT32;

    /// @brief Descrição textual da porta (opcional).
    std::string description;

    /// @brief Valor mínimo esperado (usado quando #has_constraints é @c true).
    double min_value = 0.0;

    /// @brief Valor máximo esperado (usado quando #has_constraints é @c true).
    double max_value = 0.0;

    /// @brief @c true se #min_value e #max_value definem restrições válidas.
    bool has_constraints = false;

    /// @brief @c true se a entrada é um dict/list Python aninhado arbitrário.
    ///
    /// Quando @c true, o warmup injeta um #client::Object vazio em vez de
    /// gerar um array aleatório, pois a estrutura exata depende do modelo.
    /// Corresponde ao campo `structured` do `TensorSpec` Python.
    bool structured = false;
};

// -----------------------------------------------------------------------------
// ModelSchema
// -----------------------------------------------------------------------------

/// @brief Schema completo de I/O retornado por um backend após a carga.
///
/// @details
/// Obtido via `ModelBackend::get_schema()`.  Para modelos Python, é extraído
/// do retorno de `model.get_schema()` via CPython.  Para modelos ONNX, é
/// extraído das sessões ONNX Runtime durante `load()`.
struct ModelSchema {
    /// @brief Especificações dos tensores de entrada, em ordem de declaração.
    std::vector<TensorSpecData> inputs;

    /// @brief Especificações dos tensores de saída, em ordem de declaração.
    std::vector<TensorSpecData> outputs;

    /// @brief Descrição textual do modelo (fornecida por `get_schema()`).
    std::string description;

    /// @brief Autor do modelo (fornecido por `get_schema()`).
    std::string author;

    /// @brief Metadados arbitrários em formato chave-valor.
    ///
    /// Exemplos: `{"type": "navigation", "algorithm": "potential_field"}`.
    std::map<std::string, std::string> tags;
};

// -----------------------------------------------------------------------------
// RuntimeMetrics
// -----------------------------------------------------------------------------

/// @brief Métricas de runtime acumuladas por instância de backend.
///
/// @details
/// Mantida dentro de cada #ModelBackend e atualizada via `record()` a cada
/// chamada de `predict()`.  Lida pelo #InferenceEngine via
/// `ModelBackend::metrics()` e serializada no protobuf `GetMetricsResponse`.
///
/// ### Decisões de design
///
/// **`min_time_ms` inicializado em `0.0`** (não em `+∞`): evita que o valor
/// inicial apareça como número muito grande ou negativo após conversão
/// `double → float` no protobuf ou corrupção de bit de sinal.  O método
/// `record()` trata a primeira amostra explicitamente para inicializar
/// `min_time_ms` e `max_time_ms` corretamente.
///
/// **Buffer circular de latência** (`latency_samples`): armazena as últimas
/// `LATENCY_WINDOW` amostras bem-sucedidas para cálculo de p95/p99 via
/// nearest-rank.  Amostras mais antigas são sobrescritas em modo circular,
/// mantendo uso de memória constante independente do número de inferências.
///
/// **Medidas negativas**: clock skew ou wrap de timer podem produzir valores
/// negativos; `record()` clampa para `0.0` antes de acumular.
struct RuntimeMetrics {
    /// @brief Total de chamadas a `predict()`, incluindo falhas.
    uint64_t total_inferences  = 0;

    /// @brief Chamadas a `predict()` que retornaram `success = false`.
    uint64_t failed_inferences = 0;

    /// @brief Soma dos tempos de todas as inferências bem-sucedidas, em ms.
    double total_time_ms = 0.0;

    /// @brief Menor latência observada entre inferências bem-sucedidas, em ms.
    ///
    /// @note Inicializado em `0.0` ("não observado ainda") — não em `+∞`.
    ///       Válido apenas quando `total_inferences - failed_inferences > 0`.
    double min_time_ms = 0.0;

    /// @brief Maior latência observada entre inferências bem-sucedidas, em ms.
    double max_time_ms = 0.0;

    /// @brief Capacidade do buffer circular de amostras de latência.
    static constexpr size_t LATENCY_WINDOW = 1000;

    /// @brief Buffer circular com as últimas #LATENCY_WINDOW amostras bem-sucedidas.
    ///
    /// Alimenta os cálculos de p95 e p99 em `p95_time_ms()` e `p99_time_ms()`.
    std::vector<double> latency_samples;

    /// @brief Posição de escrita atual no buffer circular.
    size_t latency_write_pos = 0;

    /// @brief Registra o resultado de uma inferência nas métricas acumuladas.
    ///
    /// @details
    /// - Incrementa `total_inferences` sempre.
    /// - Se `success` é @c false, incrementa apenas `failed_inferences` e retorna.
    /// - Clampa `time_ms < 0` para `0.0` antes de acumular.
    /// - Na primeira amostra bem-sucedida, inicializa `min_time_ms` e `max_time_ms`.
    /// - Insere `time_ms` no buffer circular `latency_samples`.
    ///
    /// @param time_ms  Duração da inferência em milissegundos.
    /// @param success  @c true se a inferência produziu outputs válidos.
    void record(double time_ms, bool success) {
        total_inferences++;
        if (!success) {
            failed_inferences++;
            return;
        }

        // Clamp: medida negativa indica clock skew / wrap — tratar como zero.
        if (time_ms < 0.0) time_ms = 0.0;

        total_time_ms += time_ms;

        uint64_t ok = total_inferences - failed_inferences;
        if (ok == 1) {
            // Primeira amostra bem-sucedida — inicializa min/max.
            min_time_ms = time_ms;
            max_time_ms = time_ms;
        } else {
            if (time_ms < min_time_ms) min_time_ms = time_ms;
            if (time_ms > max_time_ms) max_time_ms = time_ms;
        }

        if (latency_samples.size() < LATENCY_WINDOW) {
            latency_samples.push_back(time_ms);
        } else {
            latency_samples[latency_write_pos % LATENCY_WINDOW] = time_ms;
        }
        latency_write_pos++;
    }

    /// @brief Retorna a latência média de inferências bem-sucedidas, em ms.
    ///
    /// @return `total_time_ms / (total - failed)`, ou `0.0` se não há amostras.
    double avg_time_ms() const {
        uint64_t ok = total_inferences - failed_inferences;
        return (ok > 0) ? total_time_ms / static_cast<double>(ok) : 0.0;
    }

    /// @brief Retorna o percentil 95 da latência (nearest-rank sobre o buffer).
    ///
    /// @details
    /// Ordena uma cópia de `latency_samples` a cada chamada — O(N log N) onde
    /// N ≤ `LATENCY_WINDOW`.  Não armazena estado adicional.
    ///
    /// @return P95 em milissegundos, ou `0.0` se o buffer estiver vazio.
    double p95_time_ms() const;

    /// @brief Retorna o percentil 99 da latência (nearest-rank sobre o buffer).
    ///
    /// @return P99 em milissegundos, ou `0.0` se o buffer estiver vazio.
    double p99_time_ms() const;
};

// =============================================================================
// ModelBackend — Interface abstrata
// =============================================================================

/// @brief Interface que todo backend de inferência deve implementar.
///
/// @details
/// Cada instância gerencia **um** modelo carregado.  O ciclo de vida é:
///
/// ```
/// load() → predict() [N vezes] → unload()
/// ```
///
/// Instâncias **não** são thread-safe — o #InferenceEngine serializa o acesso
/// via `mutex_` antes de despachar para o backend.
///
/// ### Implementações concretas
/// - **#PythonBackend** — modelos `.py` via CPython embutido.
/// - **#OnnxBackend** — modelos `.onnx` via ONNX Runtime.
///
/// ### Adicionando um novo backend
/// Implemente esta interface + #BackendFactory e registre no #BackendRegistry:
/// @code
/// BackendRegistry::instance().register_backend(
///     ".meu_ext", std::make_unique<MeuBackendFactory>());
/// @endcode
class ModelBackend {
public:
    virtual ~ModelBackend() = default;

    // -------------------------------------------------------------------------
    /// @name Ciclo de vida
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Carrega o modelo a partir de um arquivo.
    ///
    /// @param path    Caminho absoluto do arquivo do modelo.
    /// @param config  Parâmetros opcionais específicos do backend (chave-valor).
    ///
    /// @return @c true se o modelo foi carregado e está pronto para inferência.
    virtual bool load(const std::string& path,
                      const std::map<std::string, std::string>& config) = 0;

    /// @brief Libera todos os recursos alocados pelo modelo.
    ///
    /// @details
    /// Após `unload()`, a instância volta ao estado não-carregado.
    /// `load()` pode ser chamado novamente.
    virtual void unload() = 0;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Inferência
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Executa inferência sobre um conjunto de inputs.
    ///
    /// @details
    /// Os inputs chegam como #client::Object (mapa nome → #client::Value),
    /// suportando escalares, arrays e objetos aninhados arbitrariamente.
    /// Cada backend é responsável por interpretar e converter a estrutura:
    ///
    /// - **OnnxBackend:** extrai arrays de primeiro nível como tensores
    ///   `float` e os passa ao ONNX Runtime.
    /// - **PythonBackend:** converte o `Object` para `dict` Python e chama
    ///   `model.predict(inputs)`.
    ///
    /// @pre `load()` deve ter sido chamado com sucesso (`loaded_ == true`).
    ///
    /// @param inputs  Mapa de entradas do modelo.
    ///
    /// @return #InferenceResult com outputs e tempo de execução.
    virtual InferenceResult predict(const client::Object& inputs) = 0;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Introspecção
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Retorna o schema de I/O do modelo.
    ///
    /// @pre Disponível após `load()`.
    ///
    /// @return #ModelSchema com tensores de entrada, saída e metadados.
    virtual ModelSchema get_schema() const = 0;

    /// @brief Retorna o tipo de backend como enum protobuf.
    ///
    /// @return `common::BACKEND_PYTHON`, `common::BACKEND_ONNX`, etc.
    virtual common::BackendType backend_type() const = 0;

    /// @brief Estima o uso de memória do modelo em bytes.
    ///
    /// @details
    /// Implementação default retorna `0` (não disponível).
    /// Backends que conhecem o tamanho dos pesos devem sobrescrever.
    ///
    /// @return Uso estimado em bytes, ou `0` se não disponível.
    virtual int64_t memory_usage_bytes() const { return 0; }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Aquecimento
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Executa @p n inferências sintéticas para aquecer o backend.
    ///
    /// @details
    /// A implementação default gera inputs aleatórios (semente `mt19937(42)`)
    /// com base no #ModelSchema retornado por `get_schema()`, usando
    /// `std::uniform_real_distribution<float>(0.0, 1.0)` para cada elemento.
    ///
    /// Backends com warmup específico (ex.: pré-compilação JIT do ONNX Runtime)
    /// devem sobrescrever este método.
    ///
    /// @param n  Número de inferências de aquecimento.
    virtual void warmup(uint32_t n);

    /// @}

    // -------------------------------------------------------------------------
    /// @name Validação estática
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Verifica se @p path parece um arquivo válido para este backend.
    ///
    /// @details
    /// Validação estática: não carrega o modelo na memória de inferência.
    /// A implementação default não faz nenhuma verificação (aceita tudo).
    /// Backends devem sobrescrever para verificar extensão e existência.
    ///
    /// @param path  Caminho do arquivo a verificar.
    ///
    /// @return String vazia se válido; mensagem de erro caso contrário.
    virtual std::string validate(const std::string& path) const {
        (void)path;
        return "";
    }

    /// @}

    // -------------------------------------------------------------------------
    /// @name Métricas e temporização
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Retorna referência const para as métricas acumuladas do backend.
    ///
    /// @return #RuntimeMetrics com contadores, latências e buffer de percentis.
    const RuntimeMetrics& metrics() const { return metrics_; }

    /// @brief Retorna o timestamp de quando `load()` foi chamado com sucesso.
    std::chrono::steady_clock::time_point load_time() const { return load_time_; }

    /// @brief Retorna o timestamp da última chamada a `predict()`.
    std::chrono::steady_clock::time_point last_used() const { return last_used_; }

    /// @}

protected:
    /// @cond INTERNAL

    /// @brief Timestamp de carga — definido em `load()` pelos backends concretos.
    std::chrono::steady_clock::time_point load_time_;

    /// @brief Timestamp da última inferência — atualizado via `touch()`.
    std::chrono::steady_clock::time_point last_used_;

    /// @brief @c true após `load()` bem-sucedido; @c false após `unload()`.
    bool loaded_ = false;

    /// @brief Atualiza #last_used_ para o instante atual.
    ///
    /// Deve ser chamado por backends concretos no início de `predict()`.
    void touch() { last_used_ = std::chrono::steady_clock::now(); }

    /// @brief Métricas acumuladas desta instância de backend.
    RuntimeMetrics metrics_;

    /// @endcond
};

// =============================================================================
// BackendFactory — Interface de fábrica
// =============================================================================

/// @brief Interface de fábrica para criação de instâncias de #ModelBackend.
///
/// @details
/// Implementada por cada backend concreto (ex.: `PythonBackendFactory`,
/// `OnnxBackendFactory`) e registrada no #BackendRegistry associada a uma
/// extensão de arquivo.  O padrão *Factory Method* permite que o
/// #BackendRegistry crie backends por extensão sem depender de tipos concretos.
///
/// ### Implementação mínima
/// @code
/// class MeuBackendFactory : public BackendFactory {
/// public:
///     std::unique_ptr<ModelBackend> create() const override {
///         return std::make_unique<MeuBackend>();
///     }
///     common::BackendType backend_type() const override {
///         return common::BACKEND_MEU_TIPO;
///     }
///     std::string name() const override { return "meu_backend"; }
/// };
/// @endcode
class BackendFactory {
public:
    virtual ~BackendFactory() = default;

    /// @brief Cria e retorna uma nova instância do backend associado.
    ///
    /// @return Instância não-carregada do backend (estado inicial).
    virtual std::unique_ptr<ModelBackend> create() const = 0;

    /// @brief Retorna o tipo enum protobuf do backend produzido.
    virtual common::BackendType backend_type() const = 0;

    /// @brief Retorna o nome textual do backend (ex.: `"python"`, `"onnx"`).
    virtual std::string name() const = 0;
};

}  // namespace inference
}  // namespace miia

#endif  // ML_INFERENCE_MODEL_BACKEND_HPP
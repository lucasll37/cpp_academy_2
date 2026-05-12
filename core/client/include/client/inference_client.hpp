// =============================================================================
/// @file   inference_client.hpp
/// @brief  API pública do cliente MiiaClient para inferência de modelos de ML.
///
/// @details
/// Este header define todos os tipos e a classe principal necessários para
/// interagir com o servidor MiiaServer.  Dois modos de operação são suportados,
/// selecionados automaticamente pela string passada ao construtor de
/// #InferenceClient:
///
/// **Modo gRPC** — comunicação via rede com um servidor remoto:
/// @code
/// mlinference::client::InferenceClient client("localhost:50052");
/// @endcode
///
/// **Modo in-process** — servidor embutido no mesmo processo, sem rede:
/// @code
/// mlinference::client::InferenceClient client("inprocess");
/// @endcode
///
/// Strings reconhecidas como in-process: `"inprocess"`, `"in_process"`,
/// `"local"`.  Qualquer outra string é tratada como endereço gRPC
/// (`"host:porta"`).
///
/// ### Formato de inputs e outputs
///
/// Todos os campos de inferência usam o tipo #Value, que pode representar
/// qualquer estrutura arbitrária e aninhada, espelhando `google.protobuf.Value`:
///
/// @code
/// using namespace mlinference::client;
///
/// Object inputs;
/// inputs["speed"]    = Value{12.5};
/// inputs["position"] = Value{Object{{"x", Value{1.0}}, {"y", Value{2.0}}}};
/// inputs["history"]  = Value{Array{Value{1.0}, Value{2.0}, Value{3.0}}};
/// inputs["mode"]     = Value{"combat"};
/// inputs["active"]   = Value{true};
///
/// PredictionResult r = client.predict("nav_model", inputs);
/// if (r.success)
///     double heading = r.outputs["heading"].as_number();
/// @endcode
///
/// O campo `PredictionResult::outputs` segue o mesmo formato.
///
/// @author  Lucas
/// @date    2026
// =============================================================================

#ifndef ML_INFERENCE_CLIENT_HPP
#define ML_INFERENCE_CLIENT_HPP

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace mlinference {

/// @namespace mlinference::client
/// @brief     Contém todos os tipos e classes da API pública do cliente MiiaClient.
namespace client {

// =============================================================================
// Tipo de valor dinâmico
// =============================================================================

struct Value;

/// @brief Representa ausência de valor (equivalente a JSON @c null).
///
/// É o tipo armazenado por um #Value construído com o construtor default.
using Null = std::monostate;

/// @brief Mapeamento nome → #Value (equivalente a JSON @em object).
///
/// É o tipo dos inputs e outputs de inferência:
/// @code
/// Object inputs;
/// inputs["x"] = Value{1.0};
/// inputs["y"] = Value{2.0};
/// @endcode
using Object = std::map<std::string, Value>;

/// @brief Sequência ordenada de #Value (equivalente a JSON @em array).
///
/// @code
/// Array history{Value{1.0}, Value{2.0}, Value{3.0}};
/// inputs["history"] = Value{std::move(history)};
/// @endcode
using Array = std::vector<Value>;

// =============================================================================
// Value
// =============================================================================

/// @brief Valor dinâmico aninhável que espelha `google.protobuf.Value`.
///
/// @details
/// Pode armazenar exatamente um dos seguintes tipos:
///
/// | Tipo C++        | Equivalente JSON | Verificador     | Acessor         |
/// |-----------------|------------------|-----------------|-----------------|
/// | #Null           | `null`           | is_null()       | —               |
/// | `double`        | `number`         | is_number()     | as_number()     |
/// | `bool`          | `boolean`        | is_bool()       | as_bool()       |
/// | `std::string`   | `string`         | is_string()     | as_string()     |
/// | #Array          | `array`          | is_array()      | as_array()      |
/// | #Object         | `object`         | is_object()     | as_object()     |
///
/// ### Conversões inteiras
/// Para inteiros use o construtor explícito de `int`, `int64_t`, ou o cast
/// manual:
/// @code
/// Value v{42};              // int → double internamente
/// Value v{int64_t{1000}};
/// Value v{static_cast<double>(n)};
/// @endcode
///
/// ### Aninhamento
/// @code
/// Value nested{Object{
///     {"lat",  Value{-23.5}},
///     {"lon",  Value{-46.6}},
///     {"tags", Value{Array{Value{"SP"}, Value{"BR"}}}},
/// }};
/// @endcode
struct Value {
    /// @brief Variante interna — armazena exatamente um tipo em tempo de execução.
    std::variant<Null, double, bool, std::string, Array, Object> data;

    // -------------------------------------------------------------------------
    // Construtores
    // -------------------------------------------------------------------------

    /// @brief Constrói um valor nulo (#Null).
    Value() : data(Null{}) {}

    /// @brief Constrói a partir de um valor `double`.
    explicit Value(double v)      : data(v) {}

    /// @brief Constrói a partir de um valor `float` (promovido para `double`).
    explicit Value(float v)       : data(static_cast<double>(v)) {}

    /// @brief Constrói a partir de um valor `int` (promovido para `double`).
    explicit Value(int v)         : data(static_cast<double>(v)) {}

    /// @brief Constrói a partir de um valor `int64_t` (promovido para `double`).
    explicit Value(int64_t v)     : data(static_cast<double>(v)) {}

    /// @brief Constrói a partir de um valor `bool`.
    explicit Value(bool v)        : data(v) {}

    /// @brief Constrói a partir de uma `std::string` (move).
    explicit Value(std::string v) : data(std::move(v)) {}

    /// @brief Constrói a partir de um literal de string C.
    explicit Value(const char* v) : data(std::string(v)) {}

    /// @brief Constrói a partir de um #Array (move).
    explicit Value(Array v)       : data(std::move(v)) {}

    /// @brief Constrói a partir de um #Object (move).
    explicit Value(Object v)      : data(std::move(v)) {}

    // -------------------------------------------------------------------------
    // Verificadores de tipo
    // -------------------------------------------------------------------------

    /// @brief Retorna @c true se o valor armazenado é #Null.
    bool is_null()   const { return std::holds_alternative<Null>(data);        }

    /// @brief Retorna @c true se o valor armazenado é um número (`double`).
    bool is_number() const { return std::holds_alternative<double>(data);      }

    /// @brief Retorna @c true se o valor armazenado é `bool`.
    bool is_bool()   const { return std::holds_alternative<bool>(data);        }

    /// @brief Retorna @c true se o valor armazenado é `std::string`.
    bool is_string() const { return std::holds_alternative<std::string>(data); }

    /// @brief Retorna @c true se o valor armazenado é um #Array.
    bool is_array()  const { return std::holds_alternative<Array>(data);       }

    /// @brief Retorna @c true se o valor armazenado é um #Object.
    bool is_object() const { return std::holds_alternative<Object>(data);      }

    // -------------------------------------------------------------------------
    // Acessores (const)
    // -------------------------------------------------------------------------

    /// @brief Retorna o valor numérico armazenado.
    /// @pre   is_number() == true.
    /// @throws std::bad_variant_access se o tipo armazenado não for `double`.
    double             as_number() const { return std::get<double>(data);      }

    /// @brief Retorna o valor booleano armazenado.
    /// @pre   is_bool() == true.
    /// @throws std::bad_variant_access se o tipo armazenado não for `bool`.
    bool               as_bool()   const { return std::get<bool>(data);        }

    /// @brief Retorna referência const para a string armazenada.
    /// @pre   is_string() == true.
    /// @throws std::bad_variant_access se o tipo armazenado não for `std::string`.
    const std::string& as_string() const { return std::get<std::string>(data); }

    /// @brief Retorna referência const para o #Array armazenado.
    /// @pre   is_array() == true.
    /// @throws std::bad_variant_access se o tipo armazenado não for #Array.
    const Array&       as_array()  const { return std::get<Array>(data);       }

    /// @brief Retorna referência const para o #Object armazenado.
    /// @pre   is_object() == true.
    /// @throws std::bad_variant_access se o tipo armazenado não for #Object.
    const Object&      as_object() const { return std::get<Object>(data);      }

    // -------------------------------------------------------------------------
    // Acessores (não-const — para modificação in-place)
    // -------------------------------------------------------------------------

    /// @brief Retorna referência mutável para o #Array armazenado.
    /// @pre   is_array() == true.
    /// @throws std::bad_variant_access se o tipo armazenado não for #Array.
    Array&  as_array()  { return std::get<Array>(data);  }

    /// @brief Retorna referência mutável para o #Object armazenado.
    /// @pre   is_object() == true.
    /// @throws std::bad_variant_access se o tipo armazenado não for #Object.
    Object& as_object() { return std::get<Object>(data); }
};

// =============================================================================
// Structs de retorno públicos
// =============================================================================

/// @brief Resultado de uma chamada a #InferenceClient::predict().
///
/// @details
/// Verifique sempre #success antes de acessar #outputs:
/// @code
/// auto r = client.predict("my_model", inputs);
/// if (!r.success) {
///     asalog::err("agent") << "Inferência falhou: " << r.error_message;
///     return;
/// }
/// double heading = r.outputs["heading"].as_number();
/// @endcode
struct PredictionResult {
    /// @brief Indica se a inferência foi concluída com sucesso.
    bool success = false;

    /// @brief Mapa de saídas do modelo (nome → #Value).
    ///
    /// Válido apenas quando #success é @c true.
    Object outputs;

    /// @brief Tempo total de inferência no servidor, em milissegundos.
    double inference_time_ms = 0.0;

    /// @brief Mensagem de erro descritiva.
    ///
    /// Preenchida quando #success é @c false.
    std::string error_message;
};

/// @brief Metadados de um modelo carregado no servidor.
struct ModelInfo {
    /// @brief Identificador único do modelo (passado em load_model()).
    std::string model_id;

    /// @brief Versão do modelo (ex.: @c "1.0.0").
    std::string version;

    /// @brief Backend utilizado (ex.: @c "python", @c "onnx").
    std::string backend;

    /// @brief Descrição textual do modelo, fornecida por `get_schema()`.
    std::string description;

    /// @brief Autor do modelo, fornecido por `get_schema()`.
    std::string author;

    /// @brief Uso estimado de memória, em bytes.
    uint64_t memory_usage_bytes = 0;

    /// @brief Indica se o modelo passou por aquecimento prévio (warmup).
    bool is_warmed_up = false;

    /// @brief Especificação de um tensor de entrada ou saída.
    struct TensorSpec {
        /// @brief Nome do tensor (deve corresponder às chaves de #Object).
        std::string name;

        /// @brief Tipo de dado (ex.: @c "float32", @c "int64").
        std::string dtype;

        /// @brief Shape do tensor; use @c -1 para dimensões dinâmicas.
        std::vector<int64_t> shape;

        /// @brief Descrição textual da porta.
        std::string description;

        /// @brief @c true se a entrada é um dict/list aninhado (modelos Python).
        ///
        /// Quando @c true, o campo aceita #Object ou #Array arbitrários em vez
        /// de um tensor numérico plano.
        bool structured = false;
    };

    /// @brief Especificações dos tensores de entrada do modelo.
    std::vector<TensorSpec> inputs;

    /// @brief Especificações dos tensores de saída do modelo.
    std::vector<TensorSpec> outputs;

    /// @brief Metadados arbitrários em formato chave-valor.
    std::map<std::string, std::string> tags;

    /// @brief Timestamp Unix de quando o modelo foi carregado.
    int64_t loaded_at_unix = 0;
};

/// @brief Resultado da validação estática de um arquivo de modelo.
///
/// Retornado por #InferenceClient::validate_model().  A validação é feita sem
/// carregar o modelo na memória de inferência.
struct ValidationResult {
    /// @brief @c true se o arquivo é válido e pode ser carregado.
    bool valid = false;

    /// @brief Backend detectado para o arquivo (ex.: @c "python", @c "onnx").
    std::string backend;

    /// @brief Mensagem de erro quando #valid é @c false.
    std::string error_message;

    /// @brief Avisos não-fatais (ex.: ausência de `get_schema()`).
    std::vector<std::string> warnings;

    /// @brief Tensores de entrada reportados pelo modelo (se disponíveis).
    std::vector<ModelInfo::TensorSpec> inputs;

    /// @brief Tensores de saída reportados pelo modelo (se disponíveis).
    std::vector<ModelInfo::TensorSpec> outputs;
};

/// @brief Resultado de uma operação de aquecimento (warmup) de modelo.
///
/// Retornado por #InferenceClient::warmup_model().
struct WarmupResult {
    /// @brief @c true se todas as execuções de aquecimento completaram.
    bool success = false;

    /// @brief Número de execuções efetivamente concluídas.
    uint32_t runs_completed = 0;

    /// @brief Tempo médio por inferência de aquecimento, em milissegundos.
    double avg_time_ms = 0.0;

    /// @brief Tempo mínimo observado, em milissegundos.
    double min_time_ms = 0.0;

    /// @brief Tempo máximo observado, em milissegundos.
    double max_time_ms = 0.0;

    /// @brief Mensagem de erro quando #success é @c false.
    std::string error_message;
};

/// @brief Estado operacional do worker no momento da consulta.
///
/// Retornado por #InferenceClient::get_status().
struct WorkerStatus {
    /// @brief Identificador textual do worker (configurável via `--worker-id`).
    std::string worker_id;

    /// @brief Total de requisições recebidas desde o início.
    uint64_t total_requests      = 0;

    /// @brief Requisições concluídas com sucesso.
    uint64_t successful_requests = 0;

    /// @brief Requisições que resultaram em erro.
    uint64_t failed_requests     = 0;

    /// @brief Requisições em andamento no momento da consulta.
    uint32_t active_requests     = 0;

    /// @brief Tempo em segundos desde a inicialização do servidor.
    int64_t  uptime_seconds      = 0;

    /// @brief Lista de IDs dos modelos atualmente carregados.
    std::vector<std::string> loaded_models;

    /// @brief Backends disponíveis no servidor (ex.: @c "python", @c "onnx").
    std::vector<std::string> supported_backends;
};

/// @brief Métricas de desempenho agregadas por modelo.
///
/// Parte de #ServerMetrics::per_model.
struct ModelMetrics {
    /// @brief Total de inferências realizadas com este modelo.
    uint64_t total_inferences  = 0;

    /// @brief Inferências que falharam.
    uint64_t failed_inferences = 0;

    /// @brief Tempo médio de inferência, em milissegundos.
    double avg_ms       = 0.0;

    /// @brief Tempo mínimo de inferência, em milissegundos.
    double min_ms       = 0.0;

    /// @brief Tempo máximo de inferência, em milissegundos.
    double max_ms       = 0.0;

    /// @brief Percentil 95 do tempo de inferência, em milissegundos.
    double p95_ms       = 0.0;

    /// @brief Percentil 99 do tempo de inferência, em milissegundos.
    double p99_ms       = 0.0;

    /// @brief Soma total dos tempos de inferência, em milissegundos.
    double total_time_ms = 0.0;

    /// @brief Timestamp Unix da última inferência realizada.
    int64_t last_used_at_unix = 0;

    /// @brief Timestamp Unix de quando o modelo foi carregado.
    int64_t loaded_at_unix    = 0;
};

/// @brief Métricas globais do servidor e por modelo.
///
/// Retornado por #InferenceClient::get_metrics().
struct ServerMetrics {
    /// @brief Total de requisições recebidas pelo servidor.
    uint64_t total_requests      = 0;

    /// @brief Requisições concluídas com sucesso.
    uint64_t successful_requests = 0;

    /// @brief Requisições que resultaram em erro.
    uint64_t failed_requests     = 0;

    /// @brief Requisições em andamento no momento da consulta.
    uint32_t active_requests     = 0;

    /// @brief Tempo em segundos desde a inicialização do servidor.
    int64_t  uptime_seconds      = 0;

    /// @brief Métricas individuais por modelo, indexadas pelo model_id.
    std::map<std::string, ModelMetrics> per_model;
};

/// @brief Descreve um arquivo de modelo encontrado em um diretório.
///
/// Retornado por #InferenceClient::list_available_models().
struct AvailableModel {
    /// @brief Nome do arquivo (sem caminho).
    std::string filename;

    /// @brief Caminho absoluto do arquivo.
    std::string path;

    /// @brief Extensão do arquivo (ex.: @c ".py", @c ".onnx").
    std::string extension;

    /// @brief Backend inferido para este arquivo (ex.: @c "python", @c "onnx").
    std::string backend;

    /// @brief Tamanho do arquivo em bytes.
    int64_t file_size_bytes = 0;

    /// @brief @c true se o modelo já está carregado no servidor.
    bool is_loaded = false;

    /// @brief model_id sob o qual o modelo está carregado (quando #is_loaded).
    std::string loaded_as;
};

// =============================================================================
// Detalhe de implementação (não documentado publicamente)
// =============================================================================

/// @cond INTERNAL
class IClientBackend;
/// @endcond

// =============================================================================
// InferenceClient
// =============================================================================

/// @brief Cliente principal do MiiaClient para execução de inferência de modelos ML.
///
/// @details
/// `InferenceClient` é o ponto de entrada para todas as operações do sistema:
/// carga de modelos, inferência, introspecção e observabilidade.  O modo de
/// transporte é determinado pela string passada ao construtor:
///
/// | String passada            | Modo         | Descrição                          |
/// |---------------------------|--------------|------------------------------------|
/// | `"inprocess"` / `"in_process"` / `"local"` | In-process | Servidor embutido, sem rede |
/// | qualquer outra string     | gRPC         | Conexão com servidor remoto        |
///
/// ### Uso típico (in-process)
/// @code
/// using namespace mlinference::client;
///
/// InferenceClient client("inprocess");
/// client.connect();
///
/// client.load_model("nav", "/app/models/ship_avoidance.py");
///
/// Object inputs;
/// inputs["state"] = Value{Object{
///     {"toHeading", Value{45.0}},
///     {"speed",     Value{10.0}},
/// }};
///
/// auto result = client.predict("nav", inputs);
/// if (result.success)
///     double heading = result.outputs["heading"].as_number();
/// @endcode
///
/// ### Uso típico (gRPC)
/// @code
/// InferenceClient client("localhost:50052");
/// client.connect();
/// client.load_model("nav", "/app/models/ship_avoidance.py");
/// auto result = client.predict("nav", inputs);
/// @endcode
///
/// @note A classe não é copiável.  Use `std::unique_ptr<InferenceClient>` ou
///       passe-a por referência entre componentes do simulador.
class InferenceClient {
public:
    /// @brief Constrói o cliente e seleciona o backend de transporte.
    ///
    /// @param target  Endereço gRPC (`"host:porta"`) ou string in-process
    ///                (`"inprocess"`, `"in_process"`, `"local"`).
    explicit InferenceClient(const std::string& target);

    /// @brief Destrói o cliente e libera o backend subjacente.
    ~InferenceClient();

    // -------------------------------------------------------------------------
    /// @name Conexão
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Estabelece a conexão com o backend.
    ///
    /// @details
    /// - **Modo in-process:** inicializa o servidor embutido (CPython, ONNX
    ///   Runtime) no mesmo processo.  Deve ser chamado uma vez antes de
    ///   qualquer outra operação.
    /// - **Modo gRPC:** abre o canal e verifica a acessibilidade do servidor.
    ///
    /// @return @c true em caso de sucesso; @c false se a conexão falhar.
    bool connect();

    /// @brief Verifica se o cliente está conectado ao backend.
    ///
    /// @return @c true se connect() foi chamado com sucesso e o canal está ativo.
    bool is_connected() const;

    /// @}

    // -------------------------------------------------------------------------
    /// @name Ciclo de vida dos modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Carrega um modelo no servidor a partir de um caminho local.
    ///
    /// @details
    /// O backend é detectado automaticamente pela extensão do arquivo:
    /// - `.py`   → Python backend (via CPython embutido)
    /// - `.onnx` → ONNX Runtime backend
    ///
    /// Um mesmo arquivo pode ser carregado com IDs diferentes:
    /// @code
    /// client.load_model("nav_a", "/models/nav.py");
    /// client.load_model("nav_b", "/models/nav.py");  // segunda instância
    /// @endcode
    ///
    /// @param model_id    Identificador único para referenciar o modelo nas
    ///                    chamadas subsequentes.
    /// @param model_path  Caminho absoluto ou relativo ao arquivo do modelo.
    /// @param version     Rótulo de versão (informativo, padrão `"1.0.0"`).
    ///
    /// @return @c true se o modelo foi carregado com sucesso.
    ///         @c false se o arquivo não existir, o ID já estiver em uso, ou
    ///         ocorrer erro de inicialização.
    bool load_model(const std::string& model_id,
                    const std::string& model_path,
                    const std::string& version = "1.0.0");

    /// @brief Remove um modelo da memória do servidor.
    ///
    /// @param model_id  ID do modelo previamente carregado via load_model().
    ///
    /// @return @c true se o modelo foi descarregado com sucesso.
    ///         @c false se o ID não existir ou ocorrer erro durante o unload.
    bool unload_model(const std::string& model_id);

    /// @}

    // -------------------------------------------------------------------------
    /// @name Inferência
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Executa inferência em um modelo carregado.
    ///
    /// @details
    /// Os inputs são passados como um #Object (mapa nome → #Value), permitindo
    /// estruturas arbitrariamente aninhadas.  O modelo Python recebe o #Object
    /// convertido para um `dict` Python; o ONNX Runtime recebe tensores planos
    /// extraídos dos #Array de primeiro nível.
    ///
    /// @param model_id  ID do modelo carregado com load_model().
    /// @param inputs    Mapa de entradas do modelo.
    ///
    /// @return #PredictionResult com os outputs e metadados da inferência.
    ///         Verifique PredictionResult::success antes de acessar os outputs.
    PredictionResult predict(const std::string& model_id,
                             const Object& inputs);

    /// @brief Executa inferência em lote sobre um conjunto de inputs.
    ///
    /// @details
    /// Cada elemento de @p batch_inputs é processado independentemente e
    /// produz um #PredictionResult correspondente na saída.  A ordem é
    /// preservada.
    ///
    /// @param model_id    ID do modelo carregado.
    /// @param batch_inputs  Vetor de objetos de entrada.
    ///
    /// @return Vetor de #PredictionResult na mesma ordem de @p batch_inputs.
    std::vector<PredictionResult> batch_predict(
        const std::string& model_id,
        const std::vector<Object>& batch_inputs);

    /// @}

    // -------------------------------------------------------------------------
    /// @name Introspecção
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Lista todos os modelos atualmente carregados no servidor.
    ///
    /// @return Vetor de #ModelInfo com metadados de cada modelo carregado.
    std::vector<ModelInfo> list_models();

    /// @brief Retorna os metadados de um modelo específico.
    ///
    /// @param model_id  ID do modelo carregado.
    ///
    /// @return #ModelInfo populado.  Se o modelo não existir, os campos
    ///         estarão com valores default (string vazia, zeros).
    ModelInfo get_model_info(const std::string& model_id);

    /// @brief Valida um arquivo de modelo sem carregá-lo.
    ///
    /// @details
    /// Verifica existência, extensão, e — para modelos Python — se a classe
    /// implementa a interface #MiiaModel corretamente.  Útil para validação
    /// antes de uma operação de carga.
    ///
    /// @param model_path  Caminho do arquivo a validar.
    ///
    /// @return #ValidationResult com o resultado da análise estática.
    ValidationResult validate_model(const std::string& model_path);

    /// @brief Aquece um modelo com execuções de inferência sintética.
    ///
    /// @details
    /// Pré-compila kernels JIT e preenche caches de memória, reduzindo a
    /// latência nas primeiras inferências reais.
    ///
    /// @param model_id   ID do modelo carregado.
    /// @param num_runs   Número de execuções de aquecimento (padrão: 5).
    ///
    /// @return #WarmupResult com estatísticas das execuções.
    WarmupResult warmup_model(const std::string& model_id,
                              uint32_t num_runs = 5);

    /// @}

    // -------------------------------------------------------------------------
    /// @name Observabilidade
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Verifica se o servidor está operacional.
    ///
    /// @return @c true se o servidor respondeu ao ping de saúde.
    bool health_check();

    /// @brief Retorna o estado operacional atual do worker.
    ///
    /// @return #WorkerStatus com contadores e lista de modelos carregados.
    WorkerStatus get_status();

    /// @brief Retorna métricas de desempenho globais e por modelo.
    ///
    /// @return #ServerMetrics com estatísticas acumuladas desde o início
    ///         do servidor.
    ServerMetrics get_metrics();

    /// @}

    // -------------------------------------------------------------------------
    /// @name Descoberta de modelos
    /// @{
    // -------------------------------------------------------------------------

    /// @brief Lista os arquivos de modelo disponíveis em um diretório.
    ///
    /// @details
    /// Varre o diretório em busca de arquivos com extensões suportadas
    /// (`.py`, `.onnx`) e informa quais já estão carregados.
    ///
    /// @param directory  Caminho do diretório a varrer.  Se vazio, usa o
    ///                   diretório de modelos configurado no servidor.
    ///
    /// @return Vetor de #AvailableModel descrevendo cada arquivo encontrado.
    std::vector<AvailableModel> list_available_models(
        const std::string& directory = "");

    /// @}

private:
    /// @cond INTERNAL
    std::unique_ptr<IClientBackend> backend_;
    /// @endcond
};

}  // namespace client
}  // namespace mlinference

#endif  // ML_INFERENCE_CLIENT_HPP
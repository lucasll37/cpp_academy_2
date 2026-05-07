# Mecanismos de Interoperabilidade entre Modelos de Inteligência Artificial e Agentes Autônomos em Simuladores Construtivos (MIIA) {#mainpage}

MIIA (Mecanismos de Interoperabilidade entre Modelos de Inteligência Artificial e Agentes Autônomos em Simuladores Construtivos) é um sistema de inferência de modelos desenvolvido no âmbito do projeto ASA, conduzido pelo Instituto de Estudos Avançados. Seu objetivo é integrar modelos de comportamento ao ambiente de simulação construtiva ASA/MIXR, permitindo que agentes autônomos deleguem decisões de raciocínio a modelos previamente criados.

O sistema fornece uma API C++ unificada que abstrai dois mecanismos de transporte (in-process e gRPC) e dois backends de execução (Python via CPython embutido e ONNX via ONNX Runtime), totalizando quatro modos de operação intercambiáveis sem necessidade de alterações no código do cliente.

---

## Visão geral da arquitetura {#visao-geral}

```
Agente / Simulador
       │
       ▼
 InferenceClient          ← API pública (include/client/inference_client.hpp)
       │
       ├── InProcessBackend   ← motor embutido no mesmo processo (sem rede)
       │       └── InferenceEngine
       │               ├── PythonBackend   (.py via CPython)
       │               └── OnnxBackend     (.onnx via ONNX Runtime)
       │
       └── GrpcClientBackend  ← chamadas gRPC para servidor remoto
               └── WorkerServer  (binário AsaMiia)
                       └── InferenceEngine
                               ├── PythonBackend
                               └── OnnxBackend
```

O modo de transporte é selecionado automaticamente pela string passada ao
construtor de @ref mlinference::client::InferenceClient :

```cpp
// In-process — motor embutido, sem rede
InferenceClient client("inprocess");

// gRPC — servidor remoto
InferenceClient client("localhost:50052");
```

O backend de execução é selecionado automaticamente pela extensão do arquivo
do modelo:

| Extensão | Backend       | Descrição                                  |
|----------|---------------|--------------------------------------------|
| `.py`    | PythonBackend | CPython embutido, inputs estruturados       |
| `.onnx`  | OnnxBackend   | ONNX Runtime, tensores float32 planos       |


### Diagrama de classes

![Diagrama de Sequência — In-process + Python](docs/images/classes_diagram.png)


---

## Quatro modos de operação {#quatro-modos}

A combinação de transporte × backend produz quatro modos distintos de uso:

| Modo | Transporte  | Backend | Caso de uso típico                         |
|------|------------|---------|---------------------------------------------|
| 1    | In-process | Python  | Simulador integrado com lógica Python       |
| 2    | In-process | ONNX    | Simulador integrado com modelo ONNX         |
| 3    | gRPC       | Python  | Servidor remoto servindo modelos Python     |
| 4    | gRPC       | ONNX    | Servidor remoto servindo modelos ONNX       |

### Modo 1 — In-process + Python

![Diagrama de Sequência — In-process + Python](docs/images/predict_inprocess_python.png)

```cpp
InferenceClient client("inprocess");
client.connect();
client.load_model("nav", "/app/models/ship_avoidance.py");

Object inputs;
inputs["state"] = Value{Object{
    {"toHeading", Value{45.0}},
    {"latitude",  Value{-23.5}},
    {"longitude", Value{-46.6}},
    {"hazards",   Value{Array{}}},
}};

auto r = client.predict("nav", inputs);
if (r.success)
    double heading = r.outputs["heading"].as_number();
```

### Modo 2 — In-process + ONNX

![Diagrama de Sequência — In-process + ONNX](docs/images/predict_inprocess_onnx.png)

```cpp
InferenceClient client("inprocess");
client.connect();
client.load_model("classificador", "/app/models/modelo.onnx");

Object inputs;
Array vetor{Value{1.0}, Value{2.0}, Value{3.0}, Value{4.0}, Value{5.0}};
inputs["input"] = Value{std::move(vetor)};

auto r = client.predict("classificador", inputs);
if (r.success)
    double saida = r.outputs["output"].as_number();
```

### Modo 3 — gRPC + Python

![Diagrama de Sequência — gRPC + Python](docs/images/predict_grpc_python.png)

```cpp
// Servidor deve estar rodando:
// ./AsaMiia --address 0.0.0.0:50052 --models-dir ./models
InferenceClient client("localhost:50052");
client.connect();
client.load_model("nav", "/app/models/ship_avoidance.py");  // path no servidor

auto r = client.predict("nav", inputs);
```

### Modo 4 — gRPC + ONNX

![Diagrama de Sequência — gRPC + ONNX](docs/images/predict_grpc_onnx.png)

```cpp
InferenceClient client("localhost:50052");
client.connect();
client.load_model("modelo", "/app/models/modelo.onnx");

auto r = client.predict("modelo", inputs);
```

---

## Sistema de tipos — Value / Object / Array {#tipos}

Todos os inputs e outputs de inferência usam o tipo
@ref mlinference::client::Value, que espelha `google.protobuf.Value` e
suporta aninhamento arbitrário.

### Construção

```cpp
using namespace mlinference::client;

Value v_num  {42.0};                           // número (double)
Value v_float{3.14f};                          // float → double
Value v_int  {42};                             // int → double
Value v_bool {true};                           // booleano
Value v_str  {std::string("modo_combate")};    // string
Value v_cstr {"literal"};                      // const char*
Value v_null {};                               // null

// Array
Array arr{Value{1.0}, Value{2.0}, Value{3.0}};
Value v_arr{std::move(arr)};

// Object (mapa nome → Value)
Object obj;
obj["x"] = Value{100.0};
obj["y"] = Value{200.0};
Value v_obj{std::move(obj)};

// Aninhamento arbitrário
Object estado;
estado["toHeading"] = Value{45.0};
estado["hazards"]   = Value{Array{
    Value{Object{
        {"bearing",     Value{90.0}},
        {"distance",    Value{500.0}},
        {"minSafeDist", Value{300.0}},
    }},
}};
```

### Verificadores de tipo

```cpp
v_num.is_number()   // true
v_bool.is_bool()    // true
v_str.is_string()   // true
v_arr.is_array()    // true
v_obj.is_object()   // true
v_null.is_null()    // true
```

### Acessores

```cpp
double             n = v_num.as_number();
bool               b = v_bool.as_bool();
const std::string& s = v_str.as_string();
const Array&       a = v_arr.as_array();
const Object&      o = v_obj.as_object();

// Acesso a elemento de array
double primeiro = v_arr.as_array()[0].as_number();

// Acesso a campo de objeto
double x = v_obj.as_object().at("x").as_number();
// ou (se o contrato é confiável):
double x = v_obj.as_object()["x"].as_number();
```

### Verificando o resultado da inferência

```cpp
auto r = client.predict("modelo", inputs);

if (!r.success) {
    // r.error_message contém traceback Python ou mensagem do ONNX Runtime
    LOG_ERROR("agente") << "Inferência falhou: " << r.error_message;
    return;
}

double heading   = r.outputs["heading"].as_number();   // escalar
const Array& arr = r.outputs["vetor"].as_array();      // array
double ms        = r.inference_time_ms;                // latência
```

---

## API pública do cliente {#api-cliente}

O único header que o código de aplicação precisa incluir é:

```cpp
#include <client/inference_client.hpp>
```

A classe @ref mlinference::client::InferenceClient expõe os seguintes grupos
de métodos:

### Conexão

| Método           | Retorno | Descrição                                             |
|------------------|---------|-------------------------------------------------------|
| `connect()`      | `bool`  | Inicializa o backend (motor local ou canal gRPC)      |
| `is_connected()` | `bool`  | Verifica se a conexão está ativa                      |

### Ciclo de vida dos modelos

| Método                             | Retorno | Descrição                               |
|------------------------------------|---------|-----------------------------------------|
| `load_model(id, path, version)`    | `bool`  | Carrega modelo pelo caminho do arquivo  |
| `unload_model(id)`                 | `bool`  | Remove um modelo da memória             |

### Inferência

| Método                        | Retorno                        | Descrição               |
|-------------------------------|--------------------------------|-------------------------|
| `predict(id, inputs)`         | `PredictionResult`             | Inferência unitária     |
| `batch_predict(id, batch)`    | `vector<PredictionResult>`     | Inferência em lote      |

### Introspecção

| Método                   | Retorno              | Descrição                              |
|--------------------------|----------------------|----------------------------------------|
| `list_models()`          | `vector<ModelInfo>`  | Lista modelos carregados com metadata  |
| `get_model_info(id)`     | `ModelInfo`          | Schema e metadados de um modelo        |
| `validate_model(path)`   | `ValidationResult`   | Valida arquivo sem carregar            |
| `warmup_model(id, n)`    | `WarmupResult`       | Aquece com `n` inferências sintéticas  |

### Observabilidade

| Método           | Retorno         | Descrição                                         |
|------------------|-----------------|---------------------------------------------------|
| `health_check()` | `bool`          | Verifica se o backend está operacional            |
| `get_status()`   | `WorkerStatus`  | Contadores e lista de modelos carregados          |
| `get_metrics()`  | `ServerMetrics` | Latências por modelo (avg, min, max, p95, p99)    |

### Descoberta de modelos

| Método                          | Retorno                   | Descrição                         |
|---------------------------------|---------------------------|-----------------------------------|
| `list_available_models(dir)`    | `vector<AvailableModel>`  | Lista `.py` e `.onnx` em diretório|

---

## Criando um modelo Python {#criando-modelo}

Todo modelo Python deve herdar de `MiiaModel` (definida em
`python/models/miia_model.py`) e implementar três métodos obrigatórios:
`load()`, `predict()` e `get_schema()`.

### Modelo com tensor plano (estilo ONNX)

```python
# models/modelo_linear.py
import numpy as np
from miia_model import MiiaModel, ModelSchema, TensorSpec

class ModeloLinear(MiiaModel):

    def load(self) -> None:
        """Carregado uma vez pelo servidor após instanciação."""
        self._pesos = np.array([2.0, 1.0, 0.5], dtype=np.float32)

    def predict(self, inputs: dict) -> dict:
        """
        inputs["entrada"] → list ou np.ndarray de 3 floats
        retorna {"saida": np.ndarray shape [1,1]}
        """
        x = np.array(inputs["entrada"], dtype=np.float32)
        resultado = float(x @ self._pesos)
        # Shape [1,1] → colapsado para escalar pelo servidor
        return {"saida": np.array([[resultado]], dtype=np.float32)}

    def get_schema(self) -> ModelSchema:
        return ModelSchema(
            inputs=[
                TensorSpec(
                    name="entrada",
                    shape=[1, 3],
                    dtype="float32",
                    description="Vetor de três features",
                )
            ],
            outputs=[
                TensorSpec(
                    name="saida",
                    shape=[1, 1],
                    dtype="float32",
                    description="Resultado escalar",
                )
            ],
            description="Modelo linear de exemplo",
            author="Equipe de Simulação",
        )

    def unload(self) -> None:
        """Libera recursos (opcional)."""
        del self._pesos
```

### Modelo com input estruturado (navegação por campo potencial)

```python
# models/ship_avoidance.py
import math
import numpy as np
from miia_model import MiiaModel, ModelSchema, TensorSpec

class ShipAvoidance(MiiaModel):

    def __init__(self) -> None:
        self._exp_param: float = 2.0
        self._peso_atracao: float = 1.0

    def load(self) -> None:
        pass  # stateless — pura matemática

    def predict(self, inputs: dict) -> dict:
        estado = inputs.get("state", {})
        if not estado:
            return {"heading": np.array([[0.0]], dtype=np.float32)}

        to_heading = float(estado.get("toHeading", 0.0))

        # Força de atração em direção ao steerpoint
        xres = math.cos(math.radians(to_heading)) * self._peso_atracao
        yres = math.sin(math.radians(to_heading)) * self._peso_atracao

        # Forças de repulsão dos obstáculos
        for hazard in estado.get("hazards", []):
            brg       = float(hazard.get("bearing",     0.0))
            dist      = float(hazard.get("distance",    0.0))
            safe_dist = float(hazard.get("minSafeDist", 0.0))

            if safe_dist <= 0.0:
                continue

            if dist > safe_dist * 2.0:
                peso = 0.0
            elif dist < safe_dist:
                peso = 1.0
            else:
                peso = ((safe_dist * 2.0 - dist) ** self._exp_param
                        / safe_dist ** self._exp_param)

            xres += math.cos(math.radians(brg)) * peso
            yres += math.sin(math.radians(brg)) * peso

        rumo_resultante = math.degrees(math.atan2(yres, xres))
        return {"heading": np.array([[rumo_resultante]], dtype=np.float32)}

    def get_schema(self) -> ModelSchema:
        return ModelSchema(
            inputs=[
                TensorSpec(
                    name="state",
                    shape=[-1],
                    dtype="float32",
                    structured=True,           # input é dict aninhado, não tensor plano
                    description=(
                        "Estado estruturado: "
                        "{ toHeading, latitude, longitude, "
                        "hazards: [{ bearing, distance, minSafeDist }] }"
                    ),
                )
            ],
            outputs=[
                TensorSpec(
                    name="heading",
                    shape=[1, 1],
                    dtype="float32",
                    description="Rumo comandado resultante em graus",
                )
            ],
            description="Campo potencial — atração ao steerpoint + repulsão de obstáculos",
            author="Equipe de Simulação",
            tags={"type": "navigation", "algorithm": "potential_field"},
        )

    def unload(self) -> None:
        pass

    def memory_usage_bytes(self) -> int:
        return 0  # stateless
```

### Regras importantes

- A classe deve ser instanciável **sem argumentos** (`__init__` não pode ter parâmetros obrigatórios além de `self`).
- Apenas a **primeira** subclasse de `MiiaModel` encontrada no módulo é utilizada — evite múltiplas subclasses em um único arquivo.
- Outputs com shape `[1]` ou `[1, 1]` são automaticamente colapsados para escalar pelo servidor — acesse com `as_number()` no lado C++.
- Não use `print()` — use o módulo `logging` do Python para não poluir o stdout do servidor.
- Não use `sys.exit()` ou `os._exit()` — encerra todo o processo do servidor.

### Validando um modelo antes de servir

```bash
# Script de validação que replica a lógica do find_model_class() do C++
python3 python/scripts/check_model.py models/ship_avoidance.py \
    --models-dir ./models
```

---

## Integrando em um simulador {#integrando}

O exemplo abaixo mostra como um agente de simulação C++ monta o estado do
navio, constrói o `Object` de input e obtém o rumo comandado via inferência.

### Header do agente

```cpp
// ShipAgent.hpp
#pragma once
#include <memory>
#include <string>
#include <client/inference_client.hpp>

class ShipAgent {
public:
    ShipAgent();
    void reasoning(double dt);
    void updateState(double dt);

private:
    // Configuração do cliente MIIA
    const std::string servidor    = "inprocess";                        // ou "localhost:50052"
    const std::string modelo_id   = "nav";
    const std::string modelo_path = "/app/models/ship_avoidance.py";

    std::unique_ptr<mlinference::client::InferenceClient> client;
    mlinference::client::PredictionResult resultado;

    // Monta o Object de input a partir do estado atual do navio
    mlinference::client::Object prepareState(ShipState* state);
};
```

### Inicialização no construtor

```cpp
// ShipAgent.cpp
ShipAgent::ShipAgent()
{
    client = std::make_unique<mlinference::client::InferenceClient>(servidor);
    client->connect();
    client->load_model(modelo_id, modelo_path);

    // Opcional: validar o modelo antes de usar
    auto val = client->validate_model(modelo_path);
    if (!val.valid)
        LOG_ERROR("ship_agent") << "Modelo inválido: " << val.error_message;

    // Opcional: aquecer o modelo antes do primeiro tick
    auto warm = client->warmup_model(modelo_id, 5);
    LOG_INFO("ship_agent") << "Warmup: avg=" << warm.avg_time_ms << " ms";
}
```

### Montagem do input estruturado

```cpp
mlinference::client::Object ShipAgent::prepareState(ShipState* state)
{
    using mlinference::client::Object;
    using mlinference::client::Array;
    using mlinference::client::Value;

    // Monta a lista de obstáculos detectados pelo scanner
    Array hazards;
    for (auto player : *state->getHazardSources()) {
        auto hs = dynamic_cast<HazardSource*>(player);
        if (!hs) continue;

        double brg{}, dist{};
        // Calcula rumo e distância ao obstáculo a partir das coordenadas geodésicas
        mixr::base::nav::gll2bd(
            hs->getLatitude(),  hs->getLongitude(),
            state->getLatitude(), state->getLongitude(),
            &brg, &dist);

        dist *= mixr::base::distance::NM2M;  // nmi → metros

        Object hazard;
        hazard["bearing"]     = Value{brg};
        hazard["distance"]    = Value{dist};
        hazard["minSafeDist"] = Value{hs->getMinSafeDist()};
        hazards.push_back(Value{std::move(hazard)});
    }

    // Monta o estado do navio
    Object stateObj;
    stateObj["toHeading"] = Value{state->getToHeading()};
    stateObj["latitude"]  = Value{state->getLatitude()};
    stateObj["longitude"] = Value{state->getLongitude()};
    stateObj["hazards"]   = Value{std::move(hazards)};

    // Empacota na raiz do Object (chave "state" esperada pelo modelo)
    Object root;
    root["state"] = Value{std::move(stateObj)};
    return root;
}
```

### Ciclo de raciocínio

```cpp
void ShipAgent::reasoning(const double dt)
{
    if (!client->is_connected()) {
        LOG_WARN("ship_agent") << "Cliente não conectado";
        return;
    }

    auto preparedState = prepareState(state);
    resultado = client->predict(modelo_id, preparedState);

    if (resultado.success && !resultado.outputs.empty()) {
        double hdgRes = resultado.outputs["heading"].as_number();
        LOG_DEBUG("ship_agent") << "Rumo comandado: " << hdgRes << " graus";

        action->setHasDynamicAction(true);
        action->setReqHeading(hdgRes);
    } else {
        LOG_WARN("ship_agent") << "Inferência falhou: " << resultado.error_message;
    }
}
```

---

## Build — Conan + Meson {#build}

### Pré-requisitos

```bash
# Dependências de sistema
sudo apt install \
    build-essential python3.12 python3.12-dev \
    python3.12-venv python3-numpy lcov graphviz

# Toolchain Python
pip install "conan>=2.0.0" "meson>=1.0.0" "ninja>=1.11.0" --break-system-packages

# Perfil Conan padrão
conan profile detect --force
```

### Fluxo completo de build

```bash
# 1. Instala dependências C++ via Conan e configura Meson
make configure

# 2. Compila todos os alvos
make build

# 3. Instala em ../dist/
make install

# 4. Cria venv Python e modelos de teste
make create-models
```

### Integrando como dependência Conan

Adicione ao `conanfile.py` do seu projeto:

```python
def requirements(self):
    self.requires("asa-poc-miia/1.0.0")
```

E ao `meson.build`:

```meson
miia_dep = dependency('asa-poc-miia', method: 'pkg-config')

executable('meu_simulador',
    'src/main.cpp',
    dependencies: [miia_dep],
)
```

O pacote exporta a biblioteca `asa_miia_client` e os headers em
`include/asa-poc-miia/`.

### Dependências gerenciadas pelo Conan

| Dependência   | Versão      | Uso                                    |
|---------------|-------------|----------------------------------------|
| `grpc`        | 1.54.3      | Transporte gRPC                        |
| `protobuf`    | 3.21.12     | Serialização de mensagens              |
| `onnxruntime` | 1.18.1      | Backend de inferência ONNX             |
| `gtest`       | 1.14.0      | Framework de testes C++                |
| `abseil`      | 20230802.1  | Utilitários (dependência do gRPC)      |
| `re2`         | 20230301    | Regex (dependência do gRPC)            |

### Opções de build

| Opção        | Tipo    | Padrão  | Descrição                                  |
|--------------|---------|---------|--------------------------------------------|
| `enable_gpu` | boolean | `false` | Habilita CUDA Execution Provider no ONNX Runtime |

---

## Testes unitários {#testes-unitarios}

Os testes unitários **não requerem servidor rodando** e podem ser executados
imediatamente após o build. Cobrem os componentes internos sem dependência
de rede ou de arquivo de modelo externo.

### Executando

```bash
# Todos os unitários com saída colorida
make test-unit

# Com output completo (inclui logs de cada teste)
make test-verbose

# Filtro por suite via Meson
meson test -C build/ --suite unit

# Filtro por nome via GTest (executável direto)
build/tests/unit/test_unit_value --gtest_filter="ValueDouble.*"
build/tests/unit/test_unit_value --gtest_filter="*NaN*"
```

### Variáveis de ambiente para testes unitários

| Variável     | Padrão      | Descrição                                              |
|--------------|-------------|--------------------------------------------------------|
| `MODELS_DIR` | `./models`  | Diretório contendo `miia_model.py` (para `test_python_backend`) |
| `LOG_LEVEL`  | (desligado) | Nível de log durante os testes                         |

---

## Testes de integração {#testes-integracao}

Os testes de integração exercem a API completa de `InferenceClient` nos quatro
modos de operação. Os testes gRPC detectam automaticamente se o servidor está
disponível e pulam com `GTEST_SKIP` caso contrário — o `make test-integration`
pode ser executado sem servidor.

### Executando

```bash
# Terminal 1 — inicia o servidor (necessário apenas para testes gRPC)
make run-server

# Terminal 2 — executa todos os testes de integração
make test-integration
```

### Variáveis de ambiente para testes de integração

| Variável         | Padrão            | Descrição                                    |
|------------------|------------------|----------------------------------------------|
| `WORKER_ADDRESS` | `localhost:50052` | Endereço do servidor gRPC                   |
| `MODELS_DIR`     | `./models`        | Diretório de modelos (cliente e servidor)    |
| `LOG_LEVEL`      | `DEBUG`           | Nível de log (definido no `meson.build` de integração) |

---

## Docker {#docker}

O projeto inclui um Dockerfile multi-stage que produz imagens mínimas para
CPU e GPU. Os modelos **não** são embutidos na imagem — são montados via
volume em tempo de execução.

### Estágios do Dockerfile

| Estágio         | Base                          | Responsabilidade                              |
|-----------------|-------------------------------|-----------------------------------------------|
| `builder`       | ubuntu:24.04                  | Instala Conan, Meson, Python, Ninja           |
| `conan-deps`    | builder                       | Instala dependências C++ via Conan (cache de layer) |
| `app-builder`   | conan-deps                    | Compila, instala binário, executa testes unitários |
| `lib-collector` | app-builder                   | Coleta `.so` de runtime via `ldd` (exclui libc/libm/etc.) |
| `runtime-cpu`   | ubuntu:24.04                  | Imagem mínima CPU (binário + libs coletadas)  |
| `runtime-gpu`   | nvidia/cuda:12.0.0-runtime    | Imagem mínima GPU                             |

### Build das imagens

```bash
# CPU
make docker-build-cpu

# GPU
make docker-build-gpu
```

### Executando o servidor

```bash
# CPU
make docker-run-cpu

# GPU
make docker-run-gpu
```

### Variáveis de ambiente do container

| Variável          | Padrão               | Descrição                                    |
|-------------------|---------------------|----------------------------------------------|
| `MODELS_DIR`      | `/app/models`       | Diretório de modelos dentro do container     |
| `VIRTUAL_ENV`     | `/app/models/.venv` | Path do venv Python para o backend Python    |
| `LOG_LEVEL`       | (desligado)         | Nível de log do servidor                     |
| `LD_LIBRARY_PATH` | `/app/lib:/usr/local/lib` | Path das bibliotecas runtime coletadas |
| `ENABLE_GPU`      | `False` (cpu) / `True` (gpu) | Indica o modo de execução           |

---

## Cobertura de código {#cobertura}

O projeto suporta geração de relatório HTML de cobertura via `lcov` +
`genhtml`, instrumentado pelo Meson com `-Db_coverage=true`.

### Executando

```bash
# Configura, compila com instrumentação e executa testes
make coverage

# Abre o relatório HTML no navegador
make coverage-open
```

### Interpretando o relatório

O relatório HTML exibe, para cada arquivo de código:

- **Lines** — porcentagem de linhas executadas
- **Functions** — porcentagem de funções chamadas
- **Branches** — porcentagem de ramos de decisão cobertos

---

## Sistema de logging {#logging}

O logger é baseado em componentes nomeados com saída simultânea em arquivo
próprio e no arquivo agregado `all_*.log`. O nível mínimo é controlado pela
variável de ambiente `LOG_LEVEL` — se não definida, o logger fica desligado
e nenhum arquivo é criado.

### Configuração inicial

```cpp
#include <utils/logger.hpp>

// Chamar antes do primeiro uso de qualquer logger
Logger::set_base_dir("logs");           // diretório base dos arquivos
Logger::set_default_stderr(true);       // espelhar em stderr por padrão

// Configuração por componente (opcional, antes do primeiro uso do componente)
Logger::configure("agente", LoggerConfig{
    .dir         = "logs/agente",   // diretório próprio
    .also_stderr = true,            // sempre no terminal
});

Logger::configure("inferencia", LoggerConfig{
    .also_stderr = false,           // apenas em arquivo
});
```

### Macros de uso

```cpp
// Logger nomeado — gera arquivo próprio + espelha em all.log
LOG_DEBUG("agente") << "Tick " << tick << " — rumo: " << heading;
LOG_INFO("agente")  << "Modelo carregado: " << model_id;
LOG_WARN("agente")  << "Scanner sem dados válidos, usando fallback";
LOG_ERROR("agente") << "Inferência falhou: " << r.error_message;

// Sintaxe alternativa com level explícito
LOG("agente", INFO) << "Equivalente ao LOG_INFO acima";

// Logger "default" — apenas espelhado em all.log, sem arquivo próprio
LOG_INFO() << "Mensagem avulsa";
LOG(WARN)  << "Aviso sem componente definido";
```

### Variável de ambiente

```bash
export LOG_LEVEL=DEBUG   # DEBUG | INFO | WARN | ERROR
# Sem LOG_LEVEL → logger desligado (OFF), nenhum arquivo criado
```

### Formato de saída

```
HH:MM:SS.mmm LEVEL  componente  mensagem  [arquivo:linha]
```

Exemplo:
```
10:42:17.053 INFO   agente      [reasoning] Inferência concluída em 1.2 ms  [ShipAgent.cpp:87]
10:42:17.061 DEBUG  python_backend  [predict] GIL adquirido  [python_backend.cpp:318]
10:42:17.062 DEBUG  python_backend  [predict] model.predict() retornou  [python_backend.cpp:345]
```

No terminal, os níveis são coloridos com ANSI:

| Nível | Cor      |
|-------|----------|
| DEBUG | Ciano    |
| INFO  | Verde    |
| WARN  | Amarelo  |
| ERROR | Vermelho |

### Arquivos gerados

```
logs/
├── all_20260503_104217.log        ← todas as mensagens de todos os componentes
├── agente_20260503_104217.log     ← apenas mensagens do componente "agente"
└── inferencia_20260503_104217.log ← apenas mensagens do componente "inferencia"
```

O componente `"default"` (macros sem nome) não gera arquivo próprio — suas
mensagens vão apenas para `all_*.log`.

---








## Qualidade de código {#qualidade}

O projeto usa duas ferramentas complementares do ecossistema LLVM para manter a consistência e qualidade do código C++:

| Ferramenta | O que faz |
|---|---|
| **clang-tidy** | Analisa o código em busca de bugs, padrões problemáticos e desvios de estilo. **Não modifica arquivos.** |
| **clang-format** | Reformata o código automaticamente para seguir o estilo definido. **Modifica arquivos in-place.** |

As regras de cada ferramenta ficam em arquivos na raiz do projeto:
- `.clang-tidy` — define quais checks rodar e com quais opções
- `.clang-format` — define as regras de formatação (indentação, espaçamento, quebras de linha, etc.)

---

### clang-tidy — análise estática

O clang-tidy precisa do `compile_commands.json` para saber como o projeto é compilado — quais flags, includes e defines estão em uso. Por isso todos os targets passam `-p build/`.

**Analisar todo o código fonte:**
```bash
make tidy
```

Analisa todos os arquivos em `core/`. O output é exibido diretamente no terminal com colorização.

**Analisar um arquivo específico:**
```bash
make tidy-file FILE=core/inference/src/onnx_backend.cpp
```

Útil durante o desenvolvimento para checar um arquivo antes de commitar.

**Ver o output com paginação:**
```bash
make tidy-file FILE=core/inference/src/onnx_backend.cpp 2>&1 | less
```

**Exemplo de output:**
```
core/inference/src/onnx_backend.cpp:42:5: warning: use 'nullptr' instead of '0' [modernize-use-nullptr]
    ptr = 0;
    ^~~~~~
    nullptr
```

Cada aviso mostra: arquivo, linha, coluna, descrição, nome do check entre colchetes e o trecho de código afetado.

**Sobre os checks ativos:**

O `.clang-tidy` na raiz do projeto define três perfis — apenas um deve estar ativo por vez:

| Perfil | Quando usar |
|---|---|
| **Permissivo** (padrão) | Bugs reais e undefined behavior. Ponto de partida. |
| **Moderado** | Após zerar os avisos do permissivo. Adiciona `modernize-*` e `performance-*`. |
| **Estrito** | Meta de qualidade para PRs. Adiciona `readability-*` e `concurrency-*`. |

Para trocar de perfil, edite `.clang-tidy` e descomente o bloco `Checks:` desejado.

---

### clang-format — formatação automática

Diferente do clang-tidy, o clang-format **não precisa do diretório de build** — ele trabalha puramente na sintaxe do arquivo, sem resolver includes ou dependências.

> **Importante:** clang-format modifica apenas **estilo** (indentação, espaços, quebras de linha, ordem de includes). Ele nunca altera lógica, nomes ou comportamento do código.

**Formatar todo o código fonte (in-place):**
```bash
make format
```

**Verificar formatação sem modificar nada** (útil no CI):
```bash
make format-check
```

Retorna código de saída não-zero se algum arquivo estiver fora do padrão — útil como gate em pipelines de CI.

**Formatar um arquivo específico:**
```bash
make format-file FILE=core/inference/src/onnx_backend.cpp
```

**Ver o diff do que seria alterado sem aplicar:**
```bash
make format-diff FILE=core/inference/src/onnx_backend.cpp
```

Mostra exatamente o que mudaria, sem tocar no arquivo.

**Exemplo de output do `format-diff`:**
```diff
--- core/inference/src/onnx_backend.cpp
+++ (formatted)
@@ -38,7 +38,7 @@
-int x=1;
+int x = 1;
-std::string s="hello";
+std::string s = "hello";
```
---

### Integração com VS Code

Com as extensões **clangd** ou **C/C++** instaladas, o VS Code detecta automaticamente os arquivos `.clang-tidy` e `.clang-format` na raiz do projeto e aplica as regras em tempo real durante a edição — sublinhando avisos do tidy e formatando ao salvar.

Não é necessária nenhuma configuração adicional além de ter o `build/compile_commands.json` gerado (`make configure`).







## Documentação Doxygen {#doxygen}

A documentação gerada pelo Doxygen inclui descrições de classes, métodos e diagramas de dependência. Os arquivos de cabeçalho (`*.hpp`, `*.h`) devem conter comentários no formato Doxygen para que a documentação seja gerada corretamente.

### Gerando a documentação

```bash
# Gera a documentação HTML em docs/doxygen/html/
make docs

# Abre a página principal no navegador
make docs-open
```

---

## Estrutura do projeto {#estrutura}

```
poc-miia/
├── core/
│   ├── client/
│   │   ├── include/client/
│   │   │   ├── inference_client.hpp    ← API pública
│   │   │   ├── i_client_backend.hpp    ← interface interna de transporte
│   │   │   ├── inprocess_backend.hpp   ← backend in-process
│   │   │   ├── grpc_client_backend.hpp ← backend gRPC
│   │   │   └── value_convert.hpp       ← conversão Value ↔ protobuf
│   │   └── src/
│   ├── inference/
│   │   ├── include/inference/
│   │   │   ├── inference_engine.hpp    ← motor de inferência
│   │   │   ├── model_backend.hpp       ← interface + RuntimeMetrics
│   │   │   ├── backend_registry.hpp    ← registro singleton de backends
│   │   │   ├── onnx_backend.hpp        ← backend ONNX Runtime
│   │   │   └── python_backend.hpp      ← backend CPython embutido
│   │   └── src/
│   ├── server/
│   │   ├── include/server/
│   │   │   └── worker_server.hpp       ← servidor gRPC + RPCs
│   │   └── src/
│   └── utils/
│       ├── include/utils/
│       │   └── logger.hpp              ← sistema de logging
│       └── src/
├── proto/
│   ├── common.proto                    ← tipos compartilhados
│   └── server.proto                    ← serviço gRPC WorkerService
├── models/                             ← modelos Python e ONNX
│   ├── .venv/                          ← venv Python (gerado por make)
│   ├── miia_model.py                   ← classe base abstrata
│   └── tutorial_model.py              ← ShipAvoidance (modelo de referência)
├── python/
│   ├── models/
│   │   └── miia_model.py               ← fonte do miia_model.py
│   ├── requirements.txt
│   └── scripts/
│       ├── create_test_models.py       ← gera modelos ONNX de teste
│       └── check_model.py              ← valida implementação de MiiaModel
├── tests/
│   ├── unit/                           ← testes sem servidor
│   └── integration/                    ← testes com/sem servidor
├── docker/
│   └── Dockerfile.server               ← multi-stage CPU/GPU
├── docs/
│   ├── Doxyfile                        ← configuração Doxygen
│   └── diagrams/                       ← diagramas draw.io
├── conanfile.py                        ← dependências e build Conan
├── meson.build                         ← build principal
├── meson_options.txt                   ← opções (enable_gpu)
└── Makefile                            ← targets de conveniência
```

### Targets do Makefile

| Target              | Descrição                                                   |
|---------------------|-------------------------------------------------------------|
| `configure`         | Instala dependências Conan e configura Meson                |
| `build`             | Compila todos os alvos                                      |
| `install`           | Instala em `../dist/`                                       |
| `package`           | Cria pacote de instalação em `../dist/`                     |
| `clean`             | Remove `build/` e cache de subprojetos                      |
| `run-server`        | Inicia o servidor AsaMiia localmente                        |
| `run-client`        | Executa o cliente de exemplo CLI                            |
| `test`              | Executa todos os testes                                     |
| `test-unit`         | Executa apenas testes unitários (sem servidor)              |
| `test-integration`  | Executa testes de integração (servidor opcional)            |
| `test-verbose`      | Executa testes com saída detalhada e logs de erros          |
| `configure-coverage`| Configura build com instrumentação de cobertura             |
| `coverage`          | Gera relatório HTML de cobertura em `build/coverage-report/`|
| `python-env`        | Cria o venv Python em `models/.venv`                        |
| `python-install`    | Instala dependências Python                                 |
| `create-models`     | Gera modelos ONNX de teste e copia modelos Python           |
| `docs`              | Gera documentação Doxygen em `docs/doxygen/`                |
| `docs-open`         | Abre a documentação no navegador                            |
| `docs-clean`        | Remove a documentação gerada                                |
| `docker-build-cpu`  | Constrói imagem Docker CPU                                  |
| `docker-build-gpu`  | Constrói imagem Docker GPU                                  |
| `docker-run-cpu`    | Executa imagem Docker CPU                                   |
| `docker-run-gpu`    | Executa imagem Docker GPU                                   |
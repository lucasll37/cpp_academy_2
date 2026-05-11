# Mecanismos de Interoperabilidade entre Modelos de Inteligência Artificial e Agentes Autônomos em Simuladores Construtivos (MIIA) {#mainpage}

MIIA é um sistema de inferência de modelos desenvolvido no âmbito do projeto ASA, conduzido pelo Instituto de Estudos Avançados. Seu objetivo é integrar modelos de comportamento ao ambiente de simulação construtiva ASA/MIXR, permitindo que agentes autônomos deleguem decisões de raciocínio a modelos previamente criados.

O sistema fornece uma API C++ unificada que abstrai dois mecanismos de transporte (in-process e gRPC) e dois backends de execução (Python via CPython embutido e ONNX via ONNX Runtime), totalizando quatro modos de operação intercambiáveis sem necessidade de alterações no código do cliente.

---

## Início rápido {#inicio-rapido}

### Pré-requisitos

```bash
sudo apt install \
    build-essential python3.12 python3.12-dev \
    python3.12-venv python3-numpy lcov graphviz clangd clang-format clang-tidy
```

```bash
pip install "conan>=2.0.0" "meson>=1.0.0" "ninja>=1.11.0" --break-system-packages
```

```bash
conan profile detect --force
```

### Fluxo completo

Cria o venv Python e os modelos de teste em `models/`:

```bash
make create-models
```

Instala dependências C++ via Conan e configura o Meson:

```bash
make configure
```

Compila todos os alvos:

```bash
make build
```

Gera a documentação Doxygen em `docs/doxygen/generated/`:

```bash
make docs
```

Instala em `../dist/`:

```bash
make install
```

Cria o pacote Conan (Debug + Release):

```bash
make package
```

**Nota:** `make configure` habilita instrumentação de cobertura (`-Db_coverage=true`).

---

## Executando {#executando}

Inicia o servidor gRPC:

```bash
make run-server

# Equivalente a:
./build/core/server/AsaMiia --address 0.0.0.0:50052 --models-dir ./models --threads 8
```


Executa o cliente de exemplo CLI:

```bash
make run-client

# Equivalente a:
./build/core/client/AsaMiiaClient --models-dir ./models --address localhost:50052
```


### Opções do servidor

| Opção               | Padrão              | Descrição                        |
|---------------------|---------------------|----------------------------------|
| `--address <addr>`  | `0.0.0.0:50052`     | Endereço de escuta               |
| `--worker-id <id>`  | `worker-1`          | Identificador do worker          |
| `--threads <n>`     | `4`                 | Threads de inferência            |
| `--models-dir <path>` | `./models`        | Diretório local de modelos       |
| `--gpu`             | —                   | Habilita inferência GPU          |

---

## Docker {#docker}

Os modelos **não** são embutidos na imagem — são montados via volume em tempo de execução.

Build das imagens CPU e GPU:

```bash
make docker-build-cpu
make docker-build-gpu
```

Execução do container CPU:

```bash
make docker-run-cpu
```

Equivalente a:

```bash
docker run -d --rm -p 50052:50052 -v $(PWD)/models:/app/models:ro ml-inference-server:latest-cpu
```

Execução do container GPU:

```bash
make docker-run-gpu
```

Equivalente a:

```bash
docker run -d --rm -p 50052:50052 --gpus all -v $(PWD)/models:/app/models:ro ml-inference-server:latest-gpu
```

### Variáveis de ambiente do container

| Variável            | Padrão                    | Descrição                                      |
|---------------------|---------------------------|------------------------------------------------|
| `MODELS_DIR`        | `/app/models`             | Diretório de modelos dentro do container       |
| `VIRTUAL_ENV`       | `/app/models/.venv`       | Path do venv Python                            |
| `LOG_LEVEL`         | (desligado)               | Nível de log do servidor                       |
| `LD_LIBRARY_PATH`   | `/app/lib:/usr/local/lib` | Path das bibliotecas runtime coletadas         |
| `ENABLE_GPU`        | `False` (cpu) / `True` (gpu) | Indica o modo de execução                   |

### Estágios do Dockerfile

| Estágio         | Base                       | Responsabilidade                                    |
|-----------------|----------------------------|-----------------------------------------------------|
| `builder`       | ubuntu:24.04               | Instala Conan, Meson, Python, Ninja                 |
| `conan-deps`    | builder                    | Instala dependências C++ via Conan (cache de layer) |
| `app-builder`   | conan-deps                 | Compila, instala binário, executa testes unitários  |
| `lib-collector` | app-builder                | Coleta `.so` de runtime via `ldd`                   |
| `runtime-cpu`   | ubuntu:24.04               | Imagem mínima CPU                                   |
| `runtime-gpu`   | nvidia/cuda:12.0.0-runtime | Imagem mínima GPU                                   |

---

## Integrando como dependência Conan {#dependencia}

Declaração da dependência no `conanfile.py`:

```python
# conanfile.py
def requirements(self):
    self.requires("asa-poc-miia/1.0.0")
```

Declaração da dependência no `meson.build`:

```meson
# meson.build
miia_dep = dependency('asa-poc-miia', method: 'pkg-config')

executable('meu_simulador',
    'src/main.cpp',
    dependencies: [miia_dep],
)
```

O pacote exporta a biblioteca `asa_miia_client` e os headers em `include/asa-poc-miia/`.

---

## Quatro modos de operação {#quatro-modos}

A combinação de transporte × backend produz quatro modos distintos:

| Modo | Transporte  | Backend | Caso de uso típico                      |
|------|-------------|---------|------------------------------------------|
| 1    | In-process  | Python  | Simulador integrado com lógica Python    |
| 2    | In-process  | ONNX    | Simulador integrado com modelo ONNX      |
| 3    | gRPC        | Python  | Servidor remoto servindo modelos Python  |
| 4    | gRPC        | ONNX    | Servidor remoto servindo modelos ONNX    |

O modo de transporte é selecionado pela string passada ao construtor de `InferenceClient`:

```cpp
InferenceClient client("inprocess");       // In-process — sem rede
```

```cpp
InferenceClient client("localhost:50052"); // gRPC — servidor remoto
```

O backend de execução é selecionado automaticamente pela extensão do arquivo do modelo:

| Extensão | Backend       | Descrição                             |
|----------|---------------|---------------------------------------|
| `.py`    | PythonBackend | CPython embutido, inputs estruturados |
| `.onnx`  | OnnxBackend   | ONNX Runtime, tensores float32 planos |

---

## API pública do cliente {#api-cliente}

```cpp
#include <client/inference_client.hpp>
```

### Conexão

| Método           | Retorno | Descrição                                        |
|------------------|---------|--------------------------------------------------|
| `connect()`      | `bool`  | Inicializa o backend (motor local ou canal gRPC) |
| `is_connected()` | `bool`  | Verifica se a conexão está ativa                 |

### Ciclo de vida dos modelos

| Método                          | Retorno | Descrição                              |
|---------------------------------|---------|----------------------------------------|
| `load_model(id, path, version)` | `bool`  | Carrega modelo pelo caminho do arquivo |
| `unload_model(id)`              | `bool`  | Remove um modelo da memória            |

### Inferência

| Método                     | Retorno                    | Descrição           |
|----------------------------|----------------------------|---------------------|
| `predict(id, inputs)`      | `PredictionResult`         | Inferência unitária |
| `batch_predict(id, batch)` | `vector<PredictionResult>` | Inferência em lote  |

### Introspecção

| Método                 | Retorno             | Descrição                             |
|------------------------|---------------------|---------------------------------------|
| `list_models()`        | `vector<ModelInfo>` | Lista modelos carregados com metadata |
| `get_model_info(id)`   | `ModelInfo`         | Schema e metadados de um modelo       |
| `validate_model(path)` | `ValidationResult`  | Valida arquivo sem carregar           |
| `warmup_model(id, n)`  | `WarmupResult`      | Aquece com `n` inferências sintéticas |

### Observabilidade

| Método           | Retorno         | Descrição                                      |
|------------------|-----------------|------------------------------------------------|
| `health_check()` | `bool`          | Verifica se o backend está operacional         |
| `get_status()`   | `WorkerStatus`  | Contadores e lista de modelos carregados       |
| `get_metrics()`  | `ServerMetrics` | Latências por modelo (avg, min, max, p95, p99) |

### Descoberta de modelos

| Método                       | Retorno                  | Descrição                          |
|------------------------------|--------------------------|------------------------------------|
| `list_available_models(dir)` | `vector<AvailableModel>` | Lista `.py` e `.onnx` em diretório |

---

## Sistema de tipos — Value / Object / Array {#tipos}

Todos os inputs e outputs de inferência usam `mlinference::client::Value`, que espelha `google.protobuf.Value` e suporta aninhamento arbitrário.

### Construção

Tipos escalares:

```cpp
using namespace mlinference::client;

Value v_num  {42.0};
Value v_bool {true};
Value v_str  {std::string("modo_combate")};
Value v_null {};
```

Array e objeto simples:

```cpp
Array arr{Value{1.0}, Value{2.0}, Value{3.0}};
Value v_arr{std::move(arr)};

Object obj;
obj["x"] = Value{100.0};
obj["y"] = Value{200.0};
Value v_obj{std::move(obj)};
```

Aninhamento arbitrário (exemplo de estado de navegação):

```cpp
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

### Verificadores e acessores

```cpp
v_num.is_number();   v_num.as_number();
v_bool.is_bool();    v_bool.as_bool();
v_str.is_string();   v_str.as_string();
v_arr.is_array();    v_arr.as_array()[0].as_number();
v_obj.is_object();   v_obj.as_object()["x"].as_number();
```

### Verificando o resultado da inferência

```cpp
auto r = client.predict("modelo", inputs);

if (!r.success) {
    LOG_ERROR("agente") << "Inferência falhou: " << r.error_message;
    return;
}

double heading   = r.outputs["heading"].as_number();
const Array& arr = r.outputs["vetor"].as_array();
double ms        = r.inference_time_ms;
```

---

## Integrando em um simulador {#integrando}

### Header do agente

```cpp
// ShipAgent.hpp
#pragma once
#include <memory>
#include <client/inference_client.hpp>

class ShipAgent {
public:
    ShipAgent();
    void reasoning(double dt);

private:
    const std::string servidor    = "inprocess";
    const std::string modelo_id   = "nav";
    const std::string modelo_path = "/app/models/ship_avoidance.py";

    std::unique_ptr<mlinference::client::InferenceClient> client;
    mlinference::client::PredictionResult resultado;

    mlinference::client::Object prepareState(ShipState* state);
};
```

### Inicialização

```cpp
ShipAgent::ShipAgent()
{
    client = std::make_unique<mlinference::client::InferenceClient>(servidor);
    client->connect();
    client->load_model(modelo_id, modelo_path);

    auto val = client->validate_model(modelo_path);
    if (!val.valid)
        LOG_ERROR("ship_agent") << "Modelo inválido: " << val.error_message;

    auto warm = client->warmup_model(modelo_id, 5);
    LOG_INFO("ship_agent") << "Warmup: avg=" << warm.avg_time_ms << " ms";
}
```

### Montagem do input estruturado

```cpp
mlinference::client::Object ShipAgent::prepareState(ShipState* state)
{
    using namespace mlinference::client;

    Array hazards;
    for (auto player : *state->getHazardSources()) {
        auto hs = dynamic_cast<HazardSource*>(player);
        if (!hs) continue;

        double brg{}, dist{};
        mixr::base::nav::gll2bd(
            hs->getLatitude(),  hs->getLongitude(),
            state->getLatitude(), state->getLongitude(),
            &brg, &dist);
        dist *= mixr::base::distance::NM2M;

        Object hazard;
        hazard["bearing"]     = Value{brg};
        hazard["distance"]    = Value{dist};
        hazard["minSafeDist"] = Value{hs->getMinSafeDist()};
        hazards.push_back(Value{std::move(hazard)});
    }

    Object stateObj;
    stateObj["toHeading"] = Value{state->getToHeading()};
    stateObj["latitude"]  = Value{state->getLatitude()};
    stateObj["longitude"] = Value{state->getLongitude()};
    stateObj["hazards"]   = Value{std::move(hazards)};

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

    resultado = client->predict(modelo_id, prepareState(state));

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

## Criando um modelo Python {#criando-modelo}

Todo modelo Python deve herdar de `MiiaModel` (`python/models/miia_model.py`) e implementar `load()`, `predict()` e `get_schema()`.

### Regras

- A classe deve ser instanciável **sem argumentos** (`__init__` sem parâmetros obrigatórios além de `self`).
- Apenas a **primeira** subclasse de `MiiaModel` encontrada no módulo é utilizada.
- Outputs com shape `[1]` ou `[1, 1]` são automaticamente colapsados para escalar — acesse com `as_number()` no lado C++.
- Use `logging` em vez de `print()` para não poluir o stdout do servidor.
- Não use `sys.exit()` ou `os._exit()`.

### Modelo com tensor plano

```python
# models/modelo_linear.py
import numpy as np
from miia_model import MiiaModel, ModelSchema, TensorSpec

class ModeloLinear(MiiaModel):

    def load(self) -> None:
        self._pesos = np.array([2.0, 1.0, 0.5], dtype=np.float32)

    def predict(self, inputs: dict) -> dict:
        x = np.array(inputs["entrada"], dtype=np.float32)
        return {"saida": np.array([[float(x @ self._pesos)]], dtype=np.float32)}

    def get_schema(self) -> ModelSchema:
        return ModelSchema(
            inputs=[TensorSpec(name="entrada", shape=[1, 3], dtype="float32")],
            outputs=[TensorSpec(name="saida",  shape=[1, 1], dtype="float32")],
        )

    def unload(self) -> None:
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
        pass

    def predict(self, inputs: dict) -> dict:
        estado = inputs.get("state", {})
        if not estado:
            return {"heading": np.array([[0.0]], dtype=np.float32)}

        to_heading = float(estado.get("toHeading", 0.0))
        xres = math.cos(math.radians(to_heading)) * self._peso_atracao
        yres = math.sin(math.radians(to_heading)) * self._peso_atracao

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

        return {"heading": np.array([[math.degrees(math.atan2(yres, xres))]], dtype=np.float32)}

    def get_schema(self) -> ModelSchema:
        return ModelSchema(
            inputs=[TensorSpec(
                name="state", shape=[-1], dtype="float32", structured=True,
                description="{ toHeading, latitude, longitude, hazards: [{ bearing, distance, minSafeDist }] }",
            )],
            outputs=[TensorSpec(name="heading", shape=[1, 1], dtype="float32",
                                description="Rumo comandado resultante em graus")],
            description="Campo potencial — atração ao steerpoint + repulsão de obstáculos",
            tags={"type": "navigation", "algorithm": "potential_field"},
        )

    def unload(self) -> None:
        pass

    def memory_usage_bytes(self) -> int:
        return 0
```

### Validando antes de servir

```bash
python3 python/scripts/check_model.py models/ship_avoidance.py --models-dir ./models
```

---

## Testes {#testes}

### Unitários

Não requerem servidor. Podem ser executados imediatamente após o build.

Executa todos os testes unitários:

```bash
make test-unit
```

Executa com output completo:

```bash
make test-verbose
```

Alternativa via Meson diretamente:

```bash
meson test -C build/ --suite unit
```

Filtrando por nome (executável direto):

```bash
build/tests/unit/test_unit_value --gtest_filter="ValueDouble.*"
```

| Variável     | Padrão     | Descrição                                              |
|--------------|------------|--------------------------------------------------------|
| `MODELS_DIR` | `./models` | Diretório contendo `miia_model.py`                     |
| `LOG_LEVEL`  | (desligado)| Nível de log durante os testes                         |

### Integração

Os testes gRPC detectam automaticamente se o servidor está disponível e pulam com `GTEST_SKIP` caso contrário.

No terminal 1, inicie o servidor:

```bash
make run-server
```

No terminal 2, execute os testes de integração:

```bash
make test-integration
```

| Variável         | Padrão            | Descrição                                 |
|------------------|-------------------|-------------------------------------------|
| `WORKER_ADDRESS` | `localhost:50052` | Endereço do servidor gRPC                 |
| `MODELS_DIR`     | `./models`        | Diretório de modelos (cliente e servidor) |
| `LOG_LEVEL`      | `DEBUG`           | Nível de log                              |

---

## Cobertura de código {#cobertura}

A instrumentação é habilitada automaticamente pelo `make configure` (`-Db_coverage=true`).

Executa os testes e gera o relatório HTML:

```bash
make coverage
```

Abre o relatório em `build/coverage-report/index.html` no browser:

```bash
make coverage-open
```

O relatório exibe, por arquivo: **Lines**, **Functions** e **Branches** cobertos.

---

## Sistema de logging {#logging}

Logger baseado em componentes nomeados, com saída simultânea em arquivo próprio e em `all_*.log`. Controlado pela variável `LOG_LEVEL` — se não definida, nenhum arquivo é criado.

Configuração inicial:

```cpp
#include <utils/logger.hpp>

Logger::set_base_dir("logs");
Logger::set_default_stderr(true);

Logger::configure("agente", LoggerConfig{.dir = "logs/agente", .also_stderr = true});
```

Uso dos macros de log:

```cpp
LOG_DEBUG("agente") << "Tick " << tick << " — rumo: " << heading;
LOG_INFO("agente")  << "Modelo carregado: " << model_id;
LOG_WARN("agente")  << "Scanner sem dados válidos";
LOG_ERROR("agente") << "Inferência falhou: " << r.error_message;

LOG_INFO() << "Mensagem avulsa (apenas em all.log)";
```

Ativando o log via variável de ambiente:

```bash
export LOG_LEVEL=DEBUG   # DEBUG | INFO | WARN | ERROR
```

Formato de saída:
```
HH:MM:SS.mmm LEVEL  componente  mensagem  [arquivo:linha]
```

Arquivos gerados em `logs/`: `all_<timestamp>.log` + um arquivo por componente nomeado.

---

## Qualidade de código {#qualidade}

| Ferramenta      | O que faz                                                        |
|-----------------|------------------------------------------------------------------|
| **clang-tidy**  | Análise estática — bugs, padrões problemáticos. Não modifica.    |
| **clang-format**| Formatação automática in-place.                                  |

Análise estática com clang-tidy em todo o `core/`:

```bash
make tidy
```

Análise em um arquivo específico:

```bash
make tidy-file FILE=core/...cpp
```

Formatação in-place de todo o `core/`:

```bash
make format
```

Verificação de formatação sem modificar (CI):

```bash
make format-check
```

Formatação de um arquivo específico:

```bash
make format-file FILE=core/...cpp
```

Exibe diff sem aplicar:

```bash
make format-diff FILE=core/...cpp
```

O `.clang-tidy` define três perfis — edite o arquivo para trocar:

| Perfil        | Quando usar                                              |
|---------------|----------------------------------------------------------|
| **Permissivo**| Bugs reais e undefined behavior. Ponto de partida.       |
| **Moderado**  | Após zerar avisos do permissivo. Adiciona `modernize-*`. |
| **Estrito**   | Meta para PRs. Adiciona `readability-*`.                 |

Com a extensão **clangd** ou **C/C++** no VS Code e `build/compile_commands.json` gerado, as regras são aplicadas em tempo real.

---

## Dependências gerenciadas pelo Conan {#dependencias}

| Dependência   | Versão     | Uso                               |
|---------------|------------|-----------------------------------|
| `grpc`        | 1.54.3     | Transporte gRPC                   |
| `protobuf`    | 3.21.12    | Serialização de mensagens         |
| `onnxruntime` | 1.18.1     | Backend de inferência ONNX        |
| `gtest`       | 1.14.0     | Framework de testes C++           |
| `abseil`      | 20230802.1 | Utilitários (dependência do gRPC) |
| `re2`         | 20230301   | Regex (dependência do gRPC)       |

### Opções de build

| Opção        | Tipo    | Padrão  | Descrição                                        |
|--------------|---------|---------|--------------------------------------------------|
| `enable_gpu` | boolean | `false` | Habilita CUDA Execution Provider no ONNX Runtime |

---

## Targets do Makefile {#makefile}

| Target               | Descrição                                                  |
|----------------------|------------------------------------------------------------|
| `configure`          | Instala dependências Conan e configura Meson               |
| `build`              | Compila todos os alvos                                     |
| `install`            | Instala em `../dist/`                                      |
| `package`            | Cria pacote Conan (Debug + Release)                        |
| `clean`              | Remove `build/` e cache de subprojetos                     |
| `run-server`         | Inicia o servidor AsaMiia localmente                       |
| `run-client`         | Executa o cliente de exemplo CLI                           |
| `test`               | Executa todos os testes                                    |
| `test-unit`          | Executa apenas testes unitários (sem servidor)             |
| `test-integration`   | Executa testes de integração (servidor opcional)           |
| `test-verbose`       | Executa testes com saída detalhada                         |
| `coverage`           | Gera relatório HTML de cobertura em `build/coverage-report/` |
| `coverage-open`      | Abre o relatório no browser                                |
| `python-setup`       | Instala Python 3.12 se não disponível                      |
| `python-env`         | Cria o venv Python em `models/.venv`                       |
| `python-install`     | Instala dependências Python                                |
| `create-models`      | Gera modelos ONNX de teste e copia modelos Python          |
| `python-clean`       | Remove o venv Python                                       |
| `docs`               | Gera documentação Doxygen em `docs/doxygen/generated/`     |
| `docs-open`          | Abre a documentação no browser                             |
| `docs-clean`         | Remove a documentação gerada                               |
| `docker-build-cpu`   | Constrói imagem Docker CPU                                 |
| `docker-build-gpu`   | Constrói imagem Docker GPU                                 |
| `docker-run-cpu`     | Executa container Docker CPU                               |
| `docker-run-gpu`     | Executa container Docker GPU                               |
| `tidy`               | Analisa código com clang-tidy                              |
| `tidy-file`          | Analisa `FILE=` específico com clang-tidy                  |
| `format`             | Formata todo o código fonte in-place                       |
| `format-check`       | Verifica formatação sem modificar (CI)                     |
| `format-file`        | Formata `FILE=` específico                                 |
| `format-diff`        | Mostra diff do que seria formatado                         |

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
│   ├── doxygen/
│   │   └── Doxyfile                    ← configuração Doxygen
│   └── diagrams/                       ← diagramas draw.io
├── conanfile.py                        ← dependências e build Conan
├── meson.build                         ← build principal
├── meson_options.txt                   ← opções (enable_gpu)
└── Makefile                            ← targets de conveniência
```

---

## Visão geral da arquitetura {#arquitetura}

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

### Diagrama de classes

![Diagrama de Classes](docs/images/classes_diagram.png)

### Diagramas de sequência

**Modo 1 — In-process + Python**

![Diagrama de Sequência — In-process + Python](docs/images/predict_inprocess_python.png)

**Modo 2 — In-process + ONNX**

![Diagrama de Sequência — In-process + ONNX](docs/images/predict_inprocess_onnx.png)

**Modo 3 — gRPC + Python**

![Diagrama de Sequência — gRPC + Python](docs/images/predict_grpc_python.png)

**Modo 4 — gRPC + ONNX**

![Diagrama de Sequência — gRPC + ONNX](docs/images/predict_grpc_onnx.png)
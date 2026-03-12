# Guia do Usuário — poc-miia (AsaMiia)

> Sistema de Inferência de Machine Learning via gRPC + ONNX Runtime

---

## O que é o poc-miia?

O poc-miia é composto por duas partes que trabalham juntas:

**Servidor (Worker):** um processo que carrega modelos ONNX em memória e responde requisições de inferência via gRPC. O executável se chama `AsaMiia`.

**Biblioteca Cliente (`libasa_miia_client`):** uma biblioteca C++ que você inclui no seu projeto para se comunicar com o servidor. Ela encapsula toda a comunicação gRPC e expõe uma API simples.

A comunicação entre cliente e servidor acontece via gRPC na porta `50052` (padrão). O cliente pode estar no mesmo host que o servidor ou em outro host/container.

```
┌──────────────────────────┐          gRPC (TCP)          ┌──────────────────────────┐
│   Sua Aplicação C++      │  ───── localhost:50052 ────> │   AsaMiia (Worker)       │
│                          │                              │                          │
│   InferenceClient        │                              │   InferenceEngine        │
│   .connect()             │                              │   ONNX Runtime           │
│   .load_model()          │                              │   Modelos em memória     │
│   .predict()             │                              │                          │
└──────────────────────────┘                              └──────────────────────────┘
```

---

## 1. Instalação (única vez)

Antes de usar o sistema, você precisa compilar e instalar o poc-miia. Esse processo é feito **uma única vez**.

O sistema utiliza **Meson** como build system e **Conan** para gerenciamento de dependências C++. Certifique-se de que ambos estão instalados antes de prosseguir.


### Passo a passo

Execute os comandos abaixo na ordem exata. Cada comando depende do anterior.

```bash
# 1. Clonar o repositório
cd ~/asa
git clone <url-repositorio>
cd poc-miia

# 2. [OPCIONAL] Gerar modelos ONNX de teste
make create-models

# 3. Configurar build system (Conan instala deps, Meson configura projeto)
make configure

# 4. Compilar o projeto
make build

# 5. Instalar binários em ~/asa/dist/
make install

# 6. Empacotar como pacote Conan (necessário para usar a lib no seu projeto)
make package
```

**Tempo total estimado:** 20–30 minutos na primeira execução (a maior parte é o build de dependências pelo Conan).

Após a instalação, o executável do servidor estará em `~/asa/dist/bin/AsaMiia` e a biblioteca cliente estará disponível como pacote Conan `asa-poc-miia/1.0.0`.

> **Atenção:** todas as dependências (gRPC, protobuf, ONNX Runtime) são gerenciadas exclusivamente pelo Conan. Não misture pacotes do sistema com pacotes Conan — isso causa falhas de compilação.

---

## 2. Rodando o Servidor

O servidor pode rodar de duas formas: diretamente no host ou dentro de um container Docker.

### 2.1. No host (sem Docker)

Abra um terminal e execute:

```bash
cd ~/asa/dist
./bin/AsaMiia --address 0.0.0.0:50052 --threads 8
```

Ou, se preferir usar o Makefile a partir do diretório do projeto:

```bash
cd ~/asa/poc-miia
make run-worker
```

O servidor ficará bloqueante no terminal, imprimindo logs. Mantenha esse terminal aberto.

**Onde colocar os modelos:** os arquivos `.onnx` devem estar em `~/asa/dist/models/` (se rodando a partir de `~/asa/dist`) ou em `~/asa/poc-miia/models/` (se rodando via `make run-worker`). O comando `make install` copia automaticamente os modelos da pasta `models/` do projeto para `~/asa/dist/models/`.

### 2.2. Em container Docker

#### Build da imagem (uma única vez, ~20 minutos)

```bash
cd ~/asa/poc-miia

# Para uso apenas com CPU:
make docker-build-cpu

# Para uso com GPU (requer NVIDIA Docker):
make docker-build-gpu
```

A imagem só precisa ser reconstruída se o código-fonte do servidor mudar. Alterações nos modelos **não** requerem rebuild.

#### Executar o container

```bash
cd ~/asa/poc-miia

# CPU:
make docker-run-cpu

# GPU:
make docker-run-gpu
```

O Makefile já passa os argumentos `--address 0.0.0.0:50052 --threads 8` automaticamente e mapeia a porta `50052`.

**Onde colocar os modelos (Docker):** coloque os arquivos `.onnx` na pasta `models/` do projeto (ou seja, `~/asa/poc-miia/models/`). Essa pasta é montada automaticamente no container em `/app/models/`. Modificações nessa pasta **não requerem rebuild** da imagem — basta reiniciar o container.

Se preferir rodar manualmente sem o Makefile:

```bash
# CPU
docker run --rm -p 50052:50052 \
    -v ./models:/app/models:ro \
    ml-inference-worker:latest-cpu

# GPU
docker run --rm --gpus all -p 50052:50052 \
    -v ./models:/app/models:ro \
    ml-inference-worker:latest-gpu
```

---

## 3. Testando o Servidor com o Client CLI

Antes de integrar a biblioteca no seu projeto, você pode verificar rapidamente se o servidor está funcionando usando o **Client CLI** — um executável de linha de comando incluído no poc-miia que exercita todas as operações da API.

### 3.1. Rodando o Client CLI

```bash
# ~/asa/poc-miia
cd ~/asa/poc-miia
make run-client

# ou ~/asa/dist
cd ~/asa/dist
./bin/AsaMiiaClient localhost:50052
```

Para apontar para um servidor em outro host:

```bash
./<binario> 192.168.1.100:50052
```

### 3.2. O que o Client CLI faz

O Client CLI executa o seguinte fluxo automaticamente:

1. Conecta ao servidor e verifica saúde (`health_check`)
2. Carrega um modelo ONNX de teste
3. Lista os modelos carregados
4. Executa uma inferência única com dados gerados automaticamente
5. Executa um batch de 3 inferências
6. Exibe métricas do servidor (uptime, total de requests, etc.)
7. Executa um stress test de 10 requests e mede throughput
8. Descarrega o modelo

---

## 4. Conectando o Cliente ao Servidor (no seu projeto)

### 4.1. Incluindo no seu código C++

Adicione o header e crie uma instância do cliente apontando para o endereço do servidor:

```cpp
#include <asa-poc-miia/client/inference_client.hpp>

// Servidor local:
mlinference::client::InferenceClient client("localhost:50052");

// Servidor remoto (outra máquina ou container):
mlinference::client::InferenceClient client("192.168.1.100:50052");
```

### 4.2. Configurando seu projeto para usar a biblioteca

Seu projeto precisa declarar `asa-poc-miia/1.0.0` como dependência Conan. A estrutura mínima é:

```
meu-projeto/
├── conanfile.py      # Declara dependência asa-poc-miia
├── meson.build       # Configura o build
├── Makefile          # Automação
└── src/
    ├── meson.build
    └── main.cpp
```

No `conanfile.py`, declare a dependência:

```python
def requirements(self):
    self.requires("asa-poc-miia/1.0.0")
```

No `meson.build` raiz, encontre a dependência via pkg-config:

```meson
asa_poc_miia_dep = dependency('asa-poc-miia')
```

E no `src/meson.build`, vincule ao executável:

```meson
executable('meu-projeto', sources, dependencies: [asa_poc_miia_dep])
```

> **Importante:** o pacote `asa-poc-miia/1.0.0` precisa existir no cache local do Conan. Se ainda não existir, execute `make package` no diretório do poc-miia.

---

## 5. Usando a API do Cliente

### 5.1. Fluxo básico completo

```cpp
#include <asa-poc-miia/client/inference_client.hpp>
#include <iostream>

int main() {
    // 1. Criar cliente e conectar
    mlinference::client::InferenceClient client("localhost:50052");

    if (!client.connect()) {
        std::cerr << "Falha ao conectar no worker" << std::endl;
        return 1;
    }

    // 2. Carregar modelo (caminho relativo ao WORKER, não ao cliente)
    client.load_model("simple_linear", "./models/simple_linear.onnx");

    // 3. Preparar entrada: mapa nome_do_tensor → vetor de floats
    std::map<std::string, std::vector<float>> inputs;
    inputs["input"] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // 4. Executar predição
    auto result = client.predict("simple_linear", inputs);

    if (result.success) {
        std::cout << "Inferência em " << result.inference_time_ms << " ms" << std::endl;
        for (auto& [name, data] : result.outputs) {
            std::cout << name << ": ";
            for (float v : data) std::cout << v << " ";
            std::cout << std::endl;
        }
    } else {
        std::cerr << "Erro: " << result.error_message << std::endl;
    }

    // 5. Descarregar modelo quando não precisar mais
    client.unload_model("simple_linear");
    return 0;
}
```

### 5.2. Referência rápida de métodos

```cpp
// ── Conexão ──────────────────────────────────────────────────────────────────

InferenceClient client("localhost:50052");  // endereço no formato "host:porta"

client.connect()        → bool   // abre canal gRPC (timeout 5s); true = conectado
client.is_connected()   → bool   // verifica se o canal está ativo

// ── Ciclo de vida dos modelos ─────────────────────────────────────────────────

client.load_model(model_id, model_path)           → bool
client.load_model(model_id, model_path, version)  → bool
//   model_path é relativo ao SERVIDOR, não ao cliente
//   version é opcional — padrão: "1.0.0"

client.unload_model(model_id)  → bool

// ── Inferência ───────────────────────────────────────────────────────────────

// inputs = mapa  nome_do_tensor → vetor de floats
client.predict(model_id, inputs)  → PredictionResult

// batch: um PredictionResult por entrada, na mesma ordem
client.batch_predict(model_id, batch_inputs)  → vector<PredictionResult>

// ── Introspecção ─────────────────────────────────────────────────────────────

client.list_models()                   → vector<ModelInfo>  // modelos carregados com metadados completos
client.get_model_info(model_id)        → ModelInfo          // metadados de um modelo específico
client.list_available_models()         → vector<AvailableModel> // arquivos .onnx no servidor
client.list_available_models(dir)      → vector<AvailableModel> // idem, em diretório específico
client.validate_model(model_path)      → ValidationResult   // valida sem carregar
client.warmup_model(model_id)          → WarmupResult       // aquece com 1 run
client.warmup_model(model_id, n_runs)  → WarmupResult       // aquece com n runs

// ── Monitoramento ────────────────────────────────────────────────────────────

client.health_check()  → bool          // true se o servidor está saudável
client.get_status()    → WorkerStatus  // métricas e capacidades do servidor
```

**Tipos de retorno:**

```cpp
struct PredictionResult {
    bool success;
    std::map<std::string, std::vector<float>> outputs;  // nome_tensor → valores
    double inference_time_ms;                           // tempo no servidor (ms)
    std::string error_message;                          // preenchido se !success
};

struct ModelInfo {
    std::string model_id;
    std::string version;
    std::string backend;           // ex: "onnx", "python"
    std::string description;
    std::string author;
    uint64_t    memory_usage_bytes;
    bool        is_warmed_up;

    struct TensorSpec {
        std::string name;
        std::string dtype;          // ex: "float32", "int64"
        std::vector<int64_t> shape; // -1 indica dimensão dinâmica
        std::string description;
    };
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
    std::map<std::string, std::string> tags;
};

struct AvailableModel {
    std::string filename;           // "resnet18.onnx"
    std::string path;               // caminho completo no servidor
    std::string extension;          // ".onnx"
    std::string backend;            // "onnx"
    uint64_t    file_size_bytes;
    bool        is_loaded;          // já está em memória?
    std::string loaded_as;          // model_id, se is_loaded == true
};

struct ValidationResult {
    bool valid;
    std::string backend;
    std::string error_message;
    std::vector<std::string>          warnings;
    std::vector<ModelInfo::TensorSpec> inputs;
    std::vector<ModelInfo::TensorSpec> outputs;
};

struct WarmupResult {
    bool     success;
    uint32_t runs_completed;
    double   avg_time_ms;
    double   min_time_ms;
    double   max_time_ms;
    std::string error_message;
};

struct WorkerStatus {
    std::string worker_id;
    uint64_t total_requests;
    uint64_t successful_requests;
    uint64_t failed_requests;
    uint32_t active_requests;           // requisições em andamento agora
    int64_t  uptime_seconds;
    std::vector<std::string> loaded_models;
    std::vector<std::string> supported_backends;  // ex: ["onnx", "python"]
};
```

### 5.3. Verificação rápida de conexão

Antes de integrar no seu projeto, você pode testar se o servidor está respondendo:

```cpp
mlinference::client::InferenceClient client("localhost:50052");
client.connect();
bool ok = client.health_check();  // true se o worker respondeu
```

---

## 6. Caminhos dos Modelos — Atenção!

Esse é um ponto que causa confusão. O caminho passado em `load_model()` é relativo ao **servidor**, não ao seu código cliente.

```cpp
client.load_model("meu_modelo", "./models/meu_modelo.onnx");
```

Se você receber o erro `"Load model failed"`, verifique se o caminho está correto **do ponto de vista do servidor**, não do cliente.

---

## 7. Resumo de Comandos do Makefile

Todos os comandos abaixo devem ser executados a partir de `~/asa/poc-miia/`.

**Build e instalação (uma vez):**

| Comando | O que faz |
|---------|-----------|
| `make create-models` | Gera modelos ONNX de teste em `models/` |
| `make configure` | Instala deps (Conan) e configura build (Meson) |
| `make build` | Compila tudo |
| `make install` | Instala binários em `~/asa/dist/bin` e copia modelos |
| `make package` | Cria pacote Conan para uso em outros projetos |
| `make clean` | Remove artefatos de build |

**Execução e testes:**

| Comando | O que faz |
|---------|-----------|
| `make run-worker` | Roda o servidor localmente |
| `make run-client` | Roda o Client CLI de testes |
| `make test` | Executa a suíte de testes automatizados |

**Docker:**

| Comando | O que faz |
|---------|-----------|
| `make docker-build-cpu` | Builda imagem Docker (CPU) |
| `make docker-build-gpu` | Builda imagem Docker (GPU) |
| `make docker-run-cpu` | Roda container Docker (CPU) |
| `make docker-run-gpu` | Roda container Docker (GPU) |
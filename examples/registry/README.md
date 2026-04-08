# AIStor Free — Setup Local

## Estrutura

```
.
├── docker-compose.yml        # Sobe AIStor + inicialização
├── .env.example              # Template de credenciais
├── .env                      # Suas credenciais reais (não versionar)
├── .gitignore
├── minio.license             # Licença AIStor Free (não versionar)
├── init/
│   ├── policy-shared.json    # Bucket compartilhado (leitura e escrita)
│   ├── policy-readonly.json  # Bucket somente leitura
│   ├── policy-personal.json  # Pasta pessoal por usuário
│   └── create-user.sh        # Script para criação de usuários via terminal
└── minio-data/               # Dados persistidos (criado automaticamente)
```

## Pré-requisito — Licença AIStor Free

O AIStor Free requer um arquivo de licença gratuito:

1. Acesse [min.io/pricing](https://min.io/pricing) e registre-se (gratuito)
2. Baixe o arquivo de licença
3. Salve como `./minio.license` na raiz do projeto

## Início rápido

```bash
# 1. Copiar e editar credenciais
cp .env.example .env
# edite .env com suas credenciais

# 2. Subir os serviços
docker compose up -d

# 3. Acessar o console web
# http://localhost:9001
# Login: valores definidos em MINIO_ROOT_USER / MINIO_ROOT_PASSWORD
```

## Credenciais

| Tipo         | Usuário  | Senha              | Uso                          |
|--------------|----------|--------------------|------------------------------|
| Root (admin) | `admin`  | definido no `.env` | Console web e administração  |

> ⚠️ Somente o admin pode criar novos usuários — via console web ou via `create-user.sh`.

## Buckets e policies

O `minio-init` cria automaticamente os buckets e registra as policies no primeiro `docker compose up`:

| Bucket    | Policy              | Quem pode ler | Quem pode escrever/deletar        |
|-----------|---------------------|---------------|-----------------------------------|
| `shared`  | `policy-shared`     | Todos         | Todos                             |
| `readonly`| `policy-readonly`   | Todos         | Somente admin                     |
| `users`   | `policy-personal`   | Cada um na própria pasta | Cada um na própria pasta |

### Isolamento de pastas pessoais

A `policy-personal` usa a variável `${aws:username}` do IAM — cada usuário acessa apenas `users/<seu-username>/`. Um usuário `lucas` não consegue ver nem escrever em `users/joao/`, mesmo que ambos tenham a mesma policy vinculada.

## Criação de usuários

### Via console web (recomendado)

1. **Administrator → Identity → Users → Create User**
2. Preencha username e password
3. Em **Assign Policies**, selecione as 3 policies: `policy-shared`, `policy-readonly`, `policy-personal`
4. Clique **Save**

> A pasta pessoal `users/<username>/` é criada automaticamente no primeiro upload do usuário.

### Via terminal

O `mc` não está disponível no container do servidor — o comando é executado em um container `mc` temporário:

```bash
make user-create USER=lucas PASS=senha123
```

Ou manualmente:

```bash
docker run --rm \
  --network host \
  -v ./init:/init \
  quay.io/minio/aistor/mc:latest \
  /bin/sh /init/create-user.sh <username> <password>
```

O script `create-user.sh` cria o usuário, materializa a pasta pessoal e vincula as 3 policies automaticamente.

## Uso com boto3 (Python)

```python
import boto3
from botocore.client import Config

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="<access-key>",
    aws_secret_access_key="<secret-key>",
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)
```

> As credenciais de acesso programático (`access-key` / `secret-key`) são geradas por **Administrator → Identity → Users → [usuário] → Service Accounts → Create Access Key** no console web.

## Uso com AWS SDK C++

```cpp
config.endpointOverride = "localhost:9000";
Aws::Auth::AWSCredentials credentials("<access-key>", "<secret-key>");
// useVirtualAddressing = false  (path-style, obrigatório para MinIO)
```

## Parar / remover

```bash
docker compose down        # Para os containers (dados preservados)
docker compose down -v     # Para e remove volumes (dados perdidos)
```
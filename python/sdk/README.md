# AsaMiia — Desenvolvimento de Modelos

SDK minimalista para desenvolver e testar modelos de inferência compatíveis com o servidor
**AsaMiia**, que integra modelos Python ao simulador construtivo **ASA/MIXR**.

---

## Estrutura

```
dev/
├── pilot_model.py      # Classes base — não editar
├── pilotBT.py          # ← SEU ARQUIVO: implemente predict() e get_schema() aqui
├── state_example.json  # ← Input de exemplo para testes e warmup — edite à vontade
├── test_model.py       # Validador fornecido pelo servidor
├── requirements.txt    # Dependências Python
└── Makefile            # Configuração de Ambiente e execução
```

---

## Get started

### 1. Instalar dependências e criar o ambiente

```bash
make install
```

Instala Python 3.12 (se necessário), cria `.venv/` e instala os pacotes de
`requirements.txt`. Precisa ser feito apenas uma vez.

### 2. Rodar o modelo com o input de exemplo

```bash
make run
```

Imprime o estado de entrada completo (JSON) e os cinco valores de saída do
`predict()`.  Use para checar rapidamente se o código não quebra.

### 3. Validar o modelo

```bash
make test
```

Executa `test_model.py` contra `pilotBT.py` e reporta erros de schema, shapes
incorretos ou exceções no `predict()`.

### 4. Outros comandos

| Comando | O que faz |
|---|---|
| `make install` | Cria `.venv` e instala dependências |
| `make run` | Executa `pilotBT.py` com o input de exemplo |
| `make test` | Valida `pilotBT.py` contra o schema do servidor |
| `make clean` | Remove `.venv` e caches |

> Para rodar scripts manualmente, ative o ambiente antes:
> ```bash
> source .venv/bin/activate
> python pilotBT.py
> ```

---

## Input de exemplo (`state_example.json`)

`make_dummy_input()` lê o arquivo `state_example.json` na raiz do projeto.
Ele representa uma aeronave em cruzeiro a 6 000 m, CAP ativa e um track
inimigo a 45 km — um cenário representativo para testes e warmup.

**Edite-o livremente** para testar seu modelo em outras situações: altitude
diferente, múltiplos tracks, combustível baixo, etc. O arquivo não é
versionado junto com a lógica do modelo e serve apenas como um estado de
exemplo para desenvolvimento local.

---

## O que você deve implementar

**Abra `pilotBT.py` e implemente dois métodos:**

### `predict()` — implemente a lógica

```python
def predict(self, inputs: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
    own = inputs["ownship"]
    # ...

    # sua lógica aqui

    return {
        # uma chave por saída declarada em get_schema()
        "airspeed":  np.array([[...]], dtype=np.float32),
        # ...
    }
```

### `get_schema()` — declare suas saídas

As entradas já estão fixas na classe base. Você só precisa declarar o que
seu `predict()` vai retornar, usando os helpers prontos:

```python
def get_schema(self) -> ModelSchema:
    return self._base_schema(
        outputs=[
            self.output_spec("airspeed", description="Velocidade [m/s]"),
            # ...
        ],
        description="Meu modelo de piloto",
    )
```

As saídas podem ser qualquer coisa que faça sentido para o seu caso de uso: 
um vetor de features para processamento no ASA/MIXR, uma ação discreta,
resultados intermediários, etc. O shape padrão `[1, 1]` é convertido para
escalar pelo servidor; use `shape=[1, N]` para vetores.

---

## Arquitetura do modelo

`miia_model.py` define duas classes em sequência. Entender o papel de cada
uma evita surpresas.

```
MiiaModel        — contrato com o servidor      ┐
    └── PilotBTModel — base do piloto           ┤ pilot_model.py (não editar)
            └── PilotBT — sua implementação     ┘ pilotBT.py     (edite aqui)
```

### `MiiaModel` — o contrato com o servidor

Define os três métodos que **todo** modelo AsaMiia precisa ter:

- **`load()`** — chamado uma vez quando o servidor carrega o modelo.
- **`predict(inputs)`** — chamado a cada ciclo de reasoning.
- **`get_schema()`** — descreve entradas e saídas via `TensorSpec`/`ModelSchema`.

Também fornece implementações padrão de `validate_inputs()`, `warmup()` e
`unload()`.

O servidor faz exatamente isso ao carregar `pilotBT.py`:

```python
model = PilotBT()     # sem argumentos — __init__ não pode ter params obrigatórios
model.load()          # uma vez
model.predict(state)  # a cada ciclo
```

> **Como o servidor localiza a classe:** varre o dict do módulo em busca da
> primeira classe **concreta** (não abstrata) que herde de `MiiaModel`.
> `PilotBTModel` é ignorada automaticamente porque ainda possui `predict()` e 
> `get_schema()` abstratos — só `PilotBT` é instanciada.

### `PilotBTModel` — a base do piloto

Especialização de `MiiaModel` para o agente `AsaPilotBT`. Já implementa:

- **`_INPUTS`** — as 14 entradas estruturadas fixas (`sim`, `ownship`, `nav`,
  `air_tracks`, etc.), compartilhadas por todos os modelos deste agente.
- **`_base_schema(outputs)`** — monta o `ModelSchema` completo com as entradas
  fixas + as saídas que você declarar.
- **`output_spec(name, ...)`** — helper para criar um `TensorSpec` de saída
  de forma concisa.
- **`warmup()`** — usa `make_dummy_input()` para rodar inferências de
  aquecimento com estado realista.
- **`make_dummy_input()`** — estado completo de exemplo: aeronave em cruzeiro
  a 6 000 m, CAP ativa, um track inimigo a 45 km.
- **`load()` e `unload()`** — implementações vazias prontas para sobrescrever.

`predict()` e `get_schema()` permanecem abstratos aqui — ambos precisam ser
implementados em `PilotBT`.

### `PilotBT` — sua implementação

Herda `PilotBTModel` e precisa implementar `predict()` e `get_schema()`. Se
o modelo precisar de inicialização (pesos, redes neurais, arquivos externos),
sobrescreva `load()` também:

```python
class PilotBT(PilotBTModel):

    def load(self) -> None:
        # opcional — carregue recursos pesados aqui
        import pickle
        with open("meu_modelo.pkl", "rb") as f:
            self._clf = pickle.load(f)


    def predict(self, inputs):
        own = inputs["ownship"]
        # use self._clf aqui
        action = self._clf.predict(...)
        return {"action_id": np.array([[action]], dtype=np.float32)}

    def get_schema(self) -> ModelSchema:
        return self._base_schema(
            outputs=[
                self.output_spec("action_id", description="Ação selecionada"),
            ],
            description="Meu modelo com saída customizada",
        )
```

### Por que `structured=True` no schema?

As entradas são dicionários aninhados (`ownship`, `air_tracks`,
`shared.formation`, etc.), não tensores planos. O flag `structured=True` em
`TensorSpec` instrui o servidor a não tentar validar esses campos como arrays
— eles chegam como `dict` Python diretamente no `predict()`.

As saídas, por outro lado, são tensores de shape `[1, 1]` — o servidor extrai
o valor numérico escalar e envia ao simulador.

---

## Campos disponíveis no input (`predict()`)

```python
def predict(self, inputs):
    sim   = inputs.get("sim", {})           # fase e tempo de simulação
    own   = inputs.get("ownship", {})       # estado da própria aeronave
    area  = inputs.get("game_area", {})     # limites da área de jogo
    terr  = inputs.get("territory", {})     # limites do território
    faor  = inputs.get("faor", {})          # área de responsabilidade
    inv   = inputs.get("inventory", {})     # combustível e armamento
    shrd  = inputs.get("shared", {})        # formação, CAP e atribuições
    nav   = inputs.get("nav", {})           # waypoint ativo e navegação
    tdata = inputs.get("terrain_data", {})  # elevação do terreno à frente
    radar = inputs.get("radar", {})         # modo atual do radar
    trks  = inputs.get("air_tracks", [])    # tracks radar/datalink/RWR
    gatk  = inputs.get("gnd_atk", {})       # dados de ataque ao solo
    evts  = inputs.get("events", {})        # eventos do ciclo atual
    radio = inputs.get("radio", {})         # mensagem de rádio recebida
    bb    = inputs.get("bt_blackboard", {}) # blackboard interno da BT
    # ... 
```

Consulte as docstrings de `PilotBTModel.get_schema()` e
`PilotBTModel.make_dummy_input()` em `miia_model.py` para a lista completa
de campos com tipos e unidades.

> Modificações no método AsaPilotBT::prepereState() (`asa-models-r`) devem ser refletidas no método `make_dummy_input()`, atributo `_INPUTS`, arquivo `state_example.json`, bem como nas > informações acessadas no método `predict()`.

---

### Convenções

- Vetores 3D (`pos_ecef`, `vel_body`, etc.) chegam como listas `[x, y, z]`.
- Listas de estruturas (`air_tracks`, `formation.aircraft`) chegam como listas de dicts.
- Verifique o campo `valid` de cada grupo (quando disponível) antes de usar seus dados.

---

## Debugging durante o desenvolvimento

`PilotBT` expõe `self.log()` para escrever mensagens no arquivo de log
configurado no `__init__` (default: `./logs/pilotBT.log`). O comportamento é idêntico ao `print()`:

```python
self.log("alt:", alt, "hdg:", hdg)
self.log(f"foe range={tgt_rng:.0f}m brg={tgt_brg:.1f}°")
self.log("inputs completos:", json.dumps(inputs, default=str, indent=4))
```

## Adicionando dependências

```bash
echo "scipy>=1.13" >> requirements.txt
make clean
make install
```
Ou modifique `requirements.txt` manualmente e depois rode `make clean` e `make install` para garantir que o ambiente seja recriado com as novas dependências.

## Entrega

Ao finalizar o desenvolvimento, repasse aos desenvolvedores da integração
apenas os arquivos:

- `pilotBT.py`
- `requirements.txt`

Os demais arquivos (`pilot_model.py`, `state_example.json`, `test_model.py`,
`Makefile`) fazem parte do SDK de desenvolvimento e não devem ser incluídos
na entrega.
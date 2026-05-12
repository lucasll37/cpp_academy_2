# MIIA — Desenvolvimento de Modelos

SDK minimalista para desenvolver e testar modelos de inferência compatíveis com o servidor
**MIIA**, que integra modelos Python ao simulador construtivo **ASA/MIXR**.

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

Define os três métodos que **todo** modelo precisa ter:

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

## Integração do MIIA no projeto do simulador

Esta seção descreve como incorporar o modelo desenvolvido em um agente C++ do
ASA/MIXR usando a biblioteca **Miia**.

### 1. Declarar a dependência

No `conanfile.py` do projeto do simulador:

```python
# conanfile.py
def requirements(self):
    self.requires("miia/1.0.0")
```

No `meson.build`:

```cpp
# meson.build
miia_dep = dependency('miia', method: 'pkg-config')

executable('meu_simulador',
    'src/main.cpp',
    dependencies: [miia_dep],
)
```

### 2. Incluir o header nos arquivos *.hpp e *.cpp onde o modelo será usado

```cpp
#include "miia/client/inference_client.hpp"
```

### 3. Declarar membros no agente

```cpp
// Método que serializa o estado da simulação para envio ao modelo
const mlinference::client::Object prepareState(AsaPilotState* const);

std::unique_ptr<mlinference::client::InferenceClient> client;

const std::string server{"inprocess"}; // "localhost:50052" | "inprocess"
const std::string model_id{"pilotBT"};
const std::string model_path{"../models/pilotBT.py"};
```

O modo `"inprocess"` embute o mecanismo de inferência diretamente no processo do
simulador — sem servidor externo. Use `"localhost:50052"` (ou outro endereço)
para o modo gRPC com inferência separado no processo do servidor.

### 4. Implementar `prepareState()` para converter o estado do simulador no formato esperado pelo modelo

```cpp
const mlinference::client::Object AsaPilotBT::prepareState(AsaPilotState* const s)
{
    using mlinference::client::Array;
    using mlinference::client::Object;
    using mlinference::client::Value;

    // ── helpers locais ────────────────────────────────────────────────────────

    auto vec3 = [](const mixr::base::Vec3d& v)
    {
        Array a;
        a.push_back(Value{v[0]});
        a.push_back(Value{v[1]});
        a.push_back(Value{v[2]});
        return Value{std::move(a)};
    };

    // Serializa um AirTrackInfo* completo; valid=0 se ponteiro nulo
    auto serialize_track = [&](const AirTrackInfo* t) -> Value
    {
        Object o;
        if (t == nullptr)
        {
            o["valid"] = Value{false};
            return Value{std::move(o)};
        }
        o["valid"] = Value{true};
        o["iff_code"] = Value{t->iffCode};
        o["type"] = Value{static_cast<double>(t->type)};
        o["player_id"] = Value{static_cast<double>(t->playerID)};
        o["latitude"] = Value{t->latitude};
        o["longitude"] = Value{t->longitude};
        o["altitude"] = Value{t->altitude};
        o["rel_azimuth"] = Value{t->relAzmth};
        o["true_azimuth"] = Value{t->trueAzmth};
        o["elevation"] = Value{t->elev};
        o["range"] = Value{t->range};
        o["gnd_range"] = Value{t->gndRange};
        o["gnd_track"] = Value{t->gndTrk};
        o["gnd_speed"] = Value{t->gndSpd};
        o["age"] = Value{t->age};
        o["wez_max_o2t_true"] = Value{t->wezMaxOwn2TrkTrue};
        o["wez_nez_o2t_true"] = Value{t->wezNezOwn2TrkTrue};
        o["wez_max_t2o_true"] = Value{t->wezMaxTrk2OwnTrue};
        o["wez_nez_t2o_true"] = Value{t->wezNezTrk2OwnTrue};
        o["wez_max_t2o_pred"] = Value{t->wezMaxTrk2OwnPred};
        o["wez_nez_t2o_pred"] = Value{t->wezNezTrk2OwnPred};
        o["wez_pred_dist_corr"] = Value{t->wezPredDistCorrection};
        o["wez_nez_o2t_pred"] = Value{t->wezNezOwn2TrkPred};
        o["has_spike"] = Value{t->hasSpike};
        o["emitter_class"] = Value{t->emitterClass};
        o["emitter_mode"] = Value{t->emitterMode};

        Array pe;
        pe.push_back(Value{t->posECEF[0]});
        pe.push_back(Value{t->posECEF[1]});
        pe.push_back(Value{t->posECEF[2]});
        o["pos_ecef"] = Value{std::move(pe)};

        Array ve;
        ve.push_back(Value{t->velECEF[0]});
        ve.push_back(Value{t->velECEF[1]});
        ve.push_back(Value{t->velECEF[2]});
        o["vel_ecef"] = Value{std::move(ve)};

        return Value{std::move(o)};
    };

    // ── 1. sim ────────────────────────────────────────────────────────────────
    Object sim;
    sim["valid"] = Value{s->getHasValidSimData()};
    sim["phase"] = Value{static_cast<double>(s->getSimPhase())};
    sim["exec_time"] = Value{s->getExecTime()};
    sim["sim_time"] = Value{s->getSimTime()};

    // ── 2. ownship ────────────────────────────────────────────────────────────
    Object own;
    own["valid"] = Value{s->getHasValidPlayerData()};
    own["latitude"] = Value{s->getLatitude()};
    own["longitude"] = Value{s->getLongitude()};
    own["altitude"] = Value{s->getAltitude()};
    own["ground_speed"] = Value{s->getGroundSpeed()};
    own["ground_track"] = Value{s->getGroundTrack()};
    own["heading"] = Value{s->getHeadingD()};
    own["pitch"] = Value{s->getPitchD()};
    own["roll"] = Value{s->getRollD()};
    own["terrain_elev"] = Value{s->getTerrainElevation()};
    own["damage"] = Value{s->getDamage()};
    own["pos_ecef"] = vec3(s->getGeocPosition());
    own["vel_ecef"] = vec3(s->getGeocVelocity());
    own["vel_body"] = vec3(s->getVelocityBody());
    own["accel_ecef"] = vec3(s->getGeocAcceleration());
    own["accel_body"] = vec3(s->getAccelerationBody());
    own["euler_ned"] = vec3(s->getEulerAngles());
    own["euler_ecef"] = vec3(s->getGeocEulerAngles());
    own["ang_vel_body"] = vec3(s->getAngularVelocities());
    own["ang_vel_ecef"] = vec3(s->getGeocAngularVelocities());

    // ── 3. game_area ─────────────────────────────────────────────────────────
    Object game_area;
    game_area["valid"] = Value{s->getHasValidGameAreaInfo()};
    game_area["boundary_dist"] = Value{s->getGameAreaBoundaryDist()};
    game_area["brg_to_center"] = Value{s->getBrgToGameAreaCenter()};

    // ── 4. territory ─────────────────────────────────────────────────────────
    Object territory;
    territory["valid"] = Value{s->getHasValidTerritoryInfo()};
    territory["boundary_dist"] = Value{s->getTerritoryBoundaryDist()};
    territory["brg_to_center"] = Value{s->getBrgToTerritoryCenter()};

    // ── 5. faor ───────────────────────────────────────────────────────────────
    Object faor;
    faor["valid"] = Value{s->getHasValidFaorInfo()};
    faor["boundary_dist"] = Value{s->getFaorBoundaryDist()};
    faor["brg_to_center"] = Value{s->getBrgToFaorCenter()};
    faor["active"] = Value{s->getFaorBoundaryActive()};

    // ── 6. inventory ─────────────────────────────────────────────────────────
    Object inventory;
    inventory["valid"] = Value{s->getHasValidInventory()};
    inventory["fuel_remaining"] = Value{s->getFuelRemaining()};
    inventory["fuel_safe_amount"] = Value{s->getFuelSafeAmount()};
    inventory["num_missiles"] = Value{static_cast<double>(s->getNumMissiles())};
    inventory["num_bombs"] = Value{static_cast<double>(s->getNumBombs())};

    // ── 7. shared ─────────────────────────────────────────────────────────────
    Object shared;
    shared["valid"] = Value{s->getHasValidSharedInfo()};

    Object hold;
    hold["num_acft"] = Value{static_cast<double>(s->getHoldNumAcft())};
    hold["my_idx"] = Value{static_cast<double>(s->getHoldMyIdx())};
    hold["ref_exec_time"] = Value{s->getHoldRefExecTime()};
    hold["ref_time_length"] = Value{s->getHoldRefTimeLength()};
    hold["start_time"] = Value{s->getHoldStartTime()};
    shared["hold"] = Value{std::move(hold)};

    Object cap;
    cap["active"] = Value{s->getHasActiveCap()};
    cap["engagement_marker"] = Value{s->getHasEngagementMarker()};
    cap["start_time"] = Value{s->getCapStartTime()};
    cap["latitude"] = Value{s->getCapLatitude()};
    cap["longitude"] = Value{s->getCapLongitude()};
    cap["heading"] = Value{s->getCapHeading()};
    cap["commit_dist"] = Value{s->getCapCommitDist()};
    cap["flow_cycle_end"] = Value{s->getFlowCycleEnd()};
    shared["cap"] = Value{std::move(cap)};

    shared["gnd_atk_start_time"] = Value{s->getGndAtkStartTime()};
    shared["abort_gnd_atk"] = Value{s->getAbortGndAtk()};

    Object formation;
    formation["leader_fly_to"] = Value{static_cast<double>(s->getLeaderFlyTo())};
    formation["form_sign"] = Value{static_cast<double>(s->getFormSign())};
    formation["rel_azimuth"] = Value{s->getFormRelAzmth()};
    formation["rel_dist"] = Value{s->getFormRelDist()};
    formation["rel_alt"] = Value{s->getFormRelAlt()};
    formation["is_joined"] = Value{s->getIsJoined()};
    formation["leader_id"] = Value{s->getLeaderID()};

    Array form_list;
    for (unsigned int i = 0; i < s->getNumAcftForm(); ++i)
    {
        AcftData d{};
        s->getAcftDataByIdx(i, d);
        Object acft;
        acft["rank"] = Value{static_cast<double>(d.rank)};
        acft["player_id"] = Value{static_cast<double>(d.playerID)};
        acft["latitude"] = Value{d.latitude};
        acft["longitude"] = Value{d.longitude};
        acft["altitude"] = Value{d.altitude};
        acft["heading"] = Value{d.heading};
        acft["airspeed"] = Value{d.airspeed};
        acft["is_engaged"] = Value{d.isEngaged};
        acft["is_defending"] = Value{d.isDefending};
        acft["is_joined"] = Value{d.isJoined};
        acft["wing_id"] = Value{static_cast<double>(d.wingID)};
        form_list.push_back(Value{std::move(acft)});
    }
    formation["aircraft"] = Value{std::move(form_list)};
    shared["formation"] = Value{std::move(formation)};

    Array tgt_assign;
    for (unsigned int i = 0; i < s->getNumTargets(); ++i)
    {
        const auto& ta = s->getAirTgtAssign()[i];
        Object t;
        t["own_id"] = Value{static_cast<double>(ta.ownID)};
        t["tgt_id"] = Value{static_cast<double>(ta.tgtID)};
        t["timer"] = Value{ta.timer};
        t["engaged"] = Value{ta.engaged};
        tgt_assign.push_back(Value{std::move(t)});
    }
    shared["tgt_assign"] = Value{std::move(tgt_assign)};
    shared["tgt_assign_enabled_id"] =
        Value{static_cast<double>(s->getTgtAssignEnabledID())};

    Array tgt_shot;
    for (unsigned int i = 0; i < s->getNumTgtShot(); ++i)
    {
        const auto& ts = s->getAirTgtShot()[i];
        Object t;
        t["own_id"] = Value{static_cast<double>(ts.ownID)};
        t["tgt_id"] = Value{static_cast<double>(ts.tgtID)};
        t["lock_timer"] = Value{ts.lockTimer};
        tgt_shot.push_back(Value{std::move(t)});
    }
    shared["tgt_shot"] = Value{std::move(tgt_shot)};

    // ── 8. nav ────────────────────────────────────────────────────────────────
    Object nav;
    nav["valid"] = Value{s->getHasValidNavSysInfo()};
    nav["fly_to_idx"] = Value{static_cast<double>(s->getFlyToIdx())};
    nav["fly_to_lat"] = Value{s->getFlyToLatitude()};
    nav["fly_to_lon"] = Value{s->getFlyToLongitude()};
    nav["fly_to_elev"] = Value{s->getFlyToElevation()};
    nav["fly_to_cmd_alt"] = Value{s->getFlyToCmdAltitude()};
    nav["fly_to_cmd_speed"] = Value{s->getFlyToCmdSpeed()};
    nav["fly_to_bearing"] = Value{s->getFlyToBearing()};
    nav["fly_to_distance"] = Value{s->getFlyToDistance()};
    nav["fly_to_ttg"] = Value{s->getFlyToTimeToGo()};
    nav["auto_seq"] = Value{s->getNavSysAutoSeqState()};

    Object hold_nav;
    hold_nav["has_hold"] = Value{s->getFlyToHasHold()};
    hold_nav["leg_heading"] = Value{s->getFlyToHoldLegHeading()};
    hold_nav["leg_time"] = Value{s->getFlyToHoldLegTime()};
    hold_nav["duration"] = Value{s->getFlyToHoldDuration()};
    hold_nav["counter_clockwise"] = Value{s->getFlyToHoldCounterClockwise()};
    nav["hold"] = Value{std::move(hold_nav)};

    Object terrain_nav;
    terrain_nav["tf_leg"] = Value{s->getFlyToTerrainFollowingLeg()};
    terrain_nav["height_above_terrain"] = Value{s->getFlyToHeightAboveTerrain()};
    nav["terrain"] = Value{std::move(terrain_nav)};

    Object winding;
    winding["has_winding"] = Value{s->getFlyToHasWinding()};
    winding["timeout"] = Value{s->getFlyToWindingTimeout()};
    winding["angle"] = Value{s->getFlyToWindingAngle()};
    winding["spd_factor"] = Value{s->getFlyToWindingSpdFactor()};
    nav["winding"] = Value{std::move(winding)};

    // ── 9. terrain_data ───────────────────────────────────────────────────────
    Object terrain_data;
    terrain_data["valid"] = Value{s->getHasValidTerrainData()};
    Array elev_pts;
    const double* elevData = s->getTerrainElevData();
    const bool* elevValid = s->getValidTerrainElevFlag();
    for (unsigned int i = 0; i < AsaPilotState::NUM_TERRAIN_ELEV_PTS; ++i)
    {
        Object pt;
        pt["elev"] = Value{elevData[i]};
        pt["valid"] = Value{elevValid[i]};
        elev_pts.push_back(Value{std::move(pt)});
    }
    terrain_data["points"] = Value{std::move(elev_pts)};

    // ── 10. radar ─────────────────────────────────────────────────────────────
    Object radar;
    radar["valid"] = Value{s->getHasValidRadarInfo()};
    radar["mode"] =
        Value{static_cast<double>(static_cast<int>(s->getRadarMode()))};

    // ── 11. air_tracks ────────────────────────────────────────────────────────
    Array air_tracks;
    for (unsigned int i = 0; i < s->getNumAirTrks(); ++i)
    {
        AirTrackInfo ati{};
        s->getAirTrkInfo(i, ati);

        Object trk;
        trk["valid"] = Value{true};
        trk["iff_code"] = Value{ati.iffCode};
        trk["type"] = Value{static_cast<double>(ati.type)};
        trk["player_id"] = Value{static_cast<double>(ati.playerID)};
        trk["latitude"] = Value{ati.latitude};
        trk["longitude"] = Value{ati.longitude};
        trk["altitude"] = Value{ati.altitude};
        trk["rel_azimuth"] = Value{ati.relAzmth};
        trk["true_azimuth"] = Value{ati.trueAzmth};
        trk["elevation"] = Value{ati.elev};
        trk["range"] = Value{ati.range};
        trk["gnd_range"] = Value{ati.gndRange};
        trk["gnd_track"] = Value{ati.gndTrk};
        trk["gnd_speed"] = Value{ati.gndSpd};
        trk["age"] = Value{ati.age};
        trk["wez_max_o2t_true"] = Value{ati.wezMaxOwn2TrkTrue};
        trk["wez_nez_o2t_true"] = Value{ati.wezNezOwn2TrkTrue};
        trk["wez_max_t2o_true"] = Value{ati.wezMaxTrk2OwnTrue};
        trk["wez_nez_t2o_true"] = Value{ati.wezNezTrk2OwnTrue};
        trk["wez_max_t2o_pred"] = Value{ati.wezMaxTrk2OwnPred};
        trk["wez_nez_t2o_pred"] = Value{ati.wezNezTrk2OwnPred};
        trk["wez_pred_dist_corr"] = Value{ati.wezPredDistCorrection};
        trk["wez_nez_o2t_pred"] = Value{ati.wezNezOwn2TrkPred};
        trk["has_spike"] = Value{ati.hasSpike};
        trk["emitter_class"] = Value{ati.emitterClass};
        trk["emitter_mode"] = Value{ati.emitterMode};

        Array pe;
        pe.push_back(Value{ati.posECEF[0]});
        pe.push_back(Value{ati.posECEF[1]});
        pe.push_back(Value{ati.posECEF[2]});
        trk["pos_ecef"] = Value{std::move(pe)};

        Array ve;
        ve.push_back(Value{ati.velECEF[0]});
        ve.push_back(Value{ati.velECEF[1]});
        ve.push_back(Value{ati.velECEF[2]});
        trk["vel_ecef"] = Value{std::move(ve)};

        air_tracks.push_back(Value{std::move(trk)});
    }

    // ── 12. gnd_atk ───────────────────────────────────────────────────────────
    Object gnd_atk;
    gnd_atk["valid"] = Value{s->getHasValidGndAtkInfo()};
    gnd_atk["assigned_alt"] = Value{s->getGndAtkAssignedAlt()};
    gnd_atk["miss_dist"] = Value{s->getGndAtkMissDist()};
    gnd_atk["toss_range"] = Value{s->getGndAtkTossRange()};
    gnd_atk["tgt_ete"] = Value{s->getGndTgtETE()};
    gnd_atk["next_stpt"] = Value{static_cast<double>(s->getGndAtkNextStpt())};

    // ── 13. events ────────────────────────────────────────────────────────────
    Object events;
    events["end_of_rdr_scan"] = Value{s->getEndOfRdrScnFlag()};
    events["seeker_active"] = Value{s->getSeekerActiveFlag()};
    events["missile_detonated"] = Value{s->getMissileDetonateFlag()};
    events["timeout_tgt_id"] = Value{static_cast<double>(s->getTimeoutTgtID())};
    events["mission_accomplished"] = Value{s->getHasMissionAccomplished()};

    Array outer_msl_warn;
    for (unsigned int i = 0; i < s->getNumOuterMslWarn(); ++i)
        outer_msl_warn.push_back(Value{s->getOuterMslWarnTaz()[i]});
    events["outer_msl_warn_taz"] = Value{std::move(outer_msl_warn)};

    // ── 14. radio ─────────────────────────────────────────────────────────────
    Object radio;
    radio["has_msg"] = Value{s->getHasRadioMsg()};
    radio["voice_time_length"] = Value{s->getRadioMsgVoiceTimeLength()};
    radio["voice_content"] = Value{s->getRadioMsgVoiceContent()};
    radio["sender_name"] = Value{s->getRadioMsgSenderName()};

    // ── 15. bt_blackboard ─────────────────────────────────────────────────────
    Object bb;

    // Bingo Fuel
    bb["has_bingo_fuel"] = Value{hasBingoFuel};

    // Proximity warning
    bb["has_prox_warning"] = Value{hasProxWarning};
    bb["taz_prox_warning"] = Value{tazProxWarning};

    // RWR
    bb["rwr_has_rear_spike"] = Value{rwrHasRearSpike};
    bb["taz_rwr_rear_spike"] = Value{tazRwrRearSpike};
    bb["rwr_msl_warning"] = Value{rwrMslWarning};
    bb["taz_rwr_msl_warning"] = Value{tazRwrMslWarning};

    // Wingman support
    bb["has_wingman"] = Value{hasWingman};
    bb["wing_target"] = serialize_track(wingTarget);

    // Defensive Air Assessment
    bb["has_nez_warning"] = Value{hasNezWarning};
    bb["taz_nez_warning"] = Value{tazNezWarning};
    bb["has_wez_warning"] = Value{hasWezWarning};
    bb["hp_threat"] = serialize_track(hpThreat);
    bb["threat_score"] = Value{threatScore};
    bb["is_defending"] = Value{isDefending};

    // Offensive Air Assessment
    bb["hp_target"] = serialize_track(hpTarget);
    bb["hp_tgt_inside_territory"] = Value{hpTgtInsideTerritory};
    bb["hp_tgt_inside_faor"] = Value{hpTgtInsideFaor};
    bb["shot_distance"] = Value{shotDistance};
    bb["crank_side_after_shot"] = Value{static_cast<double>(crankSideAfterShot)};
    bb["has_opportunity_shot"] = Value{hasOpportunityShot};
    bb["opportunity_shot_dist"] = Value{opportunityShotDist};
    bb["hp_target_engaged"] = Value{hpTargetEngaged};
    bb["is_uplink_active"] = Value{isUplinkActive};

    // Ground Attack
    bb["gnd_atk_accomplished"] = Value{gndAtkAccomplished};

    // Altitude safety
    bb["alt_safety_check"] = Value{altSafetyCheck};

    // ── root ──────────────────────────────────────────────────────────────────
    Object root;
    root["sim"] = Value{std::move(sim)};
    root["ownship"] = Value{std::move(own)};
    root["game_area"] = Value{std::move(game_area)};
    root["territory"] = Value{std::move(territory)};
    root["faor"] = Value{std::move(faor)};
    root["inventory"] = Value{std::move(inventory)};
    root["shared"] = Value{std::move(shared)};
    root["nav"] = Value{std::move(nav)};
    root["terrain_data"] = Value{std::move(terrain_data)};
    root["radar"] = Value{std::move(radar)};
    root["air_tracks"] = Value{std::move(air_tracks)};
    root["gnd_atk"] = Value{std::move(gnd_atk)};
    root["events"] = Value{std::move(events)};
    root["radio"] = Value{std::move(radio)};
    root["bt_blackboard"] = Value{std::move(bb)};

    return root;
}



### 5. Inicializar o cliente no construtor do agente

```cpp
AsaPilotBT::AsaPilotBT() : AsaPilot()
{
    STANDARD_CONSTRUCTOR()

    client = std::make_unique<mlinference::client::InferenceClient>(server);
    client->connect();

    if(client->is_connected()) {
        client->load_model(model_id, model_path);
    }
}
```


### 6. Chamar o modelo no ciclo de reasoning

```cpp
//----------------------- MIIA BEGIN ----------------------------------
if (client->is_connected())
{
    auto preparedState = prepareState(state);
    auto result = client->predict(model_id, preparedState);
    auto outputs = result.outputs;

    if (result.success && !outputs.empty())
    {
        // Leia as saídas declaradas em get_schema() e aplique ao agente:
        // exemplo:
        auto airspeed = outputs["airspeed"].as_number();

        // Defining dynamic action
        action->setHas...(true);
        action->set...(airspeed);
    }
    else
    {
        LOG_DEBUG("asa_pilot_bt") << "MIIA: predição falhou";

    }
}
else
{
    LOG_WARN("asa_pilot_bt") << "Modelo MIIA não está conectado";
}
//----------------------- MIIA END ------------------------------------
```

As chaves de `outputs` correspondem exatamente aos nomes declarados em
`get_schema()` — um campo por saída do `predict()`.

---

## Usando o modelo desenvolvido no ASA/MIXR

Na pasta `asa/dist`, crie a pasta `models` (se ainda não existir) e copie `pilotBT.py` e `.venv` para esse diretório

```
asa/dist/
├── bin
|     ...
|    ├── logs
|    └── AsaCommander
...
└── models
    ├── .venv
    ├── miia_model.py
    └── pilotBT.py
```

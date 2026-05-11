"""
pilot_model.py — Interface base para o agente AsaPilotBT.

Este arquivo define:

- :class:`TensorSpec`, :class:`ModelSchema`, :class:`ValidationResult`,
  :class:`WarmupResult` — tipos de dados do contrato de I/O.
- :class:`MiiaModel` — classe base abstrata exigida pelo servidor AsaMiia.
- :class:`PilotBTModel` — especialização abstrata para o agente AsaPilotBT.

**Não edite este arquivo.**  Implemente apenas ``pilotBT.py``.
"""

from __future__ import annotations

import time
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import logging
import datetime
import json
import numpy as np
import numpy.typing as npt


# =============================================================================
# Estruturas de dados
# =============================================================================

@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Descreve uma porta de tensor (entrada ou saída) de um modelo.

    Parâmetros
    ----------
    name : str
        Nome do tensor.  Deve corresponder às chaves usadas em
        :meth:`MiiaModel.predict` e no ``client::Object`` enviado pelo C++.
    shape : list[int]
        Especificação de forma.  Use ``-1`` para dimensões dinâmicas.
        Para inputs estruturados este campo é apenas informativo.
    dtype : str
        String de dtype compatível com NumPy (ex.: ``"float32"``).
    description : str
        Descrição legível exibida aos clientes via ``GetModelInfo``.
    min_value : float | None
        Limite inferior opcional (apenas portas escalares/tensoriais).
    max_value : float | None
        Limite superior opcional (apenas portas escalares/tensoriais).
    structured : bool
        ``True`` quando a porta transporta um ``dict``/``list`` aninhado em
        vez de um tensor numérico plano.  Quando ``True``:

        - A validação de shape/dtype é ignorada em ``validate_inputs()``.
        - O warmup do C++ gera um ``Object{}`` vazio para esta porta.
        - O campo ``TensorSpecData.structured`` é marcado no protobuf.
    """

    name:        str
    shape:       list[int]
    dtype:       str        = "float32"
    description: str        = ""
    min_value:   float|None = None
    max_value:   float|None = None
    structured:  bool       = False


@dataclass(frozen=True, slots=True)
class ModelSchema:
    """Descrição completa de I/O de um modelo.

    Parâmetros
    ----------
    inputs : list[TensorSpec]
        Lista ordenada de especificações de tensores de entrada.
    outputs : list[TensorSpec]
        Lista ordenada de especificações de tensores de saída.
    description : str
        Descrição em texto livre do que o modelo faz.
    author : str
        Autor ou equipe responsável pelo modelo.
    tags : dict[str, str]
        Metadados arbitrários em chave-valor.
    """

    inputs:      list[TensorSpec]
    outputs:     list[TensorSpec]
    description: str            = ""
    author:      str            = ""
    tags:        dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    """Resultado de :meth:`MiiaModel.validate_inputs`.

    Parâmetros
    ----------
    valid : bool
        ``True`` se todos os inputs passaram na validação.
    errors : list[str]
        Mensagens de erro legíveis.  Vazio quando ``valid`` é ``True``.
    """

    valid:  bool      = True
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WarmupResult:
    """Resultado de :meth:`MiiaModel.warmup`.

    Parâmetros
    ----------
    runs_completed : int
        Número de inferências de aquecimento executadas.
    avg_time_ms : float
        Tempo médio entre todas as execuções [ms].
    min_time_ms : float
        Menor tempo observado [ms].
    max_time_ms : float
        Maior tempo observado [ms].
    """

    runs_completed: int   = 0
    avg_time_ms:    float = 0.0
    min_time_ms:    float = 0.0
    max_time_ms:    float = 0.0


# =============================================================================
# Classe base abstrata — contrato com o servidor
# =============================================================================

class MiiaModel(ABC):
    """Classe base abstrata que todo modelo Python do AsaMiia deve herdar.

    O servidor realiza exatamente esta sequência ao carregar um arquivo
    ``.py``:

    1. Importa o módulo.
    2. Encontra a primeira classe **concreta** (não abstrata) que seja
       subclasse de :class:`MiiaModel`.
    3. Instancia sem argumentos: ``model = ModelClass()``.
    4. Chama :meth:`load` uma única vez.
    5. Encaminha cada requisição ``Predict`` gRPC para :meth:`predict`.

    Regras
    ------
    - ``__init__`` não pode ter parâmetros obrigatórios além de ``self``.
    - Não use ``print()`` — use ``logging`` para não poluir o stdout do
      servidor.
    - Não use ``sys.exit()`` ou ``os._exit()``.
    - O servidor serializa chamadas via mutex — modelos não precisam ser
      thread-safe.
    """

    # -------------------------------------------------------------------------
    # Métodos obrigatórios
    # -------------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Inicializa o modelo.

        Chamado uma vez antes de qualquer :meth:`predict`.  Use para carregar
        pesos, construir grafos ou abrir arquivos externos.

        Qualquer exceção sinaliza falha de carregamento ao cliente.
        """
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
        """Executa inferência sobre um conjunto de inputs.

        Parâmetros
        ----------
        inputs : dict[str, Any]
            Mapeamento nome → valor espelhando o ``client::Object`` do C++.
            Tipos possíveis: ``float``, ``bool``, ``str``, ``list``,
            ``dict[str, Any]``, ``None``.

        Retorna
        -------
        dict[str, np.ndarray]
            Mapeamento nome → array NumPy.
            Shape ``[1]`` ou ``[1, 1]`` é convertido para escalar pelo
            servidor; qualquer outro shape vira ``Array``.

        Qualquer exceção não capturada é retornada em
        ``PredictionResult.error_message`` com ``success = false``.
        """
        ...

    @abstractmethod
    def get_schema(self) -> ModelSchema:
        """Retorna o schema de I/O do modelo.

        Deve ser chamável tanto antes quanto depois de :meth:`load`.
        """
        ...

    def unload(self) -> None:
        """Libera recursos mantidos pelo modelo.

        Chamado quando o modelo é descarregado.  Sobrescreva se mantiver
        memória GPU, descritores de arquivo ou conexões de rede.
        """

    def validate_inputs(self, inputs: dict[str, Any]) -> ValidationResult:
        """Verifica se ``inputs`` satisfazem o schema.

        A implementação padrão verifica presença de chaves, shapes e
        intervalos para portas não-estruturadas.  Portas com
        ``structured=True`` ignoram as verificações de shape/dtype.

        Sobrescreva para adicionar validações específicas do domínio.
        """
        schema = self.get_schema()
        errors: list[str] = []

        expected = {s.name for s in schema.inputs}
        provided = set(inputs.keys())

        for missing in expected - provided:
            errors.append(f"Input ausente: {missing}")
        for extra in provided - expected:
            errors.append(f"Input inesperado: {extra}")

        for spec in schema.inputs:
            if spec.name not in inputs or spec.structured:
                continue
            raw = inputs[spec.name]
            try:
                arr = np.asarray(raw, dtype=spec.dtype)
            except (ValueError, TypeError) as exc:
                errors.append(f"{spec.name}: não é possível converter para {spec.dtype}: {exc}")
                continue

            if len(arr.shape) != len(spec.shape):
                errors.append(f"{spec.name}: esperado {len(spec.shape)} dims, obtido {len(arr.shape)}")
            else:
                for i, (actual, expected_dim) in enumerate(zip(arr.shape, spec.shape)):
                    if expected_dim != -1 and actual != expected_dim:
                        errors.append(f"{spec.name}: dim {i} esperado {expected_dim}, obtido {actual}")

            if spec.min_value is not None and float(np.min(arr)) < spec.min_value:
                errors.append(f"{spec.name}: valores abaixo do mínimo {spec.min_value}")
            if spec.max_value is not None and float(np.max(arr)) > spec.max_value:
                errors.append(f"{spec.name}: valores acima do máximo {spec.max_value}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def warmup(self, n: int = 5) -> WarmupResult:
        """Executa *n* inferências sintéticas para aquecer caches JIT.

        A implementação padrão gera arrays aleatórios para portas não-
        estruturadas e dicts vazios para portas estruturadas.  Sobrescreva
        quando precisar de inputs representativos.
        """
        schema = self.get_schema()
        dummy: dict[str, Any] = {}
        for spec in schema.inputs:
            if spec.structured:
                dummy[spec.name] = {}
            else:
                shape = [d if d != -1 else 1 for d in spec.shape]
                dummy[spec.name] = np.random.rand(*shape).astype(spec.dtype)

        times: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)

        return WarmupResult(
            runs_completed=len(times),
            avg_time_ms=sum(times) / len(times) if times else 0.0,
            min_time_ms=min(times) if times else 0.0,
            max_time_ms=max(times) if times else 0.0,
        )

    def memory_usage_bytes(self) -> int:
        """Retorna o uso estimado de memória em bytes.  Padrão: ``0``."""
        return 0

    def __repr__(self) -> str:
        schema = self.get_schema()
        return (
            f"<{self.__class__.__name__} "
            f"inputs={len(schema.inputs)} outputs={len(schema.outputs)} "
            f"desc={schema.description!r}>"
        )


# =============================================================================
# Base especializada para o agente AsaPilotBT
# =============================================================================

class PilotBTModel(MiiaModel):
    """Base concreta para o agente AsaPilotBT.

    Implementa tudo que é comum a qualquer piloto autônomo neste ambiente:
    entradas fixas do agente, warmup com input representativo e input de
    exemplo.

    Métodos obrigatórios para subclasses
    -------------------------------------
    - :meth:`predict` — lógica de decisão; retorna ``dict[str, np.ndarray]``
      com as chaves que o modelo produz.
    - :meth:`get_schema` — declara as saídas via :meth:`output_spec` e
      :meth:`_base_schema`.  As entradas já estão fixas em :attr:`_INPUTS`.

    Opcionalmente, sobrescreva :meth:`load` para carregar pesos ou inicializar
    recursos externos.

    Hierarquia
    ----------
    .. code-block:: text

        MiiaModel          (contrato com o servidor)
            └── PilotBTModel   (este arquivo — não editar)
                    └── PilotBT    (pilotBT.py — implemente aqui)

    Atenção
    -------
    O servidor localiza a **primeira classe concreta** (não abstrata) que
    herde de :class:`MiiaModel` no módulo importado.  Como ``PilotBTModel``
    ainda possui métodos abstratos, ela é ignorada automaticamente —
    apenas ``PilotBT`` é instanciada.
    """
    
    def __init__(self, log_path: str = "./logs/pilotBT.log"):
        super().__init__()
        
        self._log_path = log_path
        self._logger   = None   # lazy — criado na primeira chamada a self.log()
        self._sim_time = 0.0    # atualizado no início de predict()
        
    def _get_logger(self) -> logging.Logger:
        if self._logger is None:
            logger = logging.getLogger(str(id(self)))   # isolado por instância
            logger.setLevel(logging.DEBUG)
            logger.propagate = False               # não vaza pro logger raiz
            pathlib.Path(self._log_path).parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self._log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            self._logger = logger
        return self._logger


    def log(self, *args) -> None:
        """Escreve no arquivo de log com prefixo de tempo automático.

        Uso idêntico ao print():
            self.log("alt:", alt, "hdg:", hdg)
            self.log(f"foe range={tgt_rng:.0f}m")
        """
        msg = " ".join(str(a) for a in args)
        # ts  = datetime.datetime.utcnow().isoformat(timespec="milliseconds")
        # self._get_logger().debug(f"[{ts} | t={self._sim_time:.1f}s] {msg}")
        self._get_logger().debug(msg)
            

    # -------------------------------------------------------------------------
    # Implementações padrão de load / unload / memory
    # -------------------------------------------------------------------------

    def load(self) -> None:
        """Implementação vazia.

        Sobrescreva se precisar carregar pesos, inicializar uma rede neural
        ou abrir arquivos externos.
        """

    def unload(self) -> None:
        """Implementação vazia.

        Sobrescreva se mantiver recursos externos que precisam ser liberados
        (memória GPU, conexões, descritores de arquivo).
        """

    def memory_usage_bytes(self) -> int:
        """Retorna ``0``.  Sobrescreva para reportar o uso real."""
        return 0

    # -------------------------------------------------------------------------
    # predict — ainda abstrato
    # -------------------------------------------------------------------------

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
        """Executa a lógica de decisão do piloto.

        Este é o **único método obrigatório** para subclasses de
        :class:`PilotBTModel`.

        Parâmetros
        ----------
        inputs : dict[str, Any]
            Estado completo do agente produzido por
            ``AsaPilotBT::prepareState()``.  Consulte :meth:`make_dummy_input`
            para a estrutura completa com valores realistas.

        Retorna
        -------
        dict[str, np.ndarray]
        """
        ...

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------

    # Entradas padrão do agente — compartilhadas por todas as subclasses.
    _INPUTS: list[TensorSpec] = []  # preenchido logo abaixo da classe

    @staticmethod
    def output_spec(name: str, shape: list[int] = [1, 1],
                    dtype: str = "float32",
                    description: str = "") -> TensorSpec:
        """Helper para declarar uma porta de saída de forma concisa.

        Parâmetros
        ----------
        name : str
            Nome do tensor — deve corresponder à chave retornada em
            :meth:`predict`.
        shape : list[int]
            Shape do tensor.  ``[1, 1]`` (padrão) é convertido para escalar
            pelo servidor; qualquer outro shape vira ``Array``.
        dtype : str
            Dtype NumPy.  Padrão: ``"float32"``.
        description : str
            Descrição livre exibida via ``GetModelInfo``.

        Exemplos
        --------
        Saída escalar simples::

            PilotBTModel.output_spec("heading", description="Proa [deg]")

        Vetor de N valores::

            PilotBTModel.output_spec("features", shape=[1, 8])
        """
        return TensorSpec(name=name, shape=shape, dtype=dtype,
                          description=description)

    @abstractmethod
    def get_schema(self) -> ModelSchema:
        """Retorna o schema de I/O do modelo.

        As entradas estão fixas em :attr:`_INPUTS` e não precisam ser
        redeclaradas.  Use :meth:`output_spec` para declarar as saídas e
        construa o :class:`ModelSchema` com o helper :meth:`_base_schema`.
        """
        ...

    def _base_schema(self, outputs: list[TensorSpec],
                     description: str = "",
                     author: str = "",
                     tags: dict[str, str] | None = None) -> ModelSchema:
        """Constrói um :class:`ModelSchema` com as entradas padrão do agente.

        Parâmetros
        ----------
        outputs : list[TensorSpec]
            Portas de saída declaradas pelo modelo.  Use :meth:`output_spec`
            para criá-las.
        description : str
            Descrição livre do modelo.
        author : str
            Autor ou equipe.
        tags : dict[str, str] | None
            Metadados arbitrários.  ``{"type": "air_combat"}`` é mesclado
            automaticamente.

        Retorna
        -------
        ModelSchema
            Schema completo com as 14 entradas fixas + as saídas fornecidas.
        """
        merged_tags = {"type": "air_combat", "framework": "AsaMiia"}
        if tags:
            merged_tags.update(tags)
        return ModelSchema(
            inputs=PilotBTModel._INPUTS,
            outputs=outputs,
            description=description,
            author=author,
            tags=merged_tags,
        )

    # -------------------------------------------------------------------------
    # Warmup com input representativo
    # -------------------------------------------------------------------------

    def warmup(self, n: int = 5) -> WarmupResult:
        """Executa *n* inferências com o input de exemplo completo.

        Usa :meth:`make_dummy_input` em vez de dicts vazios, garantindo que
        todos os campos estruturados estejam presentes com valores realistas.
        """
        dummy = self.make_dummy_input()
        times: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)

        return WarmupResult(
            runs_completed=len(times),
            avg_time_ms=sum(times) / len(times) if times else 0.0,
            min_time_ms=min(times) if times else 0.0,
            max_time_ms=max(times) if times else 0.0,
        )

    # -------------------------------------------------------------------------
    # Input de exemplo
    # -------------------------------------------------------------------------

    @staticmethod
    def make_dummy_input() -> dict[str, Any]:
        """Carrega o estado de exemplo a partir de ``dummy_input.json``.
 
        O arquivo é procurado na raiz do projeto, definida como o diretório
        que contém ``pilot_model.py``.  Se o arquivo não for encontrado ou
        estiver malformado, uma exceção é levantada com mensagem clara.
 
        Para customizar o input de teste, edite ``dummy_input.json``
        diretamente — não é necessário tocar neste arquivo.
 
        Levanta
        -------
        FileNotFoundError
            Se ``dummy_input.json`` não existir na raiz do projeto.
        ValueError
            Se o arquivo existir mas não for JSON válido ou não contiver
            um objeto no nível raiz.
        """
         
        json_path = Path(__file__).parent / "state_example.json"
 
        if not json_path.exists():
            raise FileNotFoundError(
                f"Arquivo de input de exemplo não encontrado: {json_path}\n"
                f"Certifique-se de que 'state_example.json' está na raiz do projeto."
            )
 
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"'state_example.json' contém JSON inválido: {exc}"
            ) from exc
 
        if not isinstance(data, dict):
            raise ValueError(
                f"'state_example.json' deve conter um objeto JSON no nível raiz, "
                f"mas encontrou {type(data).__name__}."
            )
 
        return data



# =============================================================================
# Entradas fixas do agente — populadas fora da classe para evitar referência
# ao nome da classe dentro do próprio corpo da classe.
# =============================================================================

def _s(name: str, desc: str) -> TensorSpec:
    return TensorSpec(name=name, shape=[-1], dtype="float32",
                      structured=True, description=desc)


PilotBTModel._INPUTS = [
    _s("sim",
       "Dados de simulação: phase, exec_time [s], sim_time [s]"),
    _s("ownship",
       "Estado do ownship: latitude [deg], longitude [deg], "
       "altitude [m], ground_speed [m/s], ground_track [deg], "
       "heading [deg], pitch [deg], roll [deg], terrain_elev [m], "
       "damage [0..1], pos_ecef [m], vel_ecef [m/s], vel_body [m/s], "
       "accel_ecef [m/s²], accel_body [m/s²], "
       "euler_ned [rad], euler_ecef [rad], "
       "ang_vel_body [rad/s], ang_vel_ecef [rad/s]"),
    _s("game_area",
       "Área de jogo: boundary_dist [m], brg_to_center [deg]"),
    _s("territory",
       "Território: boundary_dist [m], brg_to_center [deg]"),
    _s("faor",
       "FAOR: boundary_dist [m] (positivo = dentro), "
       "brg_to_center [deg], active"),
    _s("inventory",
       "Inventário: fuel_remaining [lbs], fuel_safe_amount [lbs], "
       "num_missiles, num_bombs"),
    _s("shared",
       "Dados da formação: hold{num_acft, my_idx, ref_exec_time, "
       "ref_time_length, start_time}, "
       "cap{active, engagement_marker, start_time, lat, lon, hdg, "
       "commit_dist, flow_cycle_end}, "
       "gnd_atk_start_time, abort_gnd_atk, "
       "formation{leader_fly_to, form_sign, rel_azimuth, rel_dist, "
       "rel_alt, is_joined, leader_id, "
       "aircraft[{rank, player_id, lat, lon, alt, hdg, airspeed, "
       "is_engaged, is_defending, is_joined, wing_id}]}, "
       "tgt_assign[{own_id, tgt_id, timer, engaged}], "
       "tgt_assign_enabled_id, "
       "tgt_shot[{own_id, tgt_id, lock_timer}]"),
    _s("nav",
       "Navegação: fly_to_idx, fly_to_lat [deg], fly_to_lon [deg], "
       "fly_to_elev [m], fly_to_cmd_alt [m], fly_to_cmd_speed [m/s], "
       "fly_to_bearing [deg], fly_to_distance [m], fly_to_ttg [s], "
       "auto_seq, "
       "hold{has_hold, leg_heading [deg], leg_time [s], duration [s], "
       "counter_clockwise}, "
       "terrain{tf_leg, height_above_terrain [m]}, "
       "winding{has_winding, timeout [s], angle [deg], spd_factor}"),
    _s("terrain_data",
       "Elevação do terreno: points[11] → {elev [m], valid}; "
       "índice i corresponde a i segundos à frente da aeronave"),
    _s("radar",
       "Estado do radar: mode (enum AsaRadar::RadarMode como double)"),
    _s("air_tracks",
       "Tracks do radar/datalink/RWR: iff_code, type, player_id, "
       "latitude [deg], longitude [deg], altitude [m], "
       "rel_azimuth [deg], true_azimuth [deg], elevation [deg], "
       "range [m], gnd_range [m], gnd_track [deg], gnd_speed [m/s], "
       "age [s], wez_max_o2t_true [m], wez_nez_o2t_true [m], "
       "wez_max_t2o_true [m], wez_nez_t2o_true [m], "
       "wez_max_t2o_pred [m], wez_nez_t2o_pred [m], "
       "wez_pred_dist_corr [m], wez_nez_o2t_pred [m], "
       "has_spike, emitter_class, emitter_mode, "
       "pos_ecef [m], vel_ecef [m/s]"),
    _s("gnd_atk",
       "Ataque ao solo: assigned_alt [m], miss_dist [m], "
       "toss_range [m], tgt_ete [s], next_stpt"),
    _s("events",
       "Eventos: end_of_rdr_scan, seeker_active, missile_detonated, "
       "timeout_tgt_id, mission_accomplished, "
       "outer_msl_warn_taz[] [deg]"),
    _s("radio",
       "Rádio: has_msg, voice_time_length [s], "
       "voice_content, sender_name"),
    _s("bt_blackboard",
       "Blackboard interno da BT: "
       "has_bingo_fuel, "
       "has_prox_warning, taz_prox_warning [deg], "
       "rwr_has_rear_spike, taz_rwr_rear_spike [deg], "
       "rwr_msl_warning, taz_rwr_msl_warning [deg], "
       "has_wingman, wing_target{valid, ...AirTrackInfo}, "
       "has_nez_warning, taz_nez_warning [deg], "
       "has_wez_warning, hp_threat{valid, ...AirTrackInfo}, "
       "threat_score [0..1], is_defending, "
       "hp_target{valid, ...AirTrackInfo}, "
       "hp_tgt_inside_territory, hp_tgt_inside_faor, "
       "shot_distance [m], crank_side_after_shot, "
       "has_opportunity_shot, opportunity_shot_dist [m], "
       "hp_target_engaged, is_uplink_active, "
       "gnd_atk_accomplished, alt_safety_check")
]

del _s  # helper de construção — não faz parte da API pública
"""
.. module:: miia_model
   :synopsis: Interface base para modelos Python servidos pelo AsaMiia.

.. moduleauthor:: Lucas

MiiaModel — Classe base abstrata para modelos Python do AsaMiia.
================================================================

Todo arquivo ``.py`` colocado no diretório de modelos deve expor uma classe
que herde de :class:`MiiaModel` e implemente os três métodos abstratos
obrigatórios: :meth:`~MiiaModel.load`, :meth:`~MiiaModel.predict` e
:meth:`~MiiaModel.get_schema`.

O servidor AsaMiia realiza os seguintes passos ao carregar um modelo:

1. Importa o módulo Python.
2. Encontra a **primeira** classe que seja subclasse de :class:`MiiaModel`.
3. Instancia a classe **sem argumentos** (``model = ModelClass()``).
4. Chama :meth:`~MiiaModel.load` exatamente uma vez.
5. Encaminha cada requisição ``Predict`` gRPC para :meth:`~MiiaModel.predict`.

Formato dos inputs
------------------
O dicionário ``inputs`` passado a :meth:`~MiiaModel.predict` espelha o tipo
``Object`` da API C++ (``mlinference::client::Object``).  Cada valor pode ser:

- ``float``  — de ``mlinference::client::Value`` número
- ``bool``
- ``str``
- ``list``              — de ``Value`` Array (pode ser aninhado)
- ``dict[str, Any]``   — de ``Value`` Object (aninhamento arbitrário)
- ``None``              — de ``Value`` Null

Modelos com tensores planos convertem os valores diretamente para
``np.ndarray``.  Modelos com inputs estruturados (dicts/listas aninhados)
devem percorrer a estrutura e construir arrays conforme necessário.

Exemplos
--------
**Modelo com tensor plano (estilo ONNX):**

.. code-block:: python

    import numpy as np
    from miia_model import MiiaModel, ModelSchema, TensorSpec

    class ModeloLinear(MiiaModel):
        def load(self) -> None:
            self._pesos = np.array([2.0, 1.0, 0.5])

        def predict(self, inputs: dict) -> dict:
            x = np.array(inputs["entrada"], dtype=np.float32)
            return {"saida": np.array([x @ self._pesos])}

        def get_schema(self) -> ModelSchema:
            return ModelSchema(
                inputs=[TensorSpec("entrada", [1, 3], "float32")],
                outputs=[TensorSpec("saida", [1], "float32")],
                description="Modelo linear simples",
                author="Lucas",
            )

**Modelo com input estruturado (navegação):**

.. code-block:: python

    import math, numpy as np
    from miia_model import MiiaModel, ModelSchema, TensorSpec

    class ModeloNavegacao(MiiaModel):
        def load(self) -> None:
            pass

        def predict(self, inputs: dict) -> dict:
            estado = inputs["state"]
            rumo = float(estado["toHeading"])
            return {"heading": np.array([[rumo]], dtype=np.float32)}

        def get_schema(self) -> ModelSchema:
            return ModelSchema(
                inputs=[TensorSpec("state", [-1], "float32",
                                   structured=True,
                                   description="Estado estruturado do navio")],
                outputs=[TensorSpec("heading", [1, 1], "float32",
                                    description="Rumo comandado em graus")],
            )
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


# =============================================================================
# Estruturas de dados
# =============================================================================


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Descreve uma porta de tensor (entrada ou saída) de um modelo.

    Utilizada em :class:`ModelSchema` para declarar o contrato de I/O do
    modelo.  O servidor serializa esta estrutura na mensagem protobuf
    ``TensorSpec``, retornada pelo RPC ``GetModelInfo``.

    Parâmetros
    ----------
    name : str
        Nome do tensor.  Deve corresponder às chaves usadas em
        :meth:`MiiaModel.predict` e no ``client::Object`` enviado pelo C++.
    shape : list[int]
        Especificação de forma.  Use ``-1`` para dimensões dinâmicas.
        Para inputs estruturados este campo é apenas informativo.
    dtype : str
        String de dtype compatível com NumPy (ex.: ``"float32"``, ``"int64"``).
        Padrão: ``"float32"``.
    description : str
        Descrição legível exibida aos clientes via ``GetModelInfo``.
    min_value : float | None
        Limite inferior opcional para valores de entrada (apenas portas
        escalares/tensoriais).  Ignorado para portas estruturadas.
    max_value : float | None
        Limite superior opcional para valores de entrada (apenas portas
        escalares/tensoriais).  Ignorado para portas estruturadas.
    structured : bool
        Definir como ``True`` quando a porta transporta um ``dict``/``list``
        aninhado em vez de um tensor numérico plano.

        Quando ``True``:

        - A validação de shape/dtype é ignorada em
          :meth:`MiiaModel.validate_inputs`.
        - O warmup do C++ gera um ``Object{}`` vazio para esta porta em vez
          de um array aleatório de floats.
        - O campo ``TensorSpecData.structured`` é marcado no protobuf,
          permitindo que clientes detectem inputs estruturados.

    Exemplos
    --------
    Tensor plano::

        TensorSpec("entrada", [1, 3], "float32", "Vetor de três features")

    Dimensão de batch dinâmica::

        TensorSpec("features", [-1, 128], "float32")

    Input estruturado (estado de navegação)::

        TensorSpec("state", [-1], "float32", structured=True,
                   description="{ toHeading, latitude, longitude, hazards[] }")
    """

    name: str
    shape: list[int]
    dtype: str = "float32"
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None
    structured: bool = False


@dataclass(frozen=True, slots=True)
class ModelSchema:
    """Descrição completa de I/O de um modelo.

    Retornada por :meth:`MiiaModel.get_schema` e serializada na mensagem
    protobuf ``ModelInfo`` pelo servidor.  Clientes recebem esta estrutura
    via o RPC ``GetModelInfo``.

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
        Metadados arbitrários em chave-valor (ex.:
        ``{"type": "navigation", "algorithm": "potential_field"}``).

    Exemplos
    --------
    .. code-block:: python

        ModelSchema(
            inputs=[TensorSpec("state", [-1], "float32", structured=True)],
            outputs=[TensorSpec("heading", [1, 1], "float32")],
            description="Navegação por campo potencial",
            author="ASA",
            tags={"type": "navigation"},
        )
    """

    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    description: str = ""
    author: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    """Resultado de :meth:`MiiaModel.validate_inputs`.

    Parâmetros
    ----------
    valid : bool
        ``True`` se todos os inputs passaram na validação.
    errors : list[str]
        Mensagens de erro legíveis, uma por verificação que falhou.
        Vazio quando ``valid`` é ``True``.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WarmupResult:
    """Resultado de :meth:`MiiaModel.warmup`.

    Parâmetros
    ----------
    runs_completed : int
        Número de inferências de aquecimento efetivamente executadas.
    avg_time_ms : float
        Tempo médio de inferência entre todas as execuções, em milissegundos.
    min_time_ms : float
        Menor tempo de inferência observado, em milissegundos.
    max_time_ms : float
        Maior tempo de inferência observado, em milissegundos.
    """

    runs_completed: int = 0
    avg_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0


# =============================================================================
# Classe base abstrata
# =============================================================================


class MiiaModel(ABC):
    """Classe base abstrata que todo modelo Python do AsaMiia deve herdar.

    Os três métodos abstratos — :meth:`load`, :meth:`predict` e
    :meth:`get_schema` — são **obrigatórios**.  Os demais métodos possuem
    implementações padrão que podem ser sobrescritas para comportamento mais
    rico.

    Regras impostas pelo servidor
    -----------------------------
    - A classe deve ser instanciável **sem argumentos** (``__init__`` não pode
      ter parâmetros obrigatórios além de ``self``).
    - Apenas a **primeira** subclasse de :class:`MiiaModel` encontrada no
      módulo é utilizada.  Evite definir múltiplas subclasses em um único
      arquivo.
    - Não utilize ``sys.exit()`` ou ``os._exit()`` — isso encerra todo o
      processo do worker.
    - Não utilize ``print()`` — use o logger ``asalog`` ou o módulo
      ``logging`` do Python para não poluir o stdout do servidor.

    Segurança de concorrência
    -------------------------
    O servidor serializa todas as chamadas a uma única instância de modelo
    via o mutex do ``InferenceEngine``.  Os modelos **não** precisam ser
    thread-safe.

    Ver também
    ----------
    :class:`TensorSpec`, :class:`ModelSchema`, :class:`WarmupResult`
    """

    # -------------------------------------------------------------------------
    # Métodos obrigatórios
    # -------------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Inicializa o modelo (carrega pesos, constrói grafo, abre arquivos, etc.).

        Chamado **exatamente uma vez** pelo servidor logo após a instanciação,
        antes de qualquer chamada a :meth:`predict`.

        Levanta
        -------
        Exception
            Qualquer exceção sinaliza que o modelo não pode ser carregado.
            O servidor registra o erro e reporta ``load_model() = false``
            ao cliente.
        """
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
        """Executa inferência sobre um conjunto de inputs.

        Parâmetros
        ----------
        inputs : dict[str, Any]
            Mapeamento de nome do tensor/campo → valor, espelhando o
            ``client::Object`` do C++.  Tipos possíveis de valor:

            - ``float``           — número escalar
            - ``bool``
            - ``str``
            - ``list``            — array (pode ser aninhado)
            - ``dict[str, Any]``  — objeto aninhado
            - ``None``            — valor nulo

            Para modelos com tensores planos, converta para ``np.ndarray``
            diretamente.  Para modelos com inputs estruturados, percorra o
            dict/list aninhado conforme necessário.

        Retorna
        -------
        dict[str, np.ndarray]
            Mapeamento de nome do tensor de saída → array NumPy.  Cada
            array é convertido de volta para ``client::Value`` pelo servidor:

            - Shape ``[1]`` ou ``[1, 1]`` → escalar (``double`` único).
            - Qualquer outro shape → ``Array`` de números.

        Levanta
        -------
        Exception
            Qualquer exceção não capturada é registrada como traceback Python
            completo e retornada em ``PredictionResult.error_message`` com
            ``success = false``.
        """
        ...

    @abstractmethod
    def get_schema(self) -> ModelSchema:
        """Retorna o schema de I/O do modelo.

        Chamado pelo servidor para popular ``ModelInfo`` e durante a
        validação — em alguns casos antes de :meth:`load` ser chamado.
        Deve, portanto, ser chamável tanto **antes** quanto **depois** de
        :meth:`load`.

        Retorna
        -------
        ModelSchema
            Descrição completa das entradas, saídas e metadados do modelo.
        """
        ...

    # -------------------------------------------------------------------------
    # Métodos opcionais
    # -------------------------------------------------------------------------

    def unload(self) -> None:
        """Libera recursos mantidos pelo modelo.

        Chamado quando o modelo é descarregado do servidor.  Sobrescreva
        este método se o modelo mantiver memória GPU, descritores de arquivo,
        conexões de rede ou outros recursos que devem ser explicitamente
        liberados.

        A implementação padrão não faz nada.
        """

    def validate_inputs(self, inputs: dict[str, Any]) -> ValidationResult:
        """Verifica se ``inputs`` satisfazem o schema do modelo.

        A implementação padrão realiza as seguintes verificações:

        1. Todos os nomes de entrada esperados (do :meth:`get_schema`) estão
           presentes.
        2. Nenhum nome de entrada inesperado foi fornecido.
        3. Para portas não estruturadas: o valor pode ser convertido para
           ``spec.dtype`` e o shape resultante corresponde a ``spec.shape``
           (dimensões ``-1`` são aceitas como curinga).
        4. Para portas não estruturadas com ``min_value`` / ``max_value``
           definidos: todos os valores estão dentro do intervalo declarado.

        Portas estruturadas (``TensorSpec.structured = True``) ignoram as
        verificações 3–4.  Sobrescreva este método para adicionar validações
        específicas do domínio para tais inputs.

        Parâmetros
        ----------
        inputs : dict[str, Any]
            O mesmo dicionário que seria passado a :meth:`predict`.

        Retorna
        -------
        ValidationResult
            ``valid = True`` se todas as verificações passaram;
            ``errors`` lista as falhas.
        """
        schema = self.get_schema()
        errors: list[str] = []

        expected_names = {s.name for s in schema.inputs}
        provided_names = set(inputs.keys())

        for missing in expected_names - provided_names:
            errors.append(f"Input ausente: {missing}")

        for extra in provided_names - expected_names:
            errors.append(f"Input inesperado: {extra}")

        for spec in schema.inputs:
            if spec.name not in inputs:
                continue

            # Ignora verificação de shape/dtype para portas estruturadas.
            if spec.structured:
                continue

            raw = inputs[spec.name]

            # Aceita tanto np.ndarray quanto list/float simples.
            try:
                arr = np.asarray(raw, dtype=spec.dtype)
            except (ValueError, TypeError) as exc:
                errors.append(
                    f"{spec.name}: não é possível converter para {spec.dtype}: {exc}"
                )
                continue

            if len(arr.shape) != len(spec.shape):
                errors.append(
                    f"{spec.name}: esperado {len(spec.shape)} dims, "
                    f"obtido {len(arr.shape)}"
                )
            else:
                for i, (actual, expected) in enumerate(
                    zip(arr.shape, spec.shape)
                ):
                    if expected != -1 and actual != expected:
                        errors.append(
                            f"{spec.name}: dim {i} esperado {expected}, "
                            f"obtido {actual}"
                        )

            if spec.min_value is not None and float(np.min(arr)) < spec.min_value:
                errors.append(
                    f"{spec.name}: valores abaixo do mínimo {spec.min_value}"
                )
            if spec.max_value is not None and float(np.max(arr)) > spec.max_value:
                errors.append(
                    f"{spec.name}: valores acima do máximo {spec.max_value}"
                )

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def warmup(self, n: int = 5) -> WarmupResult:
        """Executa *n* inferências sintéticas para aquecer caches JIT e pré-alocar memória.

        A implementação padrão gera arrays ``np.ndarray`` aleatórios a partir
        do schema (usando ``np.random.rand``).  Para modelos estruturados
        (``TensorSpec.structured = True``) um ``dict`` vazio é utilizado —
        sobrescreva este método para fornecer um input representativo.

        Parâmetros
        ----------
        n : int
            Número de execuções de aquecimento.  Padrão: 5.

        Retorna
        -------
        WarmupResult
            Estatísticas das execuções de aquecimento.

        Notas
        -----
        O ``PythonBackend`` delega ao warmup padrão do C++
        (``ModelBackend::warmup()``), que **não** chama este método Python
        diretamente.  Sobrescreva :meth:`warmup` quando precisar de lógica
        de aquecimento no lado Python — por exemplo, para modelos com inputs
        estruturados cujo input não pode ser gerado automaticamente pelo C++.

        Exemplos
        --------
        .. code-block:: python

            def warmup(self, n: int = 5) -> WarmupResult:
                dummy = {"state": {"toHeading": 45.0, "hazards": []}}
                times = []
                for _ in range(n):
                    t0 = time.perf_counter()
                    self.predict(dummy)
                    times.append((time.perf_counter() - t0) * 1000.0)
                return WarmupResult(
                    runs_completed=len(times),
                    avg_time_ms=sum(times) / len(times),
                    min_time_ms=min(times),
                    max_time_ms=max(times),
                )
        """
        schema = self.get_schema()
        dummy: dict[str, Any] = {}
        for spec in schema.inputs:
            if spec.structured:
                dummy[spec.name] = {}  # dict vazio — sobrescreva para warmup real
            else:
                shape = [d if d != -1 else 1 for d in spec.shape]
                dummy[spec.name] = np.random.rand(*shape).astype(spec.dtype)

        times: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(dummy)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)

        return WarmupResult(
            runs_completed=len(times),
            avg_time_ms=sum(times) / len(times) if times else 0.0,
            min_time_ms=min(times) if times else 0.0,
            max_time_ms=max(times) if times else 0.0,
        )

    def memory_usage_bytes(self) -> int:
        """Estima o uso de memória do modelo em bytes.

        Sobrescreva para fornecer um valor preciso (ex.: tamanho dos arrays
        de pesos).  O padrão retorna ``0``, que o servidor reporta como está
        em ``ModelInfo.memory_usage_bytes``.

        Retorna
        -------
        int
            Uso estimado de memória em bytes.
        """
        return 0

    # -------------------------------------------------------------------------
    # Representação
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Retorna uma representação textual concisa da instância do modelo."""
        schema = self.get_schema()
        n_in = len(schema.inputs)
        n_out = len(schema.outputs)
        return (
            f"<{self.__class__.__name__} "
            f"inputs={n_in} outputs={n_out} "
            f"desc={schema.description!r}>"
        )
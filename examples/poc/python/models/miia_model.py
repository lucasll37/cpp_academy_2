"""
MiiaModel — Abstract Base Class for Python models served by AsaMiia.

Every ``.py`` file placed in the models directory must expose a class that
inherits from :class:`MiiaModel` and implements the required abstract methods.

The server will:
    1. Import the module.
    2. Find the first class that is a subclass of ``MiiaModel``.
    3. Instantiate it (no constructor arguments).
    4. Call ``load()`` once.
    5. Forward ``predict()`` calls for each gRPC ``Predict`` request.

The signatures of ``predict`` and ``get_schema`` mirror the gRPC contract
exactly so there is zero impedance mismatch.

Example
-------
.. code-block:: python

    import numpy as np
    from miia_model import MiiaModel, ModelSchema, TensorSpec

    class MyModel(MiiaModel):
        def load(self) -> None:
            self._weights = np.array([2.0, 1.0, 0.5])

        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            x = inputs["input"]
            return {"output": x @ self._weights}

        def get_schema(self) -> ModelSchema:
            return ModelSchema(
                inputs=[TensorSpec(name="input", shape=[1, 3], dtype="float32")],
                outputs=[TensorSpec(name="output", shape=[1], dtype="float32")],
                description="Simple linear model",
                author="Lucas",
            )
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


# ============================================
# Data Structures
# ============================================


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Describes one named tensor port (input or output).

    Parameters
    ----------
    name : str
        Tensor name (must match the keys used in ``predict``).
    shape : list[int]
        Shape specification.  Use ``-1`` for dynamic dimensions.
    dtype : str
        NumPy-compatible dtype string (e.g. ``"float32"``, ``"int64"``).
    description : str
        Human-readable description shown to clients via ``GetModelInfo``.
    min_value : float | None
        Optional lower bound for valid input values.
    max_value : float | None
        Optional upper bound for valid input values.
    """

    name: str
    shape: list[int]
    dtype: str = "float32"
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True, slots=True)
class ModelSchema:
    """Complete I/O description of a model.

    Returned by :meth:`MiiaModel.get_schema` and serialised into the
    ``ModelInfo`` protobuf message by the server.
    """

    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    description: str = ""
    author: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    """Result of :meth:`MiiaModel.validate_inputs`."""

    valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WarmupResult:
    """Result of :meth:`MiiaModel.warmup`."""

    runs_completed: int = 0
    avg_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0


# ============================================
# Abstract Base Class
# ============================================


class MiiaModel(ABC):
    """Base class that every Python model must inherit from.

    The three abstract methods — ``load``, ``predict`` and ``get_schema`` —
    are **required**.  All other methods have sensible defaults that can be
    overridden for richer behaviour.
    """

    # ------------------------------------------------------------------
    # Required methods
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Initialise the model (load weights, build graph, etc.).

        Called exactly once by the server after instantiation.
        Raise any exception to signal that the model cannot be loaded.
        """
        ...

    @abstractmethod
    def predict(
        self, inputs: dict[str, npt.NDArray[Any]]
    ) -> dict[str, npt.NDArray[Any]]:
        """Run inference.

        Parameters
        ----------
        inputs
            Mapping of tensor name → NumPy array.  The arrays will
            conform to the shapes and dtypes declared in ``get_schema``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of output tensor name → NumPy array.
        """
        ...

    @abstractmethod
    def get_schema(self) -> ModelSchema:
        """Return the model's I/O schema.

        This is called by the server to populate ``ModelInfo``.  Must
        be callable both *before* and *after* ``load()`` (the server
        may call it during validation).
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release resources.  Called when the model is unloaded.

        Override this if your model holds GPU memory, file handles, or
        other resources that should be explicitly freed.
        """

    def validate_inputs(
        self, inputs: dict[str, npt.NDArray[Any]]
    ) -> ValidationResult:
        """Check whether ``inputs`` are valid for this model.

        The default implementation checks names, dtypes and shapes
        against ``get_schema()``.  Override to add domain-specific
        constraints (e.g. value ranges, mutual exclusion between inputs).
        """
        schema = self.get_schema()
        errors: list[str] = []

        expected_names = {s.name for s in schema.inputs}
        provided_names = set(inputs.keys())

        for missing in expected_names - provided_names:
            errors.append(f"Missing input: {missing}")

        for extra in provided_names - expected_names:
            errors.append(f"Unexpected input: {extra}")

        for spec in schema.inputs:
            if spec.name not in inputs:
                continue
            arr = inputs[spec.name]

            expected_dtype = np.dtype(spec.dtype)
            if arr.dtype != expected_dtype:
                errors.append(
                    f"{spec.name}: expected dtype {expected_dtype}, "
                    f"got {arr.dtype}"
                )

            if len(arr.shape) != len(spec.shape):
                errors.append(
                    f"{spec.name}: expected {len(spec.shape)} dims, "
                    f"got {len(arr.shape)}"
                )
            else:
                for i, (actual, expected) in enumerate(
                    zip(arr.shape, spec.shape)
                ):
                    if expected != -1 and actual != expected:
                        errors.append(
                            f"{spec.name}: dim {i} expected {expected}, "
                            f"got {actual}"
                        )

            if spec.min_value is not None and float(np.min(arr)) < spec.min_value:
                errors.append(
                    f"{spec.name}: values below minimum {spec.min_value}"
                )
            if spec.max_value is not None and float(np.max(arr)) > spec.max_value:
                errors.append(
                    f"{spec.name}: values above maximum {spec.max_value}"
                )

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def warmup(self, n: int = 5) -> WarmupResult:
        """Run *n* dummy inferences for JIT / cache warming.

        The default implementation generates random data from the schema.
        Override if your model requires special warm-up inputs.
        """
        schema = self.get_schema()
        dummy: dict[str, npt.NDArray[Any]] = {}
        for spec in schema.inputs:
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
        """Estimate memory footprint in bytes.

        Override to provide an accurate number.  The default returns 0.
        """
        return 0

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        schema = self.get_schema()
        n_in = len(schema.inputs)
        n_out = len(schema.outputs)
        return (
            f"<{self.__class__.__name__} "
            f"inputs={n_in} outputs={n_out} "
            f"desc={schema.description!r}>"
        )
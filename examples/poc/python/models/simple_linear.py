"""
Example: Simple linear model — y = W·x + b

Minimum viable implementation of MiiaModel.
Place in models/ directory to be loaded by the PythonBackend.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from miia_model import MiiaModel, ModelSchema, TensorSpec


class SimpleLinearModel(MiiaModel):

    def __init__(self) -> None:
        self._weights: npt.NDArray[np.float32] | None = None
        self._bias: npt.NDArray[np.float32] | None = None

    def load(self) -> None:
        self._weights = np.array(
            [[2.0, 1.0, 0.5, -1.0, 0.3]], dtype=np.float32
        )
        self._bias = np.array([0.1], dtype=np.float32)

    def predict(
        self, inputs: dict[str, npt.NDArray[Any]]
    ) -> dict[str, npt.NDArray[Any]]:
        x = inputs["input"]
        assert self._weights is not None and self._bias is not None
        y: npt.NDArray[np.float32] = x @ self._weights.T + self._bias
        return {"output": y}

    def get_schema(self) -> ModelSchema:
        return ModelSchema(
            inputs=[
                TensorSpec(
                    name="input",
                    shape=[1, 5],
                    dtype="float32",
                    description="Feature vector with 5 elements",
                    min_value=-10.0,
                    max_value=10.0,
                ),
            ],
            outputs=[
                TensorSpec(
                    name="output",
                    shape=[1, 1],
                    dtype="float32",
                    description="Scalar prediction",
                ),
            ],
            description="Simple linear model: y = W·x + b",
            author="AsaMiia Example",
            tags={"type": "regression", "complexity": "trivial"},
        )

    def unload(self) -> None:
        self._weights = None
        self._bias = None

    def memory_usage_bytes(self) -> int:
        total = 0
        if self._weights is not None:
            total += self._weights.nbytes
        if self._bias is not None:
            total += self._bias.nbytes
        return total
"""
ShipAvoidance — Potential field navigation model for AsaMiia.

Replicates the C++ ShipAgent::reasoning() logic:
  - Attraction force toward the target steerpoint heading
  - Repulsion forces from hazard sources based on distance and safe distance

Input tensor "state":
  [toHeading, lat, lon, brg_0, dist_0, safeDist_0, brg_1, dist_1, safeDist_1, ...]

  - toHeading  : bearing to target steerpoint (degrees)
  - lat, lon   : own ship position (degrees)
  - Per hazard : bearing (deg), distance (m), safe distance (m)

Output tensor "heading":
  [hdgRes] — resulting commanded heading in degrees
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from miia_model import MiiaModel, ModelSchema, TensorSpec

_FIXED_FIELDS = 3       # toHeading, lat, lon
_FIELDS_PER_HAZARD = 3  # brg, dist, safeDist


class ShipAvoidance(MiiaModel):

    def __init__(self) -> None:
        self._exp_param: float = 2.0
        self._weight_attraction: float = 1.0

    def load(self) -> None:
        pass  # stateless — pure math, nothing to load

    def predict(
        self, inputs: dict[str, npt.NDArray[Any]]
    ) -> dict[str, npt.NDArray[Any]]:

        state = inputs["state"].flatten()

        if len(state) < _FIXED_FIELDS:
            return {"heading": np.array([[0.0]], dtype=np.float32)}

        to_heading = float(state[0])

        # --- Attraction force toward steerpoint ---
        xres = math.cos(math.radians(to_heading)) * self._weight_attraction
        yres = math.sin(math.radians(to_heading)) * self._weight_attraction

        # --- Repulsion forces from hazard sources ---
        hazard_data = state[_FIXED_FIELDS:]
        n_hazards = len(hazard_data) // _FIELDS_PER_HAZARD

        for i in range(n_hazards):
            base      = i * _FIELDS_PER_HAZARD
            brg       = float(hazard_data[base + 0])  # degrees
            dist      = float(hazard_data[base + 1])  # meters
            safe_dist = float(hazard_data[base + 2])  # meters

            if safe_dist <= 0.0:
                continue

            if dist > safe_dist * 2.0:
                weight_repulsion = 0.0
            elif dist < safe_dist:
                weight_repulsion = 1.0
            else:
                weight_repulsion = (
                    (safe_dist * 2.0 - dist) ** self._exp_param
                    / safe_dist ** self._exp_param
                )

            xres += math.cos(math.radians(brg)) * weight_repulsion
            yres += math.sin(math.radians(brg)) * weight_repulsion

        hdg_res = math.degrees(math.atan2(yres, xres))

        return {"heading": np.array([[hdg_res]], dtype=np.float32)}

    def get_schema(self) -> ModelSchema:
        return ModelSchema(
            inputs=[
                TensorSpec(
                    name="state",
                    shape=[1, -1],
                    dtype="float32",
                    description=(
                        "Flat state vector: [toHeading, lat, lon, "
                        "brg_0, dist_0, safeDist_0, ...]"
                    ),
                ),
            ],
            outputs=[
                TensorSpec(
                    name="heading",
                    shape=[1, 1],
                    dtype="float32",
                    description="Resulting commanded heading in degrees",
                ),
            ],
            description="Potential field navigation — attraction + repulsion",
            author="ASA Tutorial",
            tags={"type": "navigation", "algorithm": "potential_field"},
        )

    def unload(self) -> None:
        pass  # nothing to release

    def memory_usage_bytes(self) -> int:
        return 0  # stateless, no stored arrays
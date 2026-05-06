"""
ShipAvoidance — Potential field navigation model for AsaMiia.

Replicates the C++ ShipAgent::reasoning() logic:
  - Attraction force toward the target steerpoint heading
  - Repulsion forces from hazard sources based on distance and safe distance

Input estruturado "state" (dict):
  {
    "toHeading":  float,   # bearing ao steerpoint alvo (graus)
    "latitude":   float,   # posição do navio (graus)
    "longitude":  float,   # posição do navio (graus)
    "hazards": [           # lista de obstáculos detectados
      {
        "bearing":     float,  # rumo ao obstáculo (graus)
        "distance":    float,  # distância ao obstáculo (metros)
        "minSafeDist": float,  # distância mínima de segurança (metros)
      },
      ...
    ]
  }

Output tensor "heading":
  [hdgRes] — rumo comandado resultante em graus
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from miia_model import MiiaModel, ModelSchema, TensorSpec


class ShipAvoidance(MiiaModel):

    def __init__(self) -> None:
        self._exp_param: float = 2.0
        self._weight_attraction: float = 1.0

    def load(self) -> None:
        pass  # stateless — pure math, nothing to load

    def predict(self, inputs: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
        state = inputs.get("state", {})

        if not state:
            return {"heading": np.array([[0.0]], dtype=np.float32)}

        to_heading = float(state.get("toHeading", 0.0))

        # --- Força de atração em direção ao steerpoint ---
        xres = math.cos(math.radians(to_heading)) * self._weight_attraction
        yres = math.sin(math.radians(to_heading)) * self._weight_attraction

        # --- Forças de repulsão dos obstáculos ---
        for hazard in state.get("hazards", []):
            brg       = float(hazard.get("bearing",     0.0))
            dist      = float(hazard.get("distance",    0.0))
            safe_dist = float(hazard.get("minSafeDist", 0.0))

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
                    shape=[-1],
                    dtype="float32",
                    structured=True,
                    description=(
                        "Estado estruturado do navio: "
                        "{ toHeading, latitude, longitude, "
                        "hazards: [{ bearing, distance, minSafeDist }] }"
                    ),
                ),
            ],
            outputs=[
                TensorSpec(
                    name="heading",
                    shape=[1, 1],
                    dtype="float32",
                    description="Rumo comandado resultante em graus",
                ),
            ],
            description="Campo potencial — atração ao steerpoint + repulsão de obstáculos",
            author="ASA Tutorial",
            tags={"type": "navigation", "algorithm": "potential_field"},
        )

    def unload(self) -> None:
        pass  # nothing to release

    def memory_usage_bytes(self) -> int:
        return 0  # stateless

    def warmup(self, n: int = 5):
        """Warmup com input estruturado representativo."""
        from miia_model import WarmupResult
        import time

        dummy_input = {
            "state": {
                "toHeading": 45.0,
                "latitude":  -23.5,
                "longitude": -46.6,
                "hazards": [
                    {"bearing": 90.0,  "distance": 500.0, "minSafeDist": 300.0},
                    {"bearing": 180.0, "distance": 800.0, "minSafeDist": 300.0},
                ],
            }
        }

        times: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(dummy_input)
            times.append((time.perf_counter() - t0) * 1000.0)

        return WarmupResult(
            runs_completed=len(times),
            avg_time_ms=sum(times) / len(times) if times else 0.0,
            min_time_ms=min(times) if times else 0.0,
            max_time_ms=max(times) if times else 0.0,
        )
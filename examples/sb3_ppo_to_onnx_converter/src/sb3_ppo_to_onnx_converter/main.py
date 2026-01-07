"""
Módulo para carregar um modelo treinado com Proximal Policy Optimization (PPO)
e exportá-lo para ONNX.
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy


# ---------------------------------------------------------------------
# ONNX wrapper
# ---------------------------------------------------------------------

class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return self.policy(observation, deterministic=True)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_trained_model(model_path: Path) -> PPO:
    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

    if model_path.suffix != ".zip":
        raise ValueError(f"O arquivo do modelo deve ser .zip, recebido: {model_path}")

    model = PPO.load(model_path, device="cpu")
    print(f"Modelo carregado: {model_path}")
    return model


# ---------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------

def export_model_to_onnx(model: PPO, onnx_path: Path):
    onnx_policy = OnnxableSB3Policy(model.policy)

    if not model.observation_space.shape:
        raise RuntimeError("Observation space inválido ou indefinido")

    dummy_input = th.randn(1, *model.observation_space.shape)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    th.onnx.export(
        onnx_policy,
        dummy_input,
        onnx_path.as_posix(),
        opset_version=18,
        dynamo=False, # Isto por default é True, o que é a forma mais "moderna" e recomendada, mas está false por ser menos verbose
        input_names=["observation"],
        output_names=["action", "value", "log_prob"],
        dynamic_axes={"observation": {0: "batch"}}
    )

    print(f"Modelo exportado para ONNX: {onnx_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exporta um modelo PPO treinado para formato ONNX."
    )

    parser.add_argument(
        "model_path",
        type=Path,
        help="Caminho para o arquivo .zip do modelo PPO"
    )

    parser.add_argument(
        "onnx_output",
        type=Path,
        help="Caminho de saída do arquivo ONNX (e.g. output/model.onnx)"
    )

    args = parser.parse_args()

    try:
        model = load_trained_model(args.model_path)
        export_model_to_onnx(model, args.onnx_output)
    except Exception as e:
        print(f"Erro: {e}")
        exit(1)


if __name__ == "__main__":
    main()
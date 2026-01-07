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

import onnxruntime as ort
import numpy as np


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


def verify_export(model: PPO, onnx_path: Path, n_tests: int = 5, atol: float = 1e-5):
    print("\nVerificando equivalência PyTorch <---> ONNX")

    ort_session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])

    for i in range(n_tests):
        obs = th.randn(1, *model.observation_space.shape)

        with th.no_grad():
            torch_out = model.policy(obs, deterministic=True)

        ort_inputs = {"observation": obs.numpy()}
        onnx_out = ort_session.run(None, ort_inputs)

        torch_action = torch_out[0].numpy()

        diff = np.abs(torch_action - onnx_out[0])

        print(
            f"Teste {i+1}: "
            f"max error = {diff.max():.6f} | "
            f"mean error = {diff.mean():.6f}"
        )

        if not np.allclose(torch_action, onnx_out[0], atol=atol):
            raise RuntimeError("Diferença numérica acima da tolerância")

    print("Verificação concluída: exportação válida.")


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
        verify_export(model, args.onnx_output)
    except Exception as e:
        print(f"Erro: {e}")
        exit(1)


if __name__ == "__main__": # pragma: no cover
    main()
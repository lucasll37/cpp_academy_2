"""Módulo para carregar um modelo treinado com Proximal Policy Optimization (PPO) de um caminho fornecido por linha de comando."""

import torch as th
import argparse
from pathlib import Path
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

import onnx
import onnxruntime as ort
import numpy as np

# 1. Define a wrapper class to make the policy onnx-compatible
class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)

def load_trained_model(model_path):
    """
    Carrega um modelo PPO de um caminho de arquivo zip fornecido.
    
    Args:
        model_path (str): O caminho no sistema para o arquivo .zip que contém os dados do modelo.
    """
    # Path da pathlib tem compatibilidade cross-platform
    path = Path(model_path)
    
    if not path.exists():
        print(f"Erro: O arquivo {model_path} não existe.")
        return

    # Carrega o modelo; device='cpu' é setado explicitamente
    model = PPO.load(path, device="cpu")
    print(f"Modelo carregado com sucesso de: {path}")
    return model


def export_model_to_onnx(model, onnx_path="my_ppo_model.onnx"):
    """Função isolada para exportação para evitar problemas de tracing em testes."""
    
    # 3. Create an instance of the Onnxable policy
    onnx_policy = OnnxableSB3Policy(model.policy)

    # 4. Create a dummy input tensor matching the observation space
    observation_size = model.observation_space.shape
    dummy_input = th.randn(1, *observation_size)

    # 5. Export the model to ONNX format
    th.onnx.export(
        onnx_policy,
        dummy_input,
        "my_ppo_model.onnx",
        opset_version=18,
        input_names=["input"]
    )


def main():
    """Entry point para fazer o parsing dos argumentos de linha de comando e carregamento do modelo."""
    parser = argparse.ArgumentParser(
        description="Carrega um modelo treinado SB3 PPO."
    )
    
    # Adicionar argumento posicional para o caminho do modelo
    parser.add_argument(
        "model_path", 
        type=str, 
        help="Caminho para o arquivo .zip do modelo treinado (e.g., model.zip)"
    )

    args = parser.parse_args()
    
    # Executa o carregamento
    model = load_trained_model(args.model_path)

    # Verifica se o modelo foi carregado antes de prosseguir
    if model:
        export_model_to_onnx(model)


if __name__ == "__main__":
    main()
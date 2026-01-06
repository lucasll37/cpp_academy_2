"""Módulo para carregar um modelo treinado com Proximal Policy Optimization (PPO) de um caminho fornecido por linha de comando."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO


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
    load_trained_model(args.model_path)


if __name__ == "__main__":
    main()

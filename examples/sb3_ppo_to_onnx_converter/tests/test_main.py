import pytest
from pathlib import Path
from sb3_ppo_to_onnx_converter.main import load_trained_model, main

def test_load_trained_model_file_not_found(capsys):
    """Teste se o carregador lida corretamente com arquivos ausentes."""
    load_trained_model("non_existent_model.zip")
    captured = capsys.readouterr()
    assert "Erro: O arquivo non_existent_model.zip não existe." in captured.out

def test_load_trained_model_success(mocker, tmp_path):
    """Teste o carregamento bem-sucedido do modelo usando um mock."""
    # Crie um arquivo fictício para passar na verificação de existência.
    dummy_path = tmp_path / "model.zip"
    dummy_path.write_text("dummy data")
    
    # Simule o PPO.load para que não executemos código pesado de fato.
    mock_ppo_load = mocker.patch("sb3_ppo_to_onnx_converter.main.PPO.load")
    
    load_trained_model(str(dummy_path))
    
    mock_ppo_load.assert_called_once_with(Path(dummy_path), device="cpu")

def test_cli_argument_parsing(mocker, tmp_path):
    """Teste se a CLI passa corretamente o caminho para o carregador."""
    dummy_path = tmp_path / "cli_model.zip"
    dummy_path.write_text("dummy")
    
    # Mock da função de carregamento interna para verificar se ela é chamada pela função main().
    mock_loader = mocker.patch("sb3_ppo_to_onnx_converter.main.load_trained_model")
    
    # Mock do sys.argv usando o seguinte comando: sb3ppo2onnx cli_model.zip
    mocker.patch("sys.argv", ["sb3ppo2onnx", str(dummy_path)])
    
    main()
    mock_loader.assert_called_once_with(str(dummy_path))
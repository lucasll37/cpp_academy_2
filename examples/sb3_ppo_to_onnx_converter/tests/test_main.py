import pytest
from pathlib import Path

from sb3_ppo_to_onnx_converter.main import (
    load_trained_model,
    export_model_to_onnx,
    main,
)

# ---------------------------------------------------------------------
# load_trained_model
# ---------------------------------------------------------------------

def test_load_trained_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_trained_model(Path("non_existent_model.zip"))


def test_load_trained_model_invalid_extension(tmp_path):
    bad_file = tmp_path / "model.txt"
    bad_file.write_text("nope")

    with pytest.raises(ValueError):
        load_trained_model(bad_file)


def test_load_trained_model_success(mocker, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_text("fake model")

    mock_ppo_load = mocker.patch(
        "sb3_ppo_to_onnx_converter.main.PPO.load",
        return_value="mock_model"
    )

    model = load_trained_model(model_path)

    mock_ppo_load.assert_called_once_with(model_path, device="cpu")
    assert model == "mock_model"


# ---------------------------------------------------------------------
# export_model_to_onnx
# ---------------------------------------------------------------------

def test_export_model_to_onnx_calls_torch_export(mocker, tmp_path):
    fake_model = mocker.Mock()
    fake_model.observation_space.shape = (4,)
    fake_model.policy = mocker.Mock()

    mock_export = mocker.patch("sb3_ppo_to_onnx_converter.main.th.onnx.export")

    output_path = tmp_path / "out.onnx"

    export_model_to_onnx(fake_model, output_path)

    mock_export.assert_called_once()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def test_cli_argument_parsing(mocker, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_text("fake model")

    onnx_path = tmp_path / "model.onnx"

    mock_model = mocker.Mock()

    mock_loader = mocker.patch(
        "sb3_ppo_to_onnx_converter.main.load_trained_model",
        return_value=mock_model
    )

    mock_exporter = mocker.patch(
        "sb3_ppo_to_onnx_converter.main.export_model_to_onnx"
    )

    mocker.patch(
        "sys.argv",
        ["sb3ppo2onnx", str(model_path), str(onnx_path)]
    )

    main()

    mock_loader.assert_called_once_with(model_path)
    mock_exporter.assert_called_once_with(mock_model, onnx_path)

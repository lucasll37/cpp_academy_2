import pytest
import numpy as np
from pathlib import Path

from sb3_ppo_to_onnx_converter.main import (
    OnnxableSB3Policy,
    load_trained_model,
    export_model_to_onnx,
    verify_export,
    main,
)

# ---------------------------------------------------------------------
# ONNX wrapper
# ---------------------------------------------------------------------

def test_onnxable_policy_forward_calls_wrapped_policy(mocker):
    fake_policy = mocker.Mock()
    fake_policy.return_value = ("action", "value", "log_prob")

    wrapper = OnnxableSB3Policy(fake_policy)

    obs = mocker.Mock()

    result = wrapper(obs)

    fake_policy.assert_called_once_with(obs, deterministic=True)
    assert result == ("action", "value", "log_prob")

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


def test_export_creates_directory_and_prints(mocker, tmp_path, capsys):
    fake_model = mocker.Mock()
    fake_model.observation_space.shape = (4,)
    fake_model.policy = mocker.Mock()

    mocker.patch("sb3_ppo_to_onnx_converter.main.th.onnx.export")

    nested_path = tmp_path / "deep" / "folder" / "model.onnx"

    export_model_to_onnx(fake_model, nested_path)

    assert nested_path.parent.exists()

    captured = capsys.readouterr()
    assert "Modelo exportado para ONNX" in captured.out

def test_export_raises_on_invalid_observation_space(mocker, tmp_path):
    fake_model = mocker.Mock()
    fake_model.observation_space.shape = None
    fake_model.policy = mocker.Mock()

    with pytest.raises(RuntimeError):
        export_model_to_onnx(fake_model, tmp_path / "out.onnx")

def test_verify_export_passes_when_outputs_match(mocker, tmp_path):
    fake_model = mocker.Mock()
    fake_model.observation_space.shape = (4,)

    fake_policy = mocker.Mock()
    fake_model.policy = fake_policy

    # Fake torch output
    torch_out = (mocker.Mock(), None, None)
    torch_out[0].numpy.return_value = np.array([[1.0, 2.0, 3.0, 4.0]])
    fake_policy.return_value = torch_out

    # Fake ONNX output
    fake_session = mocker.Mock()
    fake_session.run.return_value = [np.array([[1.0, 2.0, 3.0, 4.0]])]

    mocker.patch("sb3_ppo_to_onnx_converter.main.ort.InferenceSession", return_value=fake_session)
    mocker.patch("sb3_ppo_to_onnx_converter.main.th.randn", return_value=mocker.Mock())
    mocker.patch("sb3_ppo_to_onnx_converter.main.th.no_grad")

    verify_export(fake_model, tmp_path / "model.onnx", n_tests=1)

def test_verify_export_fails_when_outputs_differ(mocker, tmp_path):
    fake_model = mocker.Mock()
    fake_model.observation_space.shape = (4,)

    fake_policy = mocker.Mock()
    fake_model.policy = fake_policy

    torch_out = (mocker.Mock(), None, None)
    torch_out[0].numpy.return_value = np.array([[1.0, 2.0, 3.0, 4.0]])
    fake_policy.return_value = torch_out

    fake_session = mocker.Mock()
    fake_session.run.return_value = [np.array([[9.0, 9.0, 9.0, 9.0]])]

    mocker.patch("sb3_ppo_to_onnx_converter.main.ort.InferenceSession", return_value=fake_session)
    mocker.patch("sb3_ppo_to_onnx_converter.main.th.randn", return_value=mocker.Mock())
    mocker.patch("sb3_ppo_to_onnx_converter.main.th.no_grad")

    with pytest.raises(RuntimeError):
        verify_export(fake_model, tmp_path / "model.onnx", n_tests=1)

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

    mocker.patch("sb3_ppo_to_onnx_converter.main.verify_export")

    mocker.patch(
        "sys.argv",
        ["sb3ppo2onnx", str(model_path), str(onnx_path)]
    )

    main()

    mock_loader.assert_called_once_with(model_path)
    mock_exporter.assert_called_once_with(mock_model, onnx_path)


def test_main_handles_exception_and_exits(mocker, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_text("fake")

    onnx_path = tmp_path / "out.onnx"

    mocker.patch(
        "sb3_ppo_to_onnx_converter.main.load_trained_model",
        side_effect=RuntimeError("boom")
    )

    mocker.patch("sb3_ppo_to_onnx_converter.main.verify_export")

    mock_exit = mocker.patch("sb3_ppo_to_onnx_converter.main.exit")

    mocker.patch(
        "sys.argv",
        ["sb3ppo2onnx", str(model_path), str(onnx_path)]
    )

    main()

    mock_exit.assert_called_once_with(1)
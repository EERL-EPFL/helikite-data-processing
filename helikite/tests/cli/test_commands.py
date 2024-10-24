from typer.testing import CliRunner
from helikite import app
import os

runner = CliRunner()


def test_generate_config_create_with_input_dir(tmpdir: str):
    input_folder = os.path.join(tmpdir, "inputs")
    config_filename = "configtest.yaml"
    config_file_path = os.path.join(input_folder, config_filename)

    # Check that the folder doesn't exist first
    assert os.path.exists(input_folder) is False

    result = runner.invoke(
        app,
        [
            "generate-config",
            "--input-folder",
            input_folder,
            "--config-file",
            config_filename,
        ],
    )

    # Check that the folder has been created
    assert result.exit_code == 0
    assert os.path.exists(input_folder) is True
    assert os.path.exists(config_file_path) is True


def test_cli_preprocess(tmpdir: str):
    input_folder = os.path.join(tmpdir, "inputs")
    config_filename = "configtest.yaml"
    config_file_path = os.path.join(input_folder, config_filename)

    # Check that the folder doesn't exist first
    assert os.path.exists(input_folder) is False

    result = runner.invoke(
        app,
        [
            "preprocess",
            "--input-folder",
            input_folder,
            "--config-file",
            config_filename,
        ],
    )

    # Check that the folder has been created
    assert result.exit_code == 0
    assert os.path.exists(input_folder) is True
    assert os.path.exists(config_file_path) is True


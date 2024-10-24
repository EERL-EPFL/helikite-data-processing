from typer.testing import CliRunner
from helikite import app
import os

runner = CliRunner()


def test_generate_config(tmpdir):
    input_folder = tmpdir.mkdir("inputs")
    config_filename = "generated_config.yaml"
    config_file_path = os.path.join(input_folder, config_filename)

    # Invoke the CLI to generate the configuration
    result = runner.invoke(
        app,
        [
            "generate-config",
            "--input-folder",
            str(input_folder),
            "--config-file",
            config_filename,
        ],
    )

    # Assertions
    assert result.exit_code == 0
    assert os.path.exists(
        config_file_path
    ), "Configuration file was not created"

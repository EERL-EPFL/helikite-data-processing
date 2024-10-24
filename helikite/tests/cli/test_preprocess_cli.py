from typer.testing import CliRunner
from helikite import app
import os
import shutil

runner = CliRunner()


def test_preprocess(tmpdir):
    # Set up input directory and copy test data
    resource_folder = os.path.join(
        os.path.dirname(__file__), "..", "resources", "campaigns", "20220929"
    )
    input_folder = tmpdir.mkdir("inputs")
    config_filename = "test_config.yaml"
    config_file_path = os.path.join(input_folder, config_filename)

    for filename in os.listdir(resource_folder):
        shutil.copy(os.path.join(resource_folder, filename), input_folder)

    # Generate the configuration
    runner.invoke(
        app,
        [
            "generate-config",
            "--input-folder",
            str(input_folder),
            "--config-file",
            config_filename,
        ],
    )

    # Run preprocessing
    result = runner.invoke(
        app,
        [
            "preprocess",
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
    ), "Config file not found after preprocessing"

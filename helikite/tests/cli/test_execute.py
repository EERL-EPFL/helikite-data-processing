from typer.testing import CliRunner
from helikite import app
import os
import shutil

runner = CliRunner()


def test_execute_pipeline(tmpdir):
    # Set up input and output directories
    resource_folder = os.path.join(
        os.path.dirname(__file__), "..", "resources", "campaigns", "20220929"
    )
    input_folder = tmpdir.mkdir("inputs")
    output_folder = tmpdir.mkdir("outputs")
    config_filename = os.path.join(input_folder, "test_config.yaml")

    # Copy resources into input folder
    for filename in os.listdir(resource_folder):
        src_path = os.path.join(resource_folder, filename)
        dst_path = os.path.join(input_folder, filename)
        shutil.copy(src_path, dst_path)

    # Generate configuration
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
    assert result.exit_code == 0, f"Generate config failed: {result.stdout}"

    # Preprocess
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
    assert result.exit_code == 0, f"Preprocessing failed: {result.stdout}"

    # Execute the pipeline
    result = runner.invoke(
        app,
        [
            "execute",
            "--config-file",
            config_filename,
            "--input-folder",
            str(input_folder),
            "--output-folder",
            str(output_folder),
        ],
    )

    # Assertions
    assert (
        result.exit_code == 0
    ), f"Command failed with exit code {result.exit_code} '{result.stdout}'"
    assert len(os.listdir(output_folder)) > 0, "No output files were generated"

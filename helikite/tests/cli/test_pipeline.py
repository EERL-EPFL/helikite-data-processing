from typer.testing import CliRunner
from helikite import app
import os
import shutil

runner = CliRunner()


def test_full_processing_pipeline(tmpdir):
    # Set up input and output directories
    resource_folder = os.path.join(
        os.path.dirname(__file__), "..", "resources", "campaigns", "20220929"
    )
    input_folder = tmpdir.mkdir("inputs")
    output_folder = tmpdir.mkdir("outputs")
    config_filename = os.path.join(input_folder, "full_pipeline_config.yaml")

    # Copy resources into input folder
    for filename in os.listdir(resource_folder):
        src_path = os.path.join(resource_folder, filename)
        dst_path = os.path.join(input_folder, filename)
        shutil.copy(src_path, dst_path)

    # Step 1: Generate config
    generate_result = runner.invoke(
        app,
        [
            "generate-config",
            "--input-folder",
            str(input_folder),
            "--config-file",
            config_filename,
        ],
    )
    assert (
        generate_result.exit_code == 0
    ), f"Generate config failed: {generate_result.stdout}"

    # Step 2: Preprocess
    preprocess_result = runner.invoke(
        app,
        [
            "preprocess",
            "--input-folder",
            str(input_folder),
            "--config-file",
            config_filename,
        ],
    )
    assert (
        preprocess_result.exit_code == 0
    ), f"Preprocessing failed: {preprocess_result.stdout}"

    # Step 3: Execute full processing
    execute_result = runner.invoke(
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
    assert (
        execute_result.exit_code == 0
    ), f"Execution failed: {execute_result.stdout}"

    # Handle timestamped output directory
    output_subdirs = os.listdir(output_folder)
    assert (
        len(output_subdirs) == 1
    ), f"Expected one timestamped folder in outputs, found: {output_subdirs}"
    generated_output_dir = os.path.join(output_folder, output_subdirs[0])

    # Assertions for output content
    output_files = os.listdir(generated_output_dir)
    assert len(output_files) > 0, "No files generated in the output directory"
    expected_files = [
        "helikite-data.csv",
        "helikite-housekeeping.csv",
        "helikite-quicklooks.html",
        "helikite-qualitycheck.html",
    ]
    assert all(
        file in output_files for file in expected_files
    ), f"Expected files not found, files: {output_files}"

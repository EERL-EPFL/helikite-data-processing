from pydantic_settings import BaseSettings
from pathlib import Path
import logging
import toml
import os
from functools import lru_cache


class Constants(BaseSettings):
    # Get the following data number from pyproject.toml in get_constants()
    APPLICATION_NAME: str
    APPLICATION_NAME_PYTHON: str
    VERSION: str
    DESCRIPTION: str

    INPUTS_FOLDER: Path = Path.cwd().joinpath("inputs")
    OUTPUTS_FOLDER: Path = Path.cwd().joinpath("outputs")
    OUTPUTS_INSTRUMENT_SUBFOLDER: str = "instruments"
    CONFIG_FILE: str = "config.yaml"
    MASTER_CSV_FILENAME: str = "helikite-data.csv"
    HOUSEKEEPING_CSV_FILENAME: str = "helikite-housekeeping.csv"
    HOUSEKEEPING_VAR_PRESSURE: str = "housekeeping_pressure"
    LOGFILE_NAME: str = "helikite.log"
    LOGLEVEL_CONSOLE: str = "INFO"
    LOGLEVEL_FILE: str = "DEBUG"
    QTY_LINES_TO_IDENTIFY_INSTRUMENT: int = 50

    # Column names
    ALTITUDE_GROUND_LEVEL_COL: str = "flight_computer_Altitude_agl"
    ALTITUDE_SEA_LEVEL_COL: str = "flight_computer_Altitude"

    # Plots
    QUICKLOOK_PLOT_FILENAME: str = "helikite-quicklooks.html"
    QUALITYCHECK_PLOT_FILENAME: str = "helikite-qualitycheck.html"
    PLOT_LAYOUT_COMMON: dict = {
        "font": {
            "size": 16,
            "family": "Arial",
        },
        "template": "plotly_white",
        "height": 600,
    }
    PLOT_MARKER_SIZE: int = 8

    # Logging
    LOGFORMAT_CONSOLE: logging.Formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7.7s] %(message)s"
    )
    LOGFORMAT_FILE: logging.Formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7.7s] (%(name)25.25s) %(message)s"
    )


@lru_cache()
def get_constants():
    file_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    )

    with open(os.path.join(file_dir, "pyproject.toml"), "r") as f:
        pyproject = toml.load(f)
        pyproject_version = pyproject["tool"]["poetry"]["version"]
        application_name = pyproject["tool"]["poetry"]["name"]
        application_name_python = application_name.replace("-", "_")
        description = pyproject["tool"]["poetry"]["description"]

    return Constants(
        VERSION=pyproject_version,
        APPLICATION_NAME=application_name,
        APPLICATION_NAME_PYTHON=application_name_python,
        DESCRIPTION=description,
    )


constants = get_constants()

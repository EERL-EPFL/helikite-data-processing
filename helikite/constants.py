from pydantic_settings import BaseSettings
from pathlib import Path
import logging
from functools import lru_cache


class Constants(BaseSettings):
    INPUTS_FOLDER: Path = Path.cwd().joinpath("inputs")
    OUTPUTS_FOLDER: Path = Path.cwd().joinpath("outputs")
    OUTPUTS_INSTRUMENT_SUBFOLDER: str = "instruments"
    CONFIG_FILE: str = "config.yaml"
    MASTER_CSV_FILENAME: str = "helikite-data.csv"
    HOUSEKEEPING_CSV_FILENAME: str = "helikite-housekeeping.csv"
    HOUSEKEEPING_VAR_PRESSURE: str = "pressure"
    LOGFILE_NAME: str = "helikite.log"
    LOGLEVEL_CONSOLE: str = "INFO"
    LOGLEVEL_FILE: str = "DEBUG"
    QTY_LINES_TO_IDENTIFY_INSTRUMENT: int = 60

    # Cross correlation
    CROSSCORRELATION_DEFAULT_LAG: int = 10
    ROLLING_WINDOW_DEFAULT_SIZE: int = 20
    ROLLING_WINDOW_COLUMN_NAME: str = "pressure_rn"

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

    # with open(os.path.join(file_dir, "pyproject.toml"), "r") as f:
    # pyproject = toml.load(f)
    # pyproject_version = pyproject["tool"]["poetry"]["version"]
    # application_name = pyproject["tool"]["poetry"]["name"]
    # application_name_python = application_name.replace("-", "_")
    # description = pyproject["tool"]["poetry"]["description"]
    # from . import __version__, __name__, __description__

    return Constants()
    # VERSION=__version__,
    # APPLICATION_NAME=__name__,
    # APPLICATION_NAME_PYTHON=__name__.replace("-", "_"),
    # DESCRIPTION=__description__,
    # )


constants = get_constants()

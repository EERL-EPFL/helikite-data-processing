#!/usr/bin/env python3

import yaml
from helikite.constants import constants
from helikite import instruments
import os
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


def get_columns_from_dtype(instrument: instruments.base.Instrument):
    """Gets the column names from the instrument config"""

    return list(instrument.dtype)


def read_yaml_config(file_path):
    # Open YAML
    with open(file_path, "r") as in_yaml:
        yaml_config = yaml.load(in_yaml, Loader=yaml.Loader)

    return yaml_config


def preprocess(
    input_folder=constants.INPUTS_FOLDER,
    config_file=constants.CONFIG_FILE,
):
    yaml_config = read_yaml_config(os.path.join(input_folder, config_file))

    # Hold a list of the loaded instrument objects
    instrument_objects = {}
    for instrument, props in yaml_config["instruments"].items():
        instrument_objects[instrument] = getattr(instruments, props["config"])

    for instrument_name, instrument_obj in instrument_objects.items():
        matched_file = instrument_obj.detect_from_folder(
            input_folder,
            lines_to_read=constants.QTY_LINES_TO_IDENTIFY_INSTRUMENT,
        )

        if matched_file:
            props = yaml_config["instruments"][instrument_name]

            # Set filename in config
            props["file"] = matched_file

            # Get the date if it is in the header
            with open(matched_file) as in_file:
                header_lines = [
                    next(in_file)
                    for _ in range(constants.QTY_LINES_TO_IDENTIFY_INSTRUMENT)
                ]
                props["date"] = instrument_obj.date_extractor(header_lines)

    # Write out the updated yaml configuration
    print_preprocess_stats(yaml_config)
    export_yaml_config(
        yaml_config,
        os.path.join(input_folder, config_file),
    )


def print_preprocess_stats(yaml_config):
    found = []
    not_found = []
    for instrument, props in yaml_config["instruments"].items():
        if props["file"] is None:
            not_found.append(instrument)
        else:
            found.append(instrument)

    logger.info("File preprocessing statistics:")
    logger.info(f"Missing: {len(not_found)}: {', '.join(not_found)}")
    logger.info(f"Found:   {len(found)}: {', '.join(found)}")


def export_yaml_config(yaml_config, out_location):
    logger.info(f"Writing YAML config to {out_location}")
    # Update YAML (will remove all commented out inputs)
    with open(out_location, "w") as in_yaml:
        yaml.dump(yaml_config, in_yaml)


def generate_config(
    overwrite=False,
    input_folder=constants.INPUTS_FOLDER,
    config_file=constants.CONFIG_FILE,
) -> None:
    """Write out the configuration file

    If overwrite is True, the configuration file in the path argument is
    overwritten

    """

    # Create folder if it does not exist
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        logger.info(f"Created folder: {input_folder}")

    path = os.path.join(input_folder, config_file)
    if overwrite is False and os.path.exists(path):
        # Escape if overwrite if false and the file exists
        return

    logger.info(f"Generating YAML configuration in {path}")

    # Go through each instrument in the __init__ of config.instrument
    instrument_objects = instruments.__dict__.items()
    yaml_config: Dict[str, Any] = {}
    yaml_config["instruments"] = {}
    yaml_config["global"] = {
        "time_trim": {
            "start": None,
            "end": None,
        },
    }
    yaml_config["ground_station"] = {
        "altitude": None,
        "temperature": None,
        "pressure": None,
    }
    yaml_config["plots"] = {
        "altitude_ground_level": False,
        "grid": {"resample_seconds": None},
        "heatmap": {
            "msems_inverted": {
                "zmin": None,
                "zmax": None,
                "zmid": None,
            },
            "msems_scan": {
                "zmin": None,
                "zmax": None,
                "zmid": None,
            },
        },
        "msems_readings_averaged": {
            "Title1": {
                "time_start": None,
                "time_end": None,
                "log_y": False,
            }
        },
    }

    for instrument, obj in instrument_objects:
        # If the imported object is actually an Instrument, then proceed
        if isinstance(obj, instruments.base.Instrument):
            yaml_config["instruments"][instrument] = {
                "config": instrument,
                "file": None,
                "date": None,
                "time_offset": {
                    "hour": 0,
                    "minute": 0,
                    "second": 0,
                },
                "pressure_offset": None,
            }
    export_yaml_config(yaml_config, path)

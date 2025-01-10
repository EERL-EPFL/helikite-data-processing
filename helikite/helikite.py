import sys
from helikite.processing import preprocess, sorting
from helikite.constants import constants
from helikite import instruments
from helikite import plots
import pandas as pd
import os
import datetime
import logging
from typing import Dict, Any
import typer


# Define a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(constants.LOGLEVEL_CONSOLE)
console_handler.setFormatter(constants.LOGFORMAT_CONSOLE)
logging.getLogger().addHandler(console_handler)

# Define logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.command()
def generate_config(
    overwrite: bool = typer.Option(
        False, help="Overwrite the existing configuration file"
    ),
    input_folder: str = constants.INPUTS_FOLDER,
    config_file: str = constants.CONFIG_FILE,
) -> None:
    """Generate a configuration file in the input folder"""

    preprocess.generate_config(
        overwrite=overwrite,
        input_folder=input_folder,
        config_file=config_file,
    )


@app.command("preprocess")
def preprocess_data(
    overwrite: bool = typer.Option(
        False, help="Overwrite the existing configuration file"
    ),
    config_file: str = typer.Option(
        constants.CONFIG_FILE,
        help="The configuration file to use",
        show_default=True,
    ),
    input_folder: str = constants.INPUTS_FOLDER,
    # output_folder: str = constants.OUTPUTS_FOLDER,
) -> None:
    """Preprocess the data and generate the configuration file"""

    preprocess.generate_config(
        overwrite=overwrite,
        input_folder=input_folder,
        config_file=config_file,
    )
    preprocess.preprocess(
        input_folder=input_folder,
        config_file=config_file,
    )


@app.command()
def execute(
    config_file: str = typer.Option(
        constants.CONFIG_FILE,
        help="The configuration file to use",
        show_default=True,
    ),
    input_folder: str = constants.INPUTS_FOLDER,
    output_folder: str = constants.OUTPUTS_FOLDER,
) -> None:
    """Execute the main processing and plotting of the data"""
    try:
        config = preprocess.read_yaml_config(
            os.path.join(input_folder, config_file)
        )

        main(config=config, output_path=output_folder)
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        raise e


def main(
    config: Dict[str, Any],
    output_path: str = constants.OUTPUTS_FOLDER,
) -> None:
    """Main function to run the processing and plotting of data

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary of the configuration file

    Returns
    -------
    None

    """

    # Create a folder with the current UTC time in outputs
    output_path_with_time = os.path.join(
        output_path, datetime.datetime.now(datetime.UTC).isoformat()
    )
    output_path_instrument_subfolder = os.path.join(
        output_path_with_time, constants.OUTPUTS_INSTRUMENT_SUBFOLDER
    )
    os.makedirs(output_path_with_time)
    os.makedirs(output_path_instrument_subfolder)

    # Add a file logger
    logfile_handler = logging.FileHandler(
        os.path.join(output_path_with_time, constants.LOGFILE_NAME)
    )
    logfile_handler.setLevel(constants.LOGLEVEL_FILE)
    logfile_handler.setFormatter(constants.LOGFORMAT_FILE)
    logging.getLogger().addHandler(logfile_handler)

    # Append each df for merging
    all_export_dfs = []

    time_trim_start = pd.to_datetime(config["global"]["time_trim"]["start"])
    time_trim_end = pd.to_datetime(config["global"]["time_trim"]["end"])

    ground_station = config["ground_station"]
    plot_props = config["plots"]

    if "instruments" not in config:
        logger.error("No instruments found in configuration, exiting")
        return

    # Go through each instrument and perform the operations on each instrument
    for instrument, props in config["instruments"].items():

        instrument_obj = getattr(instruments, props["config"])
        instrument_obj.add_config(props)

        if instrument_obj.filename is None:
            logger.warning(f"Skipping {instrument}: No file assigned!")
            continue
        else:
            logger.info(f"Processing {instrument}: {instrument_obj.filename}")

        df = instrument_obj.read_data()

        # Modify the DateTime index based off the configuration offsets
        df = instrument_obj.set_time_as_index(df)

        # Using the time corrections from configuration, correct time index
        df = instrument_obj.correct_time_from_config(
            df, time_trim_start, time_trim_end
        )
        if len(df) == 0:
            logger.warning(f"Skipping {instrument}: No data in time range!")
            continue

        # Apply any corrections on the data
        df = instrument_obj.data_corrections(
            df,
            start_altitude=ground_station["altitude"],
            start_pressure=ground_station["pressure"],
            start_temperature=ground_station["temperature"],
        )

        # Create housekeeping pressure variable to help align pressure visually
        df = instrument_obj.set_housekeeping_pressure_offset_variable(
            df, column_name=constants.HOUSEKEEPING_VAR_PRESSURE
        )

        # Save dataframe to outputs folder
        df.to_csv(
            f"{os.path.join(output_path_instrument_subfolder, instrument)}.csv"
        )

        # Prepare df for combining with other instruments
        df = instrument_obj.add_device_name_to_columns(df)

        # Add tuple of df and export order to df merge list
        all_export_dfs.append((df, instrument_obj))

    if len(all_export_dfs) == 0:
        logger.error("No instruments were processed, exiting")
        return

    preprocess.export_yaml_config(
        config, os.path.join(output_path_with_time, constants.CONFIG_FILE)
    )

    # Sort the export columns in their numerical hierarchy order and log
    all_export_dfs.sort(key=sorting.df_column_sort_key)

    master_export_cols = []
    master_housekeeping_cols = []

    # Merge all the dataframes together, first df is the master
    master_df, instrument = all_export_dfs[0]

    # Combine export and housekeeping columns
    master_export_cols += instrument.export_columns
    master_housekeeping_cols += instrument.housekeeping_columns

    # Merge the rest
    logger.info("Instruments will be merged together with this column order:")
    for df, instrument in all_export_dfs[1:]:
        logger.info(
            f"Merging instrument: {instrument.name:20} "
            f"(Export order value: {instrument.export_order})"
        )
        master_df = master_df.merge(
            df, how="outer", left_index=True, right_index=True
        )
        master_export_cols += instrument.export_columns
        master_housekeeping_cols += instrument.housekeeping_columns

    all_instruments = [instrument for df, instrument in all_export_dfs]

    # Sort rows by the date index
    master_df.index = pd.to_datetime(master_df.index)
    master_df.sort_index(inplace=True)

    # Export data and housekeeping CSV files
    master_df[master_export_cols].to_csv(
        os.path.join(output_path_with_time, constants.MASTER_CSV_FILENAME)
    )

    master_df[master_housekeeping_cols].to_csv(
        os.path.join(
            output_path_with_time, constants.HOUSEKEEPING_CSV_FILENAME
        )
    )

    # Create all of the plots
    plots.campaign_2023(
        master_df, plot_props, all_instruments, output_path_with_time
    )


def version_cb(value: bool) -> None:
    f"""Prints the version number of {constants.APPLICATION_NAME}

    Parameters
    ----------
    value : bool
        The value of the --version option
    """

    if value:  # Only run on when --version is set
        typer.echo(f"{constants.APPLICATION_NAME} {constants.VERSION}")
        sys.exit()


@app.callback(
    help=f"""
    {constants.APPLICATION_NAME} - {constants.DESCRIPTION}

    Use the --help option on a subcommand to see more information about it.
    """
)
def menu(
    version: bool = typer.Option(
        False,
        "--version",
        help=f"Prints the version number of {constants.APPLICATION_NAME}",
        callback=version_cb,
        is_eager=True,
    ),
):
    pass


if __name__ == "__main__":
    app()  # Execute the Typer CLI application

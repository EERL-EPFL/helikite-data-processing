import inspect
import logging
import os
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Any

import pandas as pd
from pandas import DataFrame
from plotly.graph_objects import Figure

from helikite.constants import constants
from helikite.processing.helpers import temporary_attr

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


class Instrument(ABC):
    REGISTRY = {}

    def __init__(
        self,
        name: str,  # Used as a prefix for instrument columns in processed data
        dtype: dict[Any, Any] = {},  # Mapping of column to data type
        na_values: list[Any] | None = None,  # list of values to consider null
        header: int | None = 0,  # Row ID for the header
        expected_header_value: str | None = None,  # Expected value of the header row
        delimiter: str = ",",  # String delimiter
        lineterminator: str | None = None,  # The character to define EOL
        comment: str | None = None,  # Ignore anything after set char
        names: list[str] | None = None,  # Names of headers if nonexistant
        index_col: bool | int | None = None,  # The column ID of the index
        cols_export: list[str] = [],  # Columns to export
        cols_housekeeping: list[str] = [],  # Columns to use for housekeeping
        cols_final: list[str] | None = None, # Columns to keep in the final data file
        export_order: int | None = None,  # Order hierarchy in export file
        pressure_variable: str | None = None,  # The variable measuring pressure
        registry_name: str | None = None,
        # Groups of columns that are coupled. If a row contains an outlier in any column within a tuple,
        # then the values in all other columns of that tuple should also be treated as outliers.
        coupled_columns: list[tuple[str, ...]] | None = None,
        rename_dict: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.na_values = na_values
        self.header = header
        self.expected_header_value = expected_header_value
        self.delimiter = delimiter
        self.lineterminator = lineterminator
        self.comment = comment
        self.names = names
        self.index_col = index_col
        self.cols_export = cols_export
        self.cols_housekeeping = cols_housekeeping
        self.cols_final = cols_final
        self.export_order = export_order
        self.pressure_variable = pressure_variable
        self.coupled_columns = coupled_columns if coupled_columns is not None else []
        self._rename_dict = rename_dict if rename_dict is not None else {}

        # Properties that are not part of standard config, can be added
        self.filename: str | None = None
        self.date: datetime | None = None
        self.pressure_offset_housekeeping: float | None = None
        self.time_offset: dict[str, int] = {}
        self.time_range: tuple[Any, Any] | None = None

        # Register every new instrument instance in the registry
        self._instantiation_info = self._get_instantiation_info()
        self.registry_name = registry_name if registry_name else self.name
        if self.registry_name in self.REGISTRY:
            registered = self.REGISTRY[self.registry_name]
            # allow reregistering an instrument when working in .ipynb with autoreload
            if self._instantiation_info != registered._instantiation_info:
                raise ValueError(f"{self.registry_name} is already registered by {self.REGISTRY[self.registry_name].__class__}.\n"
                                 f"Please use a different name for this instrument.\n")
            else:
                print("Reregistering instrument")
        self.REGISTRY[self.registry_name] = self

    @abstractmethod
    def __repr__(self):
        pass

    def add_config(self, yaml_props: dict[str, Any]):
        """Adds the application's config to the Instrument class

        This is called from the main() function in helikite.py
        """

        self.filename = yaml_props["file"]
        self.date = yaml_props["date"]
        self.time_offset = yaml_props["time_offset"]
        self.pressure_offset_housekeeping = yaml_props["pressure_offset"]

    @abstractmethod
    def data_corrections(self, df, *args, **kwargs) -> pd.DataFrame:
        """Default callback function for data corrections.

        Return with no changes
        """

        return df

    def create_plots(self, df: DataFrame) -> list[Figure | None]:
        """Default callback for generated figures from dataframes

        Return nothing, as anything else will populate the list that is written
        out to HTML.
        """

        return []

    def header_lines(self, file_path: str | pathlib.Path) -> list[str]:
        lines_to_read = self.header + 1

        with open(file_path, encoding="utf-8", errors="replace") as in_file:
            try:
                header_lines = [
                    next(in_file) for _ in range(lines_to_read)
                ]
                return header_lines

            except StopIteration:
                raise StopIteration(
                    f"Instrument has less than {lines_to_read} lines "
                    f"in {os.path.basename(file_path)}. "
                )

    def file_identifier(self, first_lines_of_csv: list[str]) -> bool:
        """Default file identifier callback
        """

        if first_lines_of_csv[self.header] == self.expected_header_value:
            return True

        header_partial_set = set(map(lambda s: s.strip(), self.expected_header_value.split(",")))
        first_line_set = set(map(lambda s: s.strip(), first_lines_of_csv[self.header].split(",")))

        return header_partial_set.issubset(first_line_set)

    def date_extractor(self, first_lines_of_csv: list[str]):
        """Returns the date of the data sample from a CSV header

        Can be used for an instrument that reports the date in header
        instead of in the data field.

        Return None if there is nothing to do here
        """

        return None

    def add_device_name_to_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Updates dataframe with column names prefixed by instrument name"""

        df.columns = f"{self.name}_" + df.columns.values

        return df

    @property
    def housekeeping_columns(self) -> list[str]:
        """Returns housekeeping columns, prefixed with the instrument name

        If there are no housekeeping variables, return an empty list
        """

        if self.cols_housekeeping:
            return [f"{self.name}_{x}" for x in self.cols_housekeeping]
        else:
            return []

    @property
    def export_columns(self) -> list[str]:
        """Returns export datafile columns, prefixed with the instrument name

        If there are no variables, return an empty list

        The export data file is a combined output, similar to housekeeping
        """

        if self.cols_export:
            return [f"{self.name}_{x}" for x in self.cols_export]
        else:
            return []

    @property
    def final_columns(self) -> list[str] | None:
        """
        Returns final datafile columns, prefixed with the instrument name, if they are specified.
        Otherwise, returns None.
        """
        return self.cols_final

    @property
    def rename_dict(self) -> dict[str, str]:
        """
        Returns a dictionary of column names to rename for the final data file.
        """
        return self._rename_dict

    @property
    def has_size_distribution(self) -> bool:
        return False

    @abstractmethod
    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """This function is used to set the date in the file to the index

        By default no action is taken, but this can be overridden in the
        instrument class to set the date as the index.
        """

        return df

    def correct_time_from_config(
        self,
        df: pd.DataFrame,
        trim_start: pd.Timestamp | None = None,
        trim_end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Correct the time offset and trim from the configuration"""

        if (
            not self.time_offset
            or self.time_offset == {}
            or (
                self.time_offset["hour"] == 0
                and self.time_offset["minute"] == 0
                and self.time_offset["second"] == 0
            )
        ):
            logger.info(f"No time offset for {self.name}")

            return df
        if (
            self.time_offset["hour"] != 0
            or self.time_offset["minute"] != 0
            or self.time_offset["second"] != 0
        ):
            logger.info(f"Shifting the time offset by {self.time_offset}")

            df.index = df.index + pd.DateOffset(
                hours=self.time_offset["hour"],
                minutes=self.time_offset["minute"],
                seconds=self.time_offset["second"],
            )

        # Trim the dataframes to the time range specified in the config
        logger.debug(f"{self.name}: Original start time: {df.iloc[0].name} ")
        logger.debug(f"{self.name}: Original end time: {df.iloc[-1].name} ")
        self.time_range = (df.iloc[0].name, df.iloc[-1].name)

        if None not in (trim_start, trim_end):
            datemask = (df.index > trim_start) & (df.index <= trim_end)
            df = df[datemask]

        return df

    def set_housekeeping_pressure_offset_variable(
        self, df: pd.DataFrame, column_name=constants.HOUSEKEEPING_VAR_PRESSURE
    ) -> pd.DataFrame:
        """Generate variable to offset pressure value for housekeeping

        Using an offset in the configuration, a new variable is created
        that offset's the instruments pressure variable. This is used to align
        the pressure value on the plot to help align pressure.
        """

        if self.pressure_variable is not None:
            if self.pressure_offset_housekeeping is None:
                # If no offset, but a pressure var exists add col of same val
                df[column_name] = df[self.pressure_variable]
            else:
                df[column_name] = (
                    df[self.pressure_variable]
                    + self.pressure_offset_housekeeping
                )

        return df

    @abstractmethod
    def read_data(self) -> pd.DataFrame:
        """Read data into dataframe

        This allows a custom read function to parse the CSV/TXT into a
        dataframe, for example cleaning dirty data at the end of the file
        in memory without altering the input file (see flight computer conf).

        """

        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            na_values=self.na_values,
            skiprows=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df

    def detect_from_folder(self, input_folder: str, quiet: bool = False, interactive: bool = False) -> list[str]:
        """Scans an input folder for the instrument file

        If there are two files that match the instrument, the function will
        raise an error. This is to prevent the program from trying to process
        the same file twice.
        """
        successful_matches = []

        for filename in os.listdir(input_folder):
            # Ignore any yaml or keep files
            if not filename.lower().endswith((".csv", ".txt", ".dat")):
                continue

            full_path = os.path.join(input_folder, filename)
            try:
                header_lines = self.header_lines(full_path)
            except StopIteration as e:
                logger.warning(e)
                continue
            except UnicodeDecodeError as e:
                logger.warning(f"Could not read {filename}: {e}")
                raise e

            if self.file_identifier(header_lines):
                if not quiet:
                    logger.info(
                        f"Instrument: {self.registry_name} detected in {filename}"
                    )
                successful_matches.append(filename)
                if len(successful_matches) > 1:
                    if interactive:
                        message = (f"Instrument detect in multiple files: {successful_matches}.\n"
                                   f"Please enter an index of the instrument to use: ")
                        index = int(input(message))
                        successful_matches = [successful_matches[index]]

        if not successful_matches:
            logger.warning(f"No instrument detected for {self.registry_name}")
            return []

        return [os.path.join(input_folder, match) for match in successful_matches]

    def read_from_folder(
        self,
        input_folder: str,
        quiet: bool = False,
        interactive: bool = True,
    ) -> pd.DataFrame:
        """Reads the data from the detected file in the input folder"""

        matched_files = self.detect_from_folder(input_folder, quiet, interactive)
        if not matched_files:
            return None

        self.filename = matched_files[0]
        df = self.read_data()

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the dataframe"""
        df_unique = df.copy()
        df_unique.drop_duplicates(inplace=True)

        logger.info(
            f"Duplicates removed from {self.name}: {len(df) - len(df_unique)}"
        )

        return df_unique

    def _get_instantiation_info(self) -> tuple[str, str | None] | None:
        """Returns the filename and the line of the instrument instantiation"""
        for frame in inspect.stack()[:4]:
            if frame.code_context is not None and self.__class__.__name__ in frame.code_context[0]:
                return frame.filename, frame.code_context[0]
        return None

    def calculate_derived(self, df: pd.DataFrame, verbose: bool, *args, **kwargs) -> pd.DataFrame:
        """Calculate derived data from the raw data, i.e. total concentration"""
        if verbose:
            logger.warning(f"Derived data calculation not implemented for {self.registry_name}.")

        return df

    def normalize(self, df: pd.DataFrame, verbose: bool, *args, **kwargs) -> pd.DataFrame:
        """Apply normalization to the data, i.e., normalization to STP conditions"""
        if verbose:
            logger.warning(f"Data normalization not implemented for {self.registry_name}.")

        return df

    def plot_raw_and_normalized(self, df: pd.DataFrame, verbose: bool, *args, **kwargs):
        """Plot together raw and normalized data. If only raw data is available, plot raw data."""
        if verbose:
            logger.warning(f"Plotting of raw and normalized data not implemented for {self.registry_name}.")

        return

    def plot_distribution(self, df: pd.DataFrame, verbose: bool,
                          time_start: datetime | None, time_end: datetime | None, *args, **kwargs):
        """Plot distribution of data"""
        if verbose:
            logger.warning(f"Plotting of distribution not implemented for {self.registry_name}.")

        return

    def plot_vertical_distribution(self, df: pd.DataFrame, verbose: bool, *args, **kwargs):
        """Plot vertical distribution of data"""
        if verbose:
            logger.warning(f"Plotting of vertical distribution not implemented for {self.registry_name}.")

        return

    def column_name(self, df: pd.DataFrame, before_rename: str) -> str:
        return before_rename if before_rename in df.columns else self.rename_dict[before_rename]

def filter_columns_by_instrument(columns: list[str], instrument: Instrument) -> list[str]:
    instrument_prefix = f"{instrument.name}_"
    instruments_same_prefix = [instr.name for instr in Instrument.REGISTRY.values()
                               if instr.name.startswith(instrument_prefix)]
    return [
        col for col in columns
        if col.startswith(instrument_prefix) and not any(col.startswith(instr) for instr in instruments_same_prefix)
    ]

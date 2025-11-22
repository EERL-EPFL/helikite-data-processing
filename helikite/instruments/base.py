import inspect
import logging
import os
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd
from pandas import DataFrame
from plotly.graph_objects import Figure

from helikite.constants import constants

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
        export_order: int | None = None,  # Order hierarchy in export file
        pressure_variable: str | None = None,  # The variable measuring pressure
        registry_name: str | None = None,
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
        self.export_order = export_order
        self.pressure_variable = pressure_variable

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

        with open(file_path) as in_file:
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

        return first_lines_of_csv[self.header] == self.expected_header_value

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

    def detect_from_folder(self, input_folder: str, quiet=False) -> str:
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

            if self.file_identifier(header_lines):
                if not quiet:
                    logger.info(
                        f"Instrument: {self.registry_name} detected in {filename}"
                    )
                successful_matches.append(filename)
                if len(successful_matches) > 1:
                    raise ValueError(
                        "Multiple instruments detected: "
                        f"{successful_matches}. "
                        "Please ensure only one instrument is detected."
                    )

        if not successful_matches:
            logger.warning(f"No instrument detected for {self.name}")
            return None

        if not quiet:
            logger.info(f"Matched file: {successful_matches[0]}")

        return os.path.join(input_folder, successful_matches[0])

    def read_from_folder(
        self,
        input_folder: str,
        quiet=False,
    ) -> pd.DataFrame:
        """Reads the data from the detected file in the input folder"""

        self.filename = self.detect_from_folder(input_folder, quiet=quiet)
        if self.filename is None:
            return None

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

    def get_expected_columns(self, level: float | None, is_reference: bool) -> list[str]:
        match level:
            case None:
                return list(self.dtype.keys())

            case 0:
                df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in self.dtype.items()})
                df = self.set_time_as_index(df)

                # TODO: Remove this once addition of `scan_direction` is integrated in the cleaning pipeline
                if self.name == "msems_inverted":
                    df.insert(len(df.columns), "scan_direction", pd.Series([], dtype="Int64"))

                df = self.data_corrections(df)

                if self.pressure_variable is not None:
                    df = self.set_housekeeping_pressure_offset_variable(df, constants.HOUSEKEEPING_VAR_PRESSURE)

                if not is_reference and self.pressure_variable is not None:
                    df.insert(0, "DateTime", df.index)

                return [f"{self.name}_{column}" for column in df.columns]

            case 1 | 1.5 | 2:
                raise ValueError(f"Unsupported level: {level}")

            case _:
                raise ValueError(f"Unexpected level: {level}")


    def _get_instantiation_info(self) -> tuple[str, str | None] | None:
        """Returns the filename and the line of the instrument instantiation"""
        for frame in inspect.stack()[:4]:
            if frame.code_context is not None and self.__class__.__name__ in frame.code_context[0]:
                return frame.filename, frame.code_context[0]
        return None

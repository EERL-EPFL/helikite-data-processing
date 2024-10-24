from typing import Dict, Any, List, Tuple
from pandas import DataFrame
from plotly.graph_objects import Figure
from datetime import datetime
import pandas as pd
import logging
from helikite.constants import constants
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


class Instrument(ABC):
    def __init__(
        self,
        dtype: Dict[Any, Any] = {},  # Mapping of column to data type
        na_values: List[Any] | None = None,  # List of values to consider null
        header: int | None = 0,  # Row ID for the header
        delimiter: str = ",",  # String delimiter
        lineterminator: str | None = None,  # The character to define EOL
        comment: str | None = None,  # Ignore anything after set char
        names: List[str] | None = None,  # Names of headers if nonexistant
        index_col: bool | int | None = None,  # The column ID of the index
        cols_export: List[str] = [],  # Columns to export
        cols_housekeeping: List[str] = [],  # Columns to use for housekeeping
        export_order: int | None = None,  # Order hierarchy in export file
        pressure_variable: (
            str | None
        ) = None,  # The variable measuring pressure
    ) -> None:

        self.dtype = dtype
        self.na_values = na_values
        self.header = header
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
        self.time_offset: Dict[str, int] = {}
        self.name: str | None = None
        self.time_range: Tuple[Any, Any] | None = None

    def add_config(self, yaml_props: Dict[str, Any]):
        """Adds the application's config to the Instrument class

        This is called from the main() function in helikite.py
        """

        self.filename = yaml_props["file"]
        self.date = yaml_props["date"]
        self.time_offset = yaml_props["time_offset"]
        self.pressure_offset_housekeeping = yaml_props["pressure_offset"]

    @abstractmethod
    def data_corrections(self, df, *args, **kwargs):
        """Default callback function for data corrections.

        Return with no changes
        """

        return df

    def create_plots(self, df: DataFrame) -> List[Figure | None]:
        """Default callback for generated figures from dataframes

        Return nothing, as anything else will populate the list that is written
        out to HTML.
        """

        return []

    @abstractmethod
    def file_identifier(self, first_lines_of_csv: List[str]):
        """Default file identifier callback

        Must return false. True would provide false positives.
        """

        return False

    def date_extractor(self, first_lines_of_csv: List[str]):
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
    def housekeeping_columns(self) -> List[str]:
        """Returns housekeeping columns, prefixed with the instrument name

        If there are no housekeeping variables, return an empty list
        """

        if self.cols_housekeeping:
            return [f"{self.name}_{x}" for x in self.cols_housekeeping]
        else:
            return []

    @property
    def export_columns(self) -> List[str]:
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
            header=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df

    def detect_from_folder(
        self,
        input_folder: str,
        lines_to_read=50,
        quiet=False,
    ):
        """Scans an input folder for the instrument file

        If there are two files that match the instrument, the function will
        raise an error. This is to prevent the program from trying to process
        the same file twice.
        """
        successful_matches = []

        for filename in os.listdir(input_folder):
            # Ignore any yaml or keep files
            if not filename.lower().endswith((".csv", ".txt")):
                continue

            full_path = os.path.join(input_folder, filename)

            with open(full_path) as in_file:
                try:
                    header_lines = [
                        next(in_file) for _ in range(lines_to_read)
                    ]
                except (IndexError, StopIteration):
                    logger.warning(
                        f"Instrument has less than {lines_to_read} lines "
                        f"in {filename}. "
                        f"Stopping early at line {len(header_lines)}."
                    )

                if self.file_identifier(header_lines):
                    if not quiet:
                        logger.info(
                            f"Instrument: {self.name} detected in {filename}"
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
        df_unique["duplicate_values"] = df_unique.index.duplicated()
        df_unique.drop_duplicates(inplace=True)

        logger.info(
            f"Duplicates removed from {self.name}: {len(df) - len(df_unique)}"
        )

        return df_unique

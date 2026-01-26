"""
Filter instrument class for Helikite project.
"""
import pathlib
import re

from helikite.instruments.base import Instrument
import pandas as pd
import logging
from helikite.constants import constants


# Define logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


class Filter(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "Filter"

    @property
    def _expected_header_value_FC(self) -> str:
        columns = self.expected_header_value.split()
        columns_FC = [f"F_{col}" for col in columns]

        return ",".join(columns_FC)

    def header_lines(self, file_path: str | pathlib.Path) -> list[str]:
        with open(file_path, encoding="utf-8", errors="replace") as in_file:
            # if filter values are taken from the FC file, the header line is the 0th line
            # otherwise, it is the 13th line
            line = next(in_file)
            if self._expected_header_value_FC in line:
                header_lines = [line]
            else:
                header_lines = super().header_lines(file_path)
        header_lines = [re.sub(r"\s+", " ", line) for line in header_lines]

        return header_lines

    def file_identifier(self, first_lines_of_csv) -> bool:
        if self._expected_header_value_FC in first_lines_of_csv[0]:
            return True

        if self.expected_header_value in first_lines_of_csv[self.header]:
            return True

        return False

    def data_corrections(self, df, *args, **kwargs):
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        return df

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe

        Filter instrument contains date and time separately and appears to
        include an extra whitespace in the field of each of those two columns
        """
        # If filter values are taken from the FC file, take date and time from the "Time" column
        if "Time" in df.columns:
            df["DateTime"] = pd.to_datetime(df["Time"], format="%y%m%d-%H%M%S")
            df.drop(columns=["Time"], inplace=True)
        else:
            # Combine both date and time columns into one, strip extra whitespace
            df["DateTime"] = pd.to_datetime(
                df["#YY/MM/DD"].str.strip() + " " + df["HR:MN:SC"].str.strip(),
                format="%y/%m/%d %H:%M:%S",
            )
            df.drop(columns=["#YY/MM/DD", "HR:MN:SC"], inplace=True)

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)
        df.index = df.index.astype("datetime64[s]")

        return df

    def read_data(self) -> pd.DataFrame:
        if self._expected_header_value_FC in self.header_lines(self.filename)[0]:
            columns_FC = ["Time"] + self._expected_header_value_FC.split(",")
            df = pd.read_csv(
                self.filename,
                on_bad_lines="warn",
                low_memory=False,
                sep=",",
                usecols=columns_FC,
            )
            df = df.dropna(subset=["Time"])
            df = df.rename(columns={col: col.removeprefix("F_") for col in df.columns})
            for col in self.dtype.keys():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(self.dtype[col])
        else:
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


filter = Filter(
    name="filter",
    header=13,
    delimiter="\t",
    dtype={
        "#YY/MM/DD": "str",
        "HR:MN:SC": "str",
        "cur_pos": "Int64",
        "cntdown": "Int64",
        "smp_flw": "Float64",
        "smp_tmp": "Float64",
        "smp_prs": "Int64",
        "pump_pw": "Int64",
        "psvolts": "Float64",
        "err_rpt": "Int64",
        "pumpctl": "Int64",
        "ctlmode": "Int64",
        "intervl": "Int64",
        "flow_sp": "Float64",
    },
    expected_header_value="cur_pos cntdown smp_flw smp_tmp smp_prs pump_pw psvolts err_rpt",
    cols_export=["cur_pos", "smp_flw", "pumpctl"],
    cols_housekeeping=[
        "cur_pos",
        "cntdown",
        "smp_flw",
        "smp_tmp",
        "smp_prs",
        "pump_pw",
        "psvolts",
        "err_rpt",
        "pumpctl",
        "ctlmode",
        "intervl",
        "flow_sp",
    ],
    cols_final=["cur_pos", "smp_flw"],
    export_order=550,
    pressure_variable="smp_prs",
    rename_dict={
        'filter_cur_pos': 'Filter_position',
        'filter_smp_flw': 'Filter_flow'
    },
)

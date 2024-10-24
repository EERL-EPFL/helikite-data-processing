"""
Filter instrument class for Helikite project.
"""

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
        self.name = "filter"

    def file_identifier(self, first_lines_of_csv) -> bool:
        if "cur_pos	cntdown	smp_flw	smp_tmp	smp_prs" in first_lines_of_csv[13]:
            return True

        return False

    def data_corrections(self, df, *args, **kwargs):
        return df

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe

        Filter instrument contains date and time separately and appears to
        include an extra whitespace in the field of each of those two columns
        """

        # Combine both date and time columns into one, strip extra whitespace
        df["DateTime"] = pd.to_datetime(
            df["#YY/MM/DD"].str.strip() + " " + df["HR:MN:SC"].str.strip(),
            format="%y/%m/%d %H:%M:%S",
        )
        df.drop(columns=["#YY/MM/DD", "HR:MN:SC"], inplace=True)

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)

        return df

    def read_data(self) -> pd.DataFrame:

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


filter = Filter(
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
    export_order=550,
    pressure_variable="smp_prs",
)

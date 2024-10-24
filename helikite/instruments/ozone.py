"""
8) Ozone monitor -> LOG46.txt (can't use pressure)

Resolution: 2 sec

Headers: "ozone","cell_temp","cell_pressure","flow_rate","date","time"

(last column is nothing)
"""

from helikite.instruments.base import Instrument
import pandas as pd


class OzoneMonitor(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "ozone"

    def file_identifier(self, first_lines_of_csv) -> bool:
        # This one is tricky. There is no header! May run into conflicts later
        # Check there are six commas in the first line, and ends in 0, and only
        # a newline in the second
        if (
            first_lines_of_csv[0][-2] == "0"
            and first_lines_of_csv[0].count(",") == 6
            and first_lines_of_csv[1] == "\n"
        ):
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct"""

        df["DateTime"] = pd.to_datetime(
            df["date"] + " " + df["time"], format="%d/%m/%y %H:%M:%S"
        )
        df.drop(columns=["date", "time"], inplace=True)

        # Define the datetime column as the index
        df.reset_index(drop=False, inplace=True)
        df.set_index("DateTime", inplace=True)

        return df

    def data_corrections(self, df, *args, **kwargs):
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


ozone_monitor = OzoneMonitor(
    na_values=["-"],
    index_col=0,
    header=None,
    names=[
        "ozone",
        "cell_temp",
        "cell_pressure",
        "flow_rate",
        "date",
        "time",
        "unused",
    ],
    dtype={
        "ozone": "Float64",
        "cell_temp": "Float64",
        "cell_pressure": "Float64",
        "flow_rate": "Int64",
        "date": "str",
        "time": "str",
    },
    cols_export=["ozone"],
    cols_housekeeping=["ozone", "cell_temp", "cell_pressure", "flow_rate"],
    export_order=250,
    pressure_variable="cell_pressure",
)

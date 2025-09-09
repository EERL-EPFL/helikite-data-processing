"""
TAPIR -> tapir20250212_50 (DOESN`T have pressure)

Important variables to keep:

!!! GPS Time.

"""

from .base import Instrument
import pandas as pd
import datetime
import logging
from helikite.constants import constants

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


class TAPIR(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "tapir"

    def file_identifier(self, first_lines_of_csv) -> bool:
        if first_lines_of_csv[0] == (
            "ST,YrMoDy,HrMnSd;GT,YrMoDy,HrMnSd;GL,Lat,Le,Lon,Lm,speed,route;TP,Tproc1,Tproc2,Tproc3,Tproc4;TH,Thead1,Thead2,Thead3,Thead4;TB,Tbox\n"
        ):
            return True
    
        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Combine GT date and time into a single datetime
            df["DateTime"] = pd.to_datetime(df["GT_YrMoDy"].astype(str) + df["GT_HrMnSd"].astype(str), format="%Y%m%d%H%M%S")
            df.set_index("DateTime", inplace=True)
            df.index = df.index.astype("datetime64[s]")
            df.drop(columns=["GT_YrMoDy", "GT_HrMnSd"], inplace=True)
        except Exception as e:
            logger.error(f"Failed to convert date and time to datetime index: {e}")
            raise
        return df

    def data_corrections(self, df, *args, **kwargs):
        return df

    def read_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.filename,
                dtype=self.dtype,
                header=self.header,
                na_values=self.na_values,
                delimiter=self.delimiter,
                lineterminator=self.lineterminator,
                comment=self.comment,
                names=self.names,
                index_col=self.index_col,
            )
            return df
        except Exception as e:
            logger.error(f"Failed to read TAPIR data from {self.filename}: {e}")
            raise


tapir = TAPIR(
    dtype={
        "GT_YrMoDy": "str",
        "GT_HrMnSd": "str",
        "Lat": "Float64",
        "Le": "Float64",
        "Lon": "Float64",
        "Lm": "Float64",
        "speed": "Float64",
        "route": "Float64",
        "Tproc1": "Float64",
        "Tproc2": "Float64",
        "Tproc3": "Float64",
        "Tproc4": "Float64",
        "Thead1": "Float64",
        "Thead2": "Float64",
        "Thead3": "Float64",
        "Thead4": "Float64",
        "Tbox": "Float64",
    },
    header=0,  # Adjust if header starts lower
    export_order=620,
    cols_export=[
        "Lat", "Lon", "Le", "speed", "route", "Tproc1", "Tproc2", "Tproc3", "Tproc4", "Thead1", "Thead2", "Thead3", "Thead4", "Tbox"
    ],
    cols_housekeeping=[
        "GT_YrMoDy", "GT_HrMnSd", "Lat", "Lon", "Le", "speed", "route", "Tproc1", "Tproc2", "Tproc3", "Tproc4", "Thead1", "Thead2", "Thead3", "Thead4", "Tbox"
    ],
    pressure_variable=None  # Add if TAPIR has pressure data
)

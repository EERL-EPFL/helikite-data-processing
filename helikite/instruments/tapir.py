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

    def __repr__(self):
        return "TAPIR"

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Combine GT date and time into a single datetime
            df["DateTime"] = pd.to_datetime(df["YrMoDy"].astype(str) + df["HrMnSd"].astype(str), format="%Y%m%d%H%M%S")
            df.set_index("DateTime", inplace=True)
            df.index = df.index.astype("datetime64[s]")
            # df.drop(columns=["YrMoDy", "HrMnSd"], inplace=True)
        except Exception as e:
            logger.error(f"Failed to convert date and time to datetime index: {e}")
            raise
        return df

    def data_corrections(self, df, *args, **kwargs):
        df = df.resample("1s").asfreq()

        return df

    def read_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.filename,
                dtype=self.dtype,
                engine="python",
                skiprows=self.header,
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
    name="tapir",
    dtype={
        "ST": "str",
        "YrMoDy": "str",
        "HrMnSd": "str",
        "GT": "str",
        "YrMoDy.1": "str",
        "HrMnSd.1": "str",
        "GL": "str",
        "Lat": "Float64",
        "Le": "str",
        "Lon": "Float64",
        "Lm": "str",
        "speed": "Float64",
        "route": "Float64",
        "TP": "str",
        "Tproc1": "Float64",
        "Tproc2": "Float64",
        "Tproc3": "Float64",
        "Tproc4": "Float64",
        "TH": "str",
        "Thead1": "Float64",
        "Thead2": "Float64",
        "Thead3": "Float64",
        "Thead4": "Float64",
        "TB": "str",
        "Tbox": "Float64",
    },
    expected_header_value=(
        "ST,YrMoDy,HrMnSd;GT,YrMoDy,HrMnSd;GL,Lat,Le,Lon,Lm,speed,route;TP,Tproc1,Tproc2,Tproc3,Tproc4;"
        "TH,Thead1,Thead2,Thead3,Thead4;TB,Tbox\n"
    ),
    delimiter=r"[;,]",
    header=0,  # Adjust if header starts lower
    export_order=620,
    cols_export=[
        "Lat", "Lon", "Le", "speed", "route", "Tproc1", "Tproc2", "Tproc3", "Tproc4", "Thead1", "Thead2", "Thead3", "Thead4", "Tbox"
    ],
    cols_housekeeping=[
        "YrMoDy", "HrMnSd", "Lat", "Lon", "Le", "speed", "route", "Tproc1", "Tproc2", "Tproc3", "Tproc4", "Thead1", "Thead2", "Thead3", "Thead4", "Tbox"
    ],
    cols_final=[
        "Lat", "Lon", "speed", "route", "Tproc1", "Tproc2", "Tproc3", "Tproc4",
        "Thead1", "Thead2", "Thead3", "Thead4", "Tbox"
    ],
    pressure_variable=None  # Add if TAPIR has pressure data
)

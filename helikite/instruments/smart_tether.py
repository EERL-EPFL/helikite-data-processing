"""

2) SmartTether -> LOG_20220929_A.csv (has pressure)

The SmartTether is a weather sonde. time res 2 seconds if lon lat recorded.
1 sec if not.

Important variables to keep:
Time, Comment, P (mbar), T (deg C), RH (%), Wind (degrees), Wind (m/s),
UTC Time, Latitude (deg), Longitude (deg)

!!! Date is not reported in the data, but only in the header (yes, it's a pity)
-> therefore, I wrote a function that to includes the date but it needs to
change date if we pass midnight (not implemented yet).

"""

from .base import Instrument
import datetime
import pandas as pd
import logging
from helikite.constants import constants

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


class SmartTether(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "smart_tether"

    def date_extractor(self, first_lines_of_csv) -> datetime.datetime:
        date_line = first_lines_of_csv[1]
        date_string = date_line.split(" ")[-1].strip()

        return datetime.datetime.strptime(date_string, "%m/%d/%Y")

    def file_identifier(self, first_lines_of_csv) -> bool:
        if first_lines_of_csv[
            0
        ] == "SmartTether log file\n" and first_lines_of_csv[3] == (
            "Time,Comment,Module ID,Alt (m),P (mbar),T (deg C),%RH,Wind "
            "(degrees),Wind (m/s),Supply (V),UTC Time,Latitude (deg),"
            "Longitude (deg),Course (deg),Speed (m/s)\n"
        ):
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        Using values in the time_offset variable, correct DateTime index

        As the rows store only a time variable, a rollover at midnight is
        possible. This function checks for this and corrects the date if needed
        """

        if self.date is None:
            raise ValueError(
                "No flight date provided. Necessary for SmartTether"
            )

        date = self.date
        if isinstance(self.date, datetime.date):
            date = pd.to_datetime(self.date)

        # Date from header (stored in self.date), then add time
        df["DateTime"] = pd.to_datetime(date + pd.to_timedelta(df["Time"]))

        # Check for midnight rollover. Can assume that the data will never be
        # longer than a day, so just check once for a midnight rollover
        start_time = pd.Timestamp(df.iloc[0]["Time"])
        for i, row in df.iterrows():
            # check if the timestamp is earlier than the start time (i.e. it's
            # the next day)
            if pd.Timestamp(row["Time"]) < start_time:
                # add a day to the date column
                logger.info("SmartTether date passes midnight. Correcting...")
                logger.info(f"Adding a day at: {df.at[i, 'DateTime']}")
                df.at[i, "DateTime"] += pd.Timedelta(days=1)

        df.drop(columns=["Time"], inplace=True)

        # Define the datetime column as the index
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


smart_tether = SmartTether(
    dtype={
        "Time": "str",
        "Comment": "str",
        "Module ID": "str",
        "Alt (m)": "Int64",
        "P (mbar)": "Float64",
        "T (deg C)": "Float64",
        "%RH": "Float64",
        "Wind (degrees)": "Int64",
        "Wind (m/s)": "Float64",
        "Supply (V)": "Float64",
        "UTC Time": "str",
        "Latitude (deg)": "Float64",
        "Longitude (deg)": "Float64",
        "Course (deg)": "Float64",
        "Speed (m/s)": "Float64",
    },
    header=2,
    export_order=600,
    cols_export=[
        "Comment",
        "P (mbar)",
        "T (deg C)",
        "%RH",
        "Wind (degrees)",
        "Wind (m/s)",
        "UTC Time",
        "Latitude (deg)",
        "Longitude (deg)",
    ],
    cols_housekeeping=[
        "Comment",
        "Module ID",
        "Alt (m)",
        "P (mbar)",
        "T (deg C)",
        "%RH",
        "Wind (degrees)",
        "Wind (m/s)",
        "Supply (V)",
        "UTC Time",
        "Latitude (deg)",
        "Longitude (deg)",
        "Course (deg)",
        "Speed (m/s)",
    ],
    pressure_variable="P (mbar)",
)

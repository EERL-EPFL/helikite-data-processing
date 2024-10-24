"""

1) Flight computer -> LOG_20220929.txt (has pressure)

Data from the onboard microcontroller. Resolution: 1 sec
DateTime in seconds since 1970-01-01 (to be verified)

Variables to keep: DateTime, P_baro, CO2, TEMP1, TEMP2, TEMPsamp, RH1, RH2,
                   RHsamp, mFlow
Houskeeping variables: TEMPbox, vBat
"""

from helikite.instruments.base import Instrument
import pandas as pd
from helikite.processing.conversions import pressure_to_altitude
from io import StringIO
import logging
from helikite.constants import constants


# Define logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)

CSV_HEADER = (
    "SBI,DateTime,PartCon,CO2,P_baro,TEMPbox,mFlow,TEMPsamp,RHsamp,"
    "TEMP1,RH1,TEMP2,RH2,vBat\n"
)


class FlightComputer(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "flight_computer"

    def file_identifier(self, first_lines_of_csv) -> bool:
        if first_lines_of_csv[0] == CSV_HEADER:
            return True

        return False

    def data_corrections(
        self,
        df: pd.DataFrame,
        *,
        start_altitude: float | None = None,
        start_pressure: float | None = None,
        start_temperature: float | None = None,
        start_duration_seconds: int = 10,
    ) -> pd.DataFrame:

        # Create altitude column by using average of first 10 seconds of data
        if start_pressure is None or start_temperature is None:
            try:
                first_period = df.loc[
                    df.index[0] : df.index[0]  # noqa
                    + pd.Timedelta(seconds=start_duration_seconds)
                ]

                averaged_sample = first_period.mean(numeric_only=True)
            except IndexError as e:
                logger.error(
                    "There is not enough data in the flight computer to "
                    f"measure the first {start_duration_seconds} seconds for "
                    "pressure and temperature in order to calculate altitude. "
                    "Data only available for time "
                    f"range: {self.time_range[0]} to {self.time_range[1]}. "
                    "To bypass this, input values for ground temperature and "
                    "pressure in the config file."
                )
                raise e

        if start_pressure is None:
            pressure = round(averaged_sample[self.pressure_variable], 2)
            logger.info(
                f"No defined ground station pressure. Using estimate from "
                f"first {start_duration_seconds} seconds of data. Calculated "
                f"to: {pressure} (Flight Computer: {self.pressure_variable})"
            )
        else:
            pressure = start_pressure
            logger.info(f"Pressure at start defined in config as: {pressure}")

        if start_temperature is None:
            temperature = round(averaged_sample.TEMP1, 2)
            logger.info(
                f"No defined ground station temperature. Using estimate from "
                f"first {start_duration_seconds} seconds of data. Calculated "
                f"to: {temperature} (Flight Computer: TEMP1)"
            )
        else:
            temperature = start_temperature
            logger.info(
                f"Temperature at start defined in config as: {temperature}"
            )

        altitude = start_altitude if start_altitude else 0
        logger.info(f"Altitude at start set to: {altitude}")

        # Calculate altitude above mean sea level
        df["Altitude"] = df[self.pressure_variable].apply(
            pressure_to_altitude,
            pressure_at_start=pressure,
            temperature_at_start=temperature,
            altitude_at_start=altitude,
        )

        # Create a new column representing altitude above ground level
        # by subtracting starting altitude from calculated
        df["Altitude_agl"] = df["Altitude"] - altitude

        return df

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        Using values in the time_offset variable, correct DateTime index
        """

        # Flight computer uses seconds since 1970-01-01
        df["DateTime"] = pd.to_datetime(df["DateTime"], unit="s")

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)

        return df

    def read_data(self) -> pd.DataFrame:
        """Read data into dataframe"""

        # Parse the file first removing the duplicate header cols
        cleaned_csv = StringIO()
        header_counter = 0

        with open(self.filename, "r") as csv_data:
            for row in csv_data:
                if row == CSV_HEADER:
                    if header_counter == 0:
                        # Only append the first header, ignore all others
                        cleaned_csv.write(row)
                    header_counter += 1
                else:
                    cleaned_csv.write(row)

        # Seek back to start of memory object
        cleaned_csv.seek(0)

        df = pd.read_csv(
            cleaned_csv,  # Load the StringIO object created above
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


flight_computer = FlightComputer(
    dtype={
        "SBI": "str",
        "DateTime": "Int64",
        "PartCon": "Int64",
        "CO2": "Float64",
        "P_baro": "Float64",
        "TEMPbox": "Float64",
        "mFlow": "str",
        "TEMPsamp": "Float64",
        "RHsamp": "Float64",
        "TEMP1": "Float64",
        "RH1": "Float64",
        "TEMP2": "Float64",
        "RH2": "Float64",
        "vBat": "Float64",
    },
    na_values=["NA", "-9999.00"],
    comment="#",
    cols_export=[
        "Altitude",
        "Altitude_agl",
        "P_baro",
        "CO2",
        "TEMP1",
        "TEMP2",
        "TEMPsamp",
        "RH1",
        "RH2",
        "RHsamp",
        "mFlow",
    ],
    cols_housekeeping=[
        "Altitude",
        "Altitude_agl",
        "SBI",
        "PartCon",
        "CO2",
        "P_baro",
        "TEMPbox",
        "mFlow",
        "TEMPsamp",
        "RHsamp",
        "TEMP1",
        "RH1",
        "TEMP2",
        "RH2",
        "vBat",
    ],
    export_order=100,
    pressure_variable="P_baro",
)

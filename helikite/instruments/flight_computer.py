from helikite.instruments.base import Instrument
from helikite.processing.conversions import pressure_to_altitude
from io import StringIO
from helikite.constants import constants
import logging
import pandas as pd


# Define logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


class FlightComputerV1(Instrument):
    """
    This flight computer relates to the first version used in campaigns
    in 2023, 2024. A new version was designed in 2024. See FlightComputerV2.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "flight_computer"
        self._csv_header = (
            "SBI,DateTime,PartCon,CO2,P_baro,TEMPbox,mFlow,TEMPsamp,RHsamp,"
            "TEMP1,RH1,TEMP2,RH2,vBat\n"
        )

    def file_identifier(self, first_lines_of_csv) -> bool:
        if first_lines_of_csv[0] == self._csv_header:
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
                if row == self._csv_header:
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


class FlightComputerV2(Instrument):
    """
    This flight computer relates to the second version used in campaigns
    in 2024. This version uses a new set of metadata and a modified CSV format.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "flight_computer"

    def file_identifier(self, first_lines_of_csv) -> bool:
        # In V2, datetime is prefixed with a space, check partial is within
        # first line

        # Datetime is prefixed with space, so we need to account for that
        header_partial = (
            "_cur_pos,F_cntdown,F_smp_flw,F_smp_tmp,F_smp_prs,F_pump_pw,"
            "F_psvolts,F_err_rpt,SO_S,SO_D,SO_U,SO_V,SO_W,SO_T,SO_H,SO_P,"
            "SO_PI,SO_RO,SO_MD,POPID,POPCHAIN,POPtot,POPf,POPT,POPc1,POPc2,"
            "POPc3,POPc4,POPc5,POPc6,POPc7,POPc8,Ubat,CO2,BME_T,BME_H,BME_P,"
            "CPUTEMP,RPiT,RPiS,IYaw,IPitch,IRoll,ILat,ILong,IVX,IVY,IVZ,IAX,"
            "IAY,IAZ,IARX,IARY,IARZ,Out1_T,Out1_H,Out2_T,Out2_H,Inlet2_T,"
            "Inlet2_H,UTCTime,Status,Lat,LatDir,Long,LongDir,Speed,Course,"
            "Date,MagVar,MVdir,GLat,GLatDir,GLong,GLongDir,GPSQ,GNSats,"
            "Hprec,GAlt,AltU,Geoidal,UTCTime2,Heading,HeadTrue,Roll,Pitch,"
            "Heave,RollAcc,PitchAcc,HeadAcc,GNSSqty"
        )

        return header_partial in first_lines_of_csv[0]

    def read_data(self) -> pd.DataFrame:
        """Read data into dataframe, adjusting for duplicate headers."""

        """ Evan 11.05.25: Commented below is attempt to clean the CSV file,
        # disabled for now in favour of alternative read code
        cleaned_csv = StringIO()

        # The file needs parsing first:
        # 1. The header row starts with a datetime and space, rename it to
        #    datetime, remove the space and write to a new StringIO object
        # 2. The same applies to each row, replace space with comma, keep the
        #    data
        # 3. Some lines have been split over two lines, find any rows ending
        #    in a comma and append the next line to it
        with open(self.filename, "r") as csv_data:
            saved_row = None  # Store the last row if it's split
            for row_index, row in enumerate(csv_data):
                if row_index == 0:
                    # Split the space out of the whole header (removing the
                    # recorded timestamp), then add "datetime" and a comma
                    full_header = row.split(" ")[1]
                    fixed_header = f"DateTime,{full_header}"
                    cleaned_csv.write(fixed_header)
                else:
                    # Replace space with comma in the first 20 chars of line
                    fixed_row = row[:20].replace(" ", ",") + row[20:]

                    # If the column ends with a comma, it's a split row, so
                    # append the next row to it
                    if fixed_row[-2] in [",", "-"]:
                        saved_row = fixed_row
                        # If there is a - with no value, remove the sign
                        if fixed_row[-2] == "-":
                            saved_row = saved_row[:-2] + saved_row[-1]

                        continue

                    if fixed_row[0] == ",":  # Remove leading comma
                        # Add the saved row to the start of the current row
                        fixed_row = saved_row[:-1] + fixed_row
                        saved_row = None

                    # Get all individual columns by splitting on commas then
                    # remove any extra columns
                    number_of_columns = len(fixed_row.split(","))
                    fixed_row = fixed_row.split(",")

                    # If the number of columns is greater than expected, remove
                    # the extra columns
                    if number_of_columns > len(self.cols_housekeeping):
                        fixed_row = fixed_row[: len(self.cols_housekeeping)]

                    if number_of_columns < len(self.cols_housekeeping):
                        # If the number of columns is less than expected append
                        # empty columns
                        fixed_row += [
                            ""
                            for _ in range(
                                len(self.cols_housekeeping) - number_of_columns
                            )
                        ]

                    # Join the columns back into a string
                    fixed_row = ",".join(fixed_row)
                    cleaned_csv.write(fixed_row)

                cleaned_csv.write("\n")

        # Return to the start of StringIO for reading
        cleaned_csv.seek(0)
        print(cleaned_csv.getvalue())'
        """

        df = pd.read_csv(
            self.filename,
            on_bad_lines="skip",
            low_memory=False,
            sep=",",
        )

        # to convert all the columns to float, except the string columns
        exclude_str_cols = [
            "Time",
            "RPiS",
            "Status",
            "LongDir",
            "LatDir",
            "HeadTrue",
            "GLongDir",
            "HeadTrue",
        ]

        # columns from the INS are shifted and need to be corrected
        shifted_columns = {
            "IYaw": "issue1",
            "IPitch": "issue2",
            "IRoll": "issue3",
            "ILat": "IYaw",
            "ILong": "IPitch",
            "IVX": "IRoll",
            "IVY": "ILat",
            "IVZ": "ILong",
            "IAX": "IVX",
            "IAY": "IVY",
            "IAZ": "IVZ",
            "IARX": "IAX",
            "IARY": "IAY",
            "IARZ": "IAZ",
        }

        df.loc[:, ~df.columns.isin(exclude_str_cols)] = df.loc[
            :, ~df.columns.isin(exclude_str_cols)
        ].apply(pd.to_numeric, errors="coerce")

        # Rename columns that have been shifted
        df.rename(columns=shifted_columns, inplace=True)

        # define the timestamp
        df["DateTime"] = pd.to_datetime(df["Time"], format="%y%m%d-%H%M%S")
        df.index = df["DateTime"]

        # Remove any rows with missing time data
        df = df.dropna(subset=["DateTime"])

        # Remote rows with missing or empty pressure data
        df = df.dropna(subset=[self.pressure_variable])

        return df

    def data_corrections(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        The DateTime column is represented as YYMMMM-HHMMSS, so we need to
        convert this to a datetime object.
        Example: 240926-141516
        """

        # Flight computer uses seconds since 1970-01-01
        df["DateTime"] = pd.to_datetime(df.DateTime, format="%y%m%d-%H%M%S")

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)

        # Set to index type to seconds
        df.index = df.index.astype("datetime64[s]")

        return df


flight_computer_v1 = FlightComputerV1(
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

flight_computer_v2 = FlightComputerV2(
    dtype={
        "DateTime": "str",
        "Time": "str",
        "F_cur_pos": "Float64",
        "F_cntdown": "Float64",
        "F_smp_flw": "Float64",
        "F_smp_tmp": "Float64",
        "F_smp_prs": "Float64",
        "F_pump_pw": "Float64",
        "F_psvolts": "Float64",
        "F_err_rpt": "Float64",
        "SO_S": "Float64",
        "SO_D": "Float64",
        "SO_U": "Float64",
        "SO_V": "Float64",
        "SO_W": "Float64",
        "SO_T": "Float64",
        "SO_H": "Float64",
        "SO_P": "Float64",
        "SO_PI": "Float64",
        "SO_RO": "Float64",
        "SO_MD": "Float64",
        "POPID": "Float64",
        "POPCHAIN": "Float64",
        "POPtot": "Float64",
        "POPf": "Float64",
        "POPT": "Float64",
        "POPc1": "Float64",
        "POPc2": "Float64",
        "POPc3": "Float64",
        "POPc4": "Float64",
        "POPc5": "Float64",
        "POPc6": "Float64",
        "POPc7": "Float64",
        "POPc8": "Float64",
        "Ubat": "Float64",
        "CO2": "Float64",
        "BME_T": "Float64",
        "BME_H": "Float64",
        "BME_P": "Float64",
        "CPUTEMP": "Float64",
        "RPiT": "str",
        "RPiS": "str",
        "issue1": "str",
        "issue2": "str",
        "issue3": "str",
        "IYaw": "Float64",
        "IPitch": "Float64",
        "IRoll": "Float64",
        "ILat": "Float64",
        "ILong": "Float64",
        "IVX": "Float64",
        "IVY": "Float64",
        "IVZ": "Float64",
        "IAX": "Float64",
        "IAY": "Float64",
        "IAZ": "Float64",
        "Out1_T": "Float64",
        "Out1_H": "Float64",
        "Out2_T": "Float64",
        "Out2_H": "Float64",
        "Inlet2_T": "Float64",
        "Inlet2_H": "Float64",
        "UTCTime": "str",
        "Status": "str",
        "Lat": "str",
        "LatDir": "str",
        "Long": "str",
        "LongDir": "str",
        "Speed": "Float64",
        "Course": "Float64",
        "Date": "str",
        "MagVar": "Float64",
        "MVdir": "str",
        "GLat": "str",
        "GLatDir": "str",
        "GLong": "str",
        "GLongDir": "str",
        "GPSQ": "Float64",
        "GNSats": "Float64",
        "Hprec": "Float64",
        "GAlt": "Float64",
        "AltU": "str",
        "Geoidal": "Float64",
        "UTCTime2": "str",
        "Heading": "Float64",
        "HeadTrue": "str",
        "Roll": "Float64",
        "Pitch": "Float64",
        "Heave": "Float64",
        "RollAcc": "Float64",
        "PitchAcc": "Float64",
        "HeadAcc": "Float64",
        "GNSSqty": "Float64",
    },
    na_values=[],
    comment="#",
    cols_export=[
        "GAlt",  # GPS Altitude
        "Geoidal",
        "F_smp_prs",
        "CO2",
        "BME_T",
        "BME_H",
        "BME_P",
        "CPUTEMP",
        "RPiT",
        "RPiS",
        "Ubat",
        "Speed",
        "Course",
        "MagVar",
        "Inlet2_T",
        "Inlet2_H",
        "Out1_T",
        "Out1_H",
        "Out2_T",
        "Out2_H",
        "Heading",
        "HeadTrue",
        "Roll",
        "Pitch",
        "Heave",
        "RollAcc",
        "PitchAcc",
        "HeadAcc",
        "GNSSqty",
    ],
    cols_housekeeping=[
        "GAlt",
        "Geoidal",
        "F_cur_pos",
        "F_cntdown",
        "F_smp_flw",
        "F_smp_tmp",
        "F_smp_prs",
        "F_pump_pw",
        "F_psvolts",
        "F_err_rpt",
        "SO_S",
        "SO_D",
        "SO_U",
        "SO_V",
        "SO_W",
        "SO_T",
        "SO_H",
        "SO_P",
        "SO_PI",
        "SO_RO",
        "SO_MD",
        "POPID",
        "POPCHAIN",
        "POPtot",
        "POPf",
        "POPT",
        "POPc1",
        "POPc2",
        "POPc3",
        "POPc4",
        "POPc5",
        "POPc6",
        "POPc7",
        "POPc8",
        "Ubat",
        "CO2",
        "BME_T",
        "BME_H",
        "BME_P",
        "CPUTEMP",
        "RPiT",
        "RPiS",
        "issue1",
        "issue2",
        "issue3",
        "UTCTime",
        "Status",
        "Lat",
        "LatDir",
        "Long",
        "LongDir",
        "Speed",
        "Course",
        "Date",
        "MagVar",
        "MVdir",
        "Inlet2_T",
        "Inlet2_H",
        "Out1_T",
        "Out1_H",
        "Out2_T",
        "Out2_H",
        "GPSQ",
        "GNSats",
        "Hprec",
        "GAlt",
        "AltU",
        "Geoidal",
        "UTCTime2",
        "Heading",
        "HeadTrue",
        "Roll",
        "Pitch",
        "Heave",
        "RollAcc",
        "PitchAcc",
        "HeadAcc",
        "GNSSqty",
    ],
    export_order=100,
    pressure_variable="BME_P",
)

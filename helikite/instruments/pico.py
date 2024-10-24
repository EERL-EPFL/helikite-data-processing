"""
7) Pico -> Pico100217_220204_152813Eng.txt / Pico100217_220219_123139Eng.txt
(doesn't have pressure)

Gas monitor measuring CO, N2O and H2O

Can operate in two different modes. When it opeartes in differentioal mode,
there an additional variable called Differential.CO and should be used for the
CO reading.

In manual mode, look at CO_ppm (however, in this case, the baseline need to be
substracted but this might require some more conversation on how to do it.)

For quality check -> plot the following variables: win1Fit7 and win1Fit8
(if win1Fit8 has a sudden jump (very noticeable) this indicates a bad fit)
"""

from helikite.instruments.base import Instrument
import pandas as pd


class Pico(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "pico"

    def file_identifier(self, first_lines_of_csv) -> bool:
        if (
            "win0Fit0,win0Fit1,win0Fit2,win0Fit3,win0Fit4,win0Fit5,win0Fit6,"
            "win0Fit7,win0Fit8,win0Fit9,win1Fit0,win1Fit1,win1Fit2"
        ) in first_lines_of_csv[0]:
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe"""

        df["DateTime"] = pd.to_datetime(
            df["Time Stamp"], format="%m/%d/%Y %H:%M:%S.%f"
        )
        df.drop(columns=["Time Stamp"], inplace=True)

        # Round the milliseconds to the nearest second
        df["DateTime"] = pd.to_datetime(df.DateTime).round("s")

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


pico = Pico(
    dtype={
        "Time Stamp": "str",
        "Inlet Number": "Int64",
        "P (mbars)": "Float64",
        "T0 (degC)": "Float64",
        "T5 (degC)": "Float64",
        "Tgas(degC)": "Float64",
        "Laser PID Readout": "Int64",
        "Det PID Readout": "Int64",
        "win0Fit0": "Float64",
        "win0Fit1": "Float64",
        "win0Fit2": "Float64",
        "win0Fit3": "Float64",
        "win0Fit4": "Float64",
        "win0Fit5": "Float64",
        "win0Fit6": "Float64",
        "win0Fit7": "Float64",
        "win0Fit8": "Float64",
        "win0Fit9": "Float64",
        "win1Fit0": "Float64",
        "win1Fit1": "Float64",
        "win1Fit2": "Float64",
        "win1Fit3": "Float64",
        "win1Fit4": "Float64",
        "win1Fit5": "Float64",
        "win1Fit6": "Float64",
        "win1Fit7": "Float64",
        "win1Fit8": "Float64",
        "win1Fit9": "Float64",
        "Det Bkgd": "Float64",
        "Ramp Ampl": "Float64",
        "N2O (ppm)": "Float64",
        "H2O (ppm)": "Float64",
        "CO (ppm)": "Float64",
        "Mean N2O (ppm)": "Float64",
        "Mean H2O (ppm)": "Float64",
        "Differential CO (ppm)": "Float64",
        "Battery Charge (V)": "Int64",
        "Power Input (mV)": "Int64",
        "Current (mA)": "Int64",
        "SOC (%)": "Int64",
        "Battery T (degC)": "Float64",
        "FET T (degC)": "Float64",
    },
    export_order=300,
    cols_export=[
        "CO (ppm)",
        "N2O (ppm)",
        "H2O (ppm)",
        "Differential CO (ppm)",
        "Mean N2O (ppm)",
        "Mean H2O (ppm)",
    ],
    cols_housekeeping=[
        "Inlet Number",
        "P (mbars)",
        "T0 (degC)",
        "T5 (degC)",
        "Tgas(degC)",
        "Laser PID Readout",
        "Det PID Readout",
        "win0Fit0",
        "win0Fit1",
        "win0Fit2",
        "win0Fit3",
        "win0Fit4",
        "win0Fit5",
        "win0Fit6",
        "win0Fit7",
        "win0Fit8",
        "win0Fit9",
        "win1Fit0",
        "win1Fit1",
        "win1Fit2",
        "win1Fit3",
        "win1Fit4",
        "win1Fit5",
        "win1Fit6",
        "win1Fit7",
        "win1Fit8",
        "win1Fit9",
        "Det Bkgd",
        "Ramp Ampl",
        "N2O (ppm)",
        "H2O (ppm)",
        "CO (ppm)",
        "Mean N2O (ppm)",
        "Mean H2O (ppm)",
        "Differential CO (ppm)",
        "Battery Charge (V)",
        "Power Input (mV)",
        "Current (mA)",
        "SOC (%)",
        "Battery T (degC)",
        "FET T (degC)",
    ],
    pressure_variable="P (mbars)",
)

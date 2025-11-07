"""
CPC3007
Total particle concentration in size range of 7 - 2000 nm.
"""
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from helikite.instruments.base import Instrument


class CPC(Instrument):
    """
    Instrument definition for the cpc3007 sensor system.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def data_corrections(self, df, *args, **kwargs) -> pd.DataFrame:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.rename(columns={'Concentration (#/cm3)': 'totalconc_raw'})

        df = df.resample("1s").asfreq()
        df.insert(0, "DateTime", df.index)

        return df

    def file_identifier(self, first_lines_of_csv: List[str]) -> bool:
        if self.expected_header_value in first_lines_of_csv[17]:
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.date is None:
            raise ValueError(
                "No flight date provided. Necessary for CPC"
            )

        df["DateTime"] = df["Time"].apply(lambda t: pd.to_datetime(f"{self.date} {t}"))
        df.set_index("DateTime", inplace=True)
        df.index = df.index.astype("datetime64[s]")

        return df

    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            engine="python",
            skiprows=17,
            skipfooter=1,
            na_values=self.na_values,
            header=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df

    def CPC_STP_normalization(self, df):
        """
        Normalize CPC3007 concentrations to STP conditions and insert the results
        right after the existing CPC columns.
    
        Parameters:
        df (pd.DataFrame): DataFrame containing CPC measurements and metadata.
    
        Returns:
        df (pd.DataFrame): Updated DataFrame with STP-normalized columns inserted.
        """
        plt.close('all')
    
        # Constants for STP
        P_STP = 1013.25  # hPa
        T_STP = 273.15   # Kelvin
    
        # Measured conditions
        P_measured = df["flight_computer_pressure"]
        T_measured = df["Average_Temperature"] + 273.15  # Convert °C to Kelvin
    
        # Calculate STP correction
        correction_factor = (P_measured / P_STP) * (T_STP / T_measured)
        normalized_column = df['cpc_totalconc_raw'] * correction_factor
    
        # Prepare to insert
        cpc_columns = [col for col in df.columns if col.startswith('cpc_')]
        if cpc_columns:
            last_cpc_index = df.columns.get_loc(cpc_columns[-1]) + 1
        else:
            last_cpc_index = len(df.columns)
    
        # Insert STP-normalized column (only if it doesn't already exist)
        if 'cpc_totalconc_stp' in df.columns:
            df = df.drop(columns='cpc_totalconc_stp')
    
        df = pd.concat(
            [df.iloc[:, :last_cpc_index],
             pd.DataFrame({'cpc_totalconc_stp': normalized_column}, index=df.index),
             df.iloc[:, last_cpc_index:]],
            axis=1
        )
        
        # PLOT
        plt.figure(figsize=(8, 6))
        plt.plot(df['cpc_totalconc_raw'], df['Altitude'], label='Measured', color='blue', marker='.', linestyle='none')
        plt.plot(df['cpc_totalconc_stp'], df['Altitude'], label='STP-normalized', color='red', marker='.', linestyle='none')
        plt.xlabel('CPC3007 total concentration (cm$^{-3}$)', fontsize=12)
        plt.ylabel('Altitude (m)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        return df

cpc = CPC(
    name="cpc",
    dtype={
        "Time": "str",
        "Concentration (#/cm3)": "Int64"
    },
    expected_header_value="Time,Concentration (#/cm3),\n",
    header=0,
    pressure_variable=None,
)
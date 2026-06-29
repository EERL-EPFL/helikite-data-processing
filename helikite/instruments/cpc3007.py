"""
CPC3007
Total particle concentration in size range of 7 - 2000 nm.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helikite.instruments.base import Instrument, filter_columns_by_instrument


class CPC(Instrument):
    """
    Instrument definition for the cpc3007 sensor system.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "CPC"

    def data_corrections(self, df, *args, **kwargs) -> pd.DataFrame:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.rename(columns={'Concentration (#/cm3)': 'totalconc_raw',
                                'Concentration (#/cm�)': 'totalconc_raw'})

        df = df.resample("1s").asfreq()

        return df

    def file_identifier(self, first_lines_of_csv: list[str]) -> bool:
        if self.expected_header_value[:-4] in first_lines_of_csv[self.header]:
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.date is None:
            raise ValueError(
                "No flight date provided. Necessary for CPC"
            )

        df["DateTime"] = df["Time"].apply(lambda t: pd.to_datetime(f"{self.date} {t}", errors="coerce"))
        df.drop(columns=["Time"], inplace=True)
        df.dropna(subset=["DateTime"], inplace=True)
        df.set_index("DateTime", inplace=True)
        df.index = df.index.astype("datetime64[s]")

        return df

    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            engine="python",
            skipfooter=1,
            na_values=self.na_values,
            skiprows=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
            encoding_errors="replace",
        )

        return df

    def normalize(self, df: pd.DataFrame, reference_instrument: Instrument, verbose: bool, *args, **kwargs) -> pd.DataFrame:
        """
        Normalize CPC3007 concentrations to STP conditions and insert the results
        right after the existing CPC columns.
    
        Parameters:
        df (pd.DataFrame): DataFrame containing CPC measurements and metadata.
    
        Returns:
        df (pd.DataFrame): Updated DataFrame with STP-normalized columns inserted.
        """
        metadata = kwargs.get("metadata", None)

        if metadata is None:
            raise ValueError("CPC requires metadata (cpc_mode missing)")
    
        mode = metadata.cpc_mode
        
        # Constants for STP
        P_STP = 1013.25  # hPa
        T_STP = 273.15   # Kelvin
    
        # Measured conditions
        if mode == "V":
            P_measured = df[f"{reference_instrument.name}_pressure"]
            T_measured = df["Average_Temperature"] + 273.15  # Convert °C to Kelvin
            output_column = "cpc_totalconc_stp"
    
        elif mode == "S":
            P_measured = df["Pressure_ground"]
            T_measured = df["Temperature_ground"]
            output_column = "cpc_totalconc_surface_stp"

            # Ensure the vertical CPC column exists
            if "cpc_totalconc_stp" not in df.columns:
                df["cpc_totalconc_stp"] = np.nan

        else:
            raise ValueError(f"Unknown CPC mode: {mode}")
            
        # Calculate STP correction
        correction_factor = (P_measured / P_STP) * (T_STP / T_measured)
        normalized_column = df['cpc_totalconc_raw'] * correction_factor
    
        # Prepare to insert
        cpc_columns = filter_columns_by_instrument(df.columns, cpc)
        if cpc_columns:
            last_cpc_index = df.columns.get_loc(cpc_columns[-1]) + 1
        else:
            last_cpc_index = len(df.columns)
    
        # Insert STP-normalized column (only if it doesn't already exist)
        if output_column in df.columns:
            df = df.drop(columns=output_column)
    
        df = pd.concat(
            [df.iloc[:, :last_cpc_index],
             pd.DataFrame({output_column: normalized_column}, index=df.index),
             df.iloc[:, last_cpc_index:]],
            axis=1
        )

        return df

    def plot_raw_and_normalized(self, df: pd.DataFrame, verbose: bool, *args, **kwargs):
        
        metadata = kwargs.get("metadata", None)

        if metadata is None:
            raise ValueError("CPC plotting requires metadata")
    
        mode = metadata.cpc_mode
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # -------------------------
        # VERTICAL PROFILE
        # -------------------------
        if mode == "V":
        
            x_raw = df["cpc_totalconc_raw"]
            x_stp = df["cpc_totalconc_stp"] if "cpc_totalconc_stp" in df.columns else None
        
            y = df["Altitude"]
            ylabel = "Altitude (m)"
            xlabel = "CPC3007 total concentration (cm$^{-3}$)"
        
            plt.plot(x_raw, y, label="Measured", color="blue", marker=".", linestyle="none")
        
            if x_stp is not None:
                plt.plot(x_stp, y, label="STP-normalized", color="red", marker=".", linestyle="none")
        
        # -------------------------
        # SURFACE TIME SERIES
        # -------------------------
        elif mode == "S":
        
            # time axis
            x = df.index  # or df["Time"] if you have explicit column
        
            plt.plot(x, df["cpc_totalconc_raw"], label="Measured", color="blue", marker=".", linestyle="none")
        
            if "cpc_totalconc_surface_stp" in df.columns:
                plt.plot(x, df["cpc_totalconc_surface_stp"], label="STP-normalized", color="red", marker=".", linestyle="none")
        
            xlabel = "Time"
            ylabel = "CPC3007 total concentration (cm$^{-3}$)"
        
        else:
            raise ValueError(f"Unknown CPC mode: {mode}")
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


cpc = CPC(
    name="cpc",
    dtype={
        "Time": "str",
        "Concentration (#/cm3)": "Int64"
    },
    expected_header_value="Time,Concentration (#/cm3),\n",
    cols_final=["totalconc_stp", "totalconc_surface_stp"],
    header=17,
    pressure_variable=None,
    rename_dict={'cpc_totalconc_stp': 'CPC_total_N',
                'cpc_totalconc_surface_stp': 'CPC_surface_total_N'},
)
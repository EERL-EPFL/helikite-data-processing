"""
3) POPS ->  HK_20220929x001.csv (has pressure)

The POPS is an optical particle counter. It provides information on the
particle number concentration (how many particles per cubic centimeter)
and the size distribution for particles larger than 180 nm roughly.
Resolution: 1 sec

Important variables to keep:
DateTime, P, POPS_Flow, b0 -> b15

PartCon needs to be re-calculated by adding b3 to b15 and deviding by averaged
POPS_Flow (b0 -> b15 can be converted to dN/dlogDp values with conversion
factors I have)

Housekeeping variables to look at:
POPS_flow -> flow should be just below 3, and check for variability increase
"""
import datetime
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from helikite.instruments.base import Instrument, filter_columns_by_instrument


class POPS(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "POPS"

    @property
    def has_size_distribution(self) -> bool:
        return True

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        Using values in the time_offset variable, correct DateTime index
        """

        df["DateTime"] = pd.to_datetime(df["DateTime"], unit="s")

        # Round the milliseconds to the nearest second
        df["DateTime"] = pd.to_datetime(df.DateTime).dt.round("1s")

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)
        df.index = df.index.floor('s').astype("datetime64[s]")

        return df

    def data_corrections(
        self, df: pd.DataFrame, *args, **kwargs
    ) -> pd.DataFrame:

        df.columns = df.columns.str.strip()

        # Calculate PartCon_186
        df["PartCon_186"] = (
            df["b3"]
            + df["b4"]
            + df["b5"]
            + df["b6"]
            + df["b7"]
            + df["b8"]
            + df["b9"]
            + df["b10"]
            + df["b11"]
            + df["b12"]
            + df["b13"]
            + df["b14"]
            + df["b15"]
        ) / df["POPS_Flow"].mean()
        df.drop(columns="PartCon", inplace=True)

        return df

    def read_data(self) -> pd.DataFrame:

        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            na_values=self.na_values,
            skiprows=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df

    def calculate_derived(self, df: pd.DataFrame, verbose: bool, *args, **kwargs) -> pd.DataFrame:
        """
        This function calculates the total concentration of POPS particles and adds it to the dataframe.

        Parameters:
        - df: DataFrame with POPS data and altitude.

        Returns:
        - df: Updated DataFrame with POPS total concentration and dNdlogDp for each bin.
        """
        # Define the path to the POPS DP notes file
        filenotePOPS = Path(__file__).parent / "POPS_dNdlogDp.txt"

        # Read the DP notes file
        dp_notes = pd.read_csv(filenotePOPS, sep="\t", skiprows=[0])

        # Select only POPS data columns
        pops_data = filter_columns_by_instrument(df.columns, pops)
        df_pops = df[pops_data].copy()

        # Remove duplicate column names before processing
        df_pops = df_pops.loc[:, ~df_pops.columns.duplicated()].copy()

        # Calculate dN for the POPS columns and dNdlogdP for each bin
        df_pops = self._dNdlogDp_calculation(df_pops, dp_notes)

        # Insert pops into df at the right position
        if pops_data:
            # Find the index of the last "pops_" column
            last_pops_index = df.columns.get_loc(pops_data[-1]) + 1  # Insert after this column
        else:
            # If no such column exists, append to the end
            last_pops_index = len(df.columns)

        # Concatenate the df_pops to the original df
        df = pd.concat([df.iloc[:, :last_pops_index], df_pops, df.iloc[:, last_pops_index:]], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    def normalize(self, df: pd.DataFrame, reference_instrument: Instrument,
                  verbose: bool, *args, **kwargs) -> pd.DataFrame:
        """
        Normalize POPS concentrations to STP conditions.

        Parameters:
        df (pd.DataFrame): DataFrame containing POPS measurements and necessary metadata
                           like 'flight_computer_pressure' and 'Average_Temperature'.

        Returns:
        df (pd.DataFrame): Updated DataFrame with new STP-normalized columns added.
        """
        # Constants for STP
        P_STP = 1013.25  # hPa
        T_STP = 273.15  # Kelvin

        # Measured conditions
        P_measured = df[f"{reference_instrument.name}_pressure"]
        T_measured = df["Average_Temperature"] + 273.15  # Convert °C to Kelvin

        # Calculate the STP correction factor
        correction_factor = (P_measured / P_STP) * (T_STP / T_measured)

        # List of columns to correct
        columns_to_normalize = [
            'pops_total_conc', 'pops_b3_dlogDp', 'pops_b4_dlogDp', 'pops_b5_dlogDp',
            'pops_b6_dlogDp', 'pops_b7_dlogDp', 'pops_b8_dlogDp', 'pops_b9_dlogDp',
            'pops_b10_dlogDp', 'pops_b11_dlogDp', 'pops_b12_dlogDp',
            'pops_b13_dlogDp', 'pops_b14_dlogDp', 'pops_b15_dlogDp'
        ]

        # Dictionary to hold new columns temporarily
        normalized_columns = {}

        # Calculate the normalized values
        for col in columns_to_normalize:
            if col in df.columns:
                normalized_columns[col + '_stp'] = df[col] * correction_factor

        # Insert the new columns after the last existing POPS column
        pops_columns = filter_columns_by_instrument(df.columns, pops)
        if pops_columns:
            last_pops_index = df.columns.get_loc(pops_columns[-1]) + 1
        else:
            last_pops_index = len(df.columns)

        # Merge the DataFrame
        df = pd.concat(
            [df.iloc[:, :last_pops_index],
             pd.DataFrame(normalized_columns, index=df.index),
             df.iloc[:, last_pops_index:]],
            axis=1
        )

        return df

    def plot_raw_and_normalized(self, df: pd.DataFrame, verbose: bool, *args, **kwargs):
        """Plots POPS concentration, raw and normalized to STP conditions, against altitude"""
        plt.figure(figsize=(8, 6))

        plt.plot(df['pops_total_conc'], df['Altitude'], label='Measured', color='blue')
        if 'pops_total_conc_stp' in df.columns:
            plt.plot(df['pops_total_conc_stp'], df['Altitude'], label='STP-normalized', color='red', linestyle='--')
        plt.xlabel('POPS total concentration (cm$^{-3}$)', fontsize=12)
        plt.ylabel('Altitude (m)', fontsize=12)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    def plot_distribution(self, df: pd.DataFrame, verbose: bool,
                          time_start: datetime.datetime, time_end: datetime.datetime,
                          subplot: tuple[plt.Figure, plt.Axes] | None = None):
        """
        This function generates a contour plot for POPS size distribution and total concentration.

        Parameters:
        - df: DataFrame with the POPS data.
        - time_start: Optional, start time for the x-axis (datetime formatted).
        - time_end: Optional, end time for the x-axis (datetime formatted).
        """
        if subplot is not None:
            fig, ax = subplot
            is_custom_subplot = True
        else:
            fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
            is_custom_subplot = False

        if time_start is not None:
            df = df[df.index >= pd.to_datetime(time_start)]
        if time_end is not None:
            df = df[df.index <= pd.to_datetime(time_end)]

        # Define pops_dlogDp variable from Hendix documentation
        pops_dia = [
            149.0801282, 162.7094017, 178.3613191, 195.2873341,
            212.890625, 234.121875, 272.2136986, 322.6106374,
            422.0817873, 561.8906456, 748.8896681, 1054.138693,
            1358.502538, 1802.347716, 2440.99162, 3061.590212
        ]

        pops_dlogDp = [
            0.036454582, 0.039402553, 0.040330922, 0.038498955,
            0.036550107, 0.045593506, 0.082615487, 0.066315868,
            0.15575785, 0.100807113, 0.142865049, 0.152476328,
            0.077693935, 0.157186601, 0.113075192, 0.086705426
        ]

        # Define the range of columns for POPS concentration
        start_conc = self.column_name(df, 'pops_b3_dlogDp_stp')
        end_conc = self.column_name(df, 'pops_b15_dlogDp_stp')

        # Extract the relevant columns
        counts = df.loc[:, start_conc:end_conc]
        counts = counts.set_index(df.index).astype(float)

        vmax_value = np.nanmax(counts.values) if not counts.isna().all().all() else np.nan
        print(f"max value ({self.name}): {vmax_value}")

        # Create 2D grid
        bin_diameters = pops_dia[3:16]
        xx, yy = np.meshgrid(counts.index.values, bin_diameters)
        Z = counts.values.T

        # Color normalization
        norm = mcolors.LogNorm(vmin=1, vmax=300)

        # Create the pcolormesh plot
        mesh = ax.pcolormesh(xx, yy, Z, cmap='viridis', norm=norm, shading="gouraud")

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = inset_axes(ax, width="1.5%", height="100%", loc='lower left',
                         bbox_to_anchor=(1.08, -0.025, 1, 1), bbox_transform=ax.transAxes)
        cb = fig.colorbar(mesh, cax=cax, orientation='vertical')
        cb.set_label('dN/dlogD$_p$ (cm$^{-3}$)', fontsize=12, fontweight='bold')
        cb.ax.tick_params(labelsize=11)

        if not is_custom_subplot:
            # Define custom date formatter for better x-axis labels
            class CustomDateFormatter(mdates.DateFormatter):
                def __init__(self, fmt="%H:%M", date_fmt="%Y-%m-%d %H:%M", *args, **kwargs):
                    super().__init__(fmt, *args, **kwargs)
                    self.date_fmt = date_fmt
                    self.prev_date = None

                def __call__(self, x, pos=None):
                    date = mdates.num2date(x)
                    current_date = date.date()
                    if self.prev_date != current_date:
                        self.prev_date = current_date
                        return date.strftime(self.date_fmt)
                    else:
                        return date.strftime(self.fmt)

            # Apply custom formatter
            custom_formatter = CustomDateFormatter(fmt="%H:%M", date_fmt="%Y-%m-%d %H:%M")
            ax.xaxis.set_major_formatter(custom_formatter)
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            ax.tick_params(axis='x', rotation=90, labelsize=11)

        # Set axis labels and limits
        ax.set_ylim(180, 3370)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_yscale('log')
        ax.set_ylabel('Part. Diameter (nm)', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.5, axis='x')
        ax.grid(False, axis='y')

        if not is_custom_subplot:
            ax.set_title('POPS size distribution and total concentration', fontsize=13, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12, fontweight='bold', labelpad=10)

        # Plot total concentration on a secondary y-axis
        total_conc = df[self.column_name(df, 'pops_total_conc_stp')]
        total_conc_max = total_conc.max() if not total_conc.isna().all() else 40
        ax2 = ax.twinx()
        ax2.plot(df.index, total_conc, color='red', linewidth=2, label='Total POPS Conc.')
        ax2.set_ylabel('POPS conc (cm$^{-3}$)', fontsize=12, fontweight='bold', color='red', labelpad=8)
        ax2.tick_params(axis='y', labelsize=11, colors='red')
        ax2.set_ylim(-20, total_conc_max * 1.1)

        if not is_custom_subplot:
            plt.show()

    @staticmethod
    def _dNdlogDp_calculation(df_pops, dp_notes):
        # Adjust dN_pops and calculate dNdlogDp
        popsflow_mean = df_pops['pops_POPS_Flow'].mean()  # 2.9866
        dN_pops = df_pops.filter(like='pops_b') / popsflow_mean
        df_pops.loc[:, 'pops_total_conc'] = dN_pops.loc[:, 'pops_b3':'pops_b15'].sum(axis=1, skipna=True, min_count=1)
        dNdlogDp = dN_pops.loc[:, 'pops_b3':'pops_b15'].div(dp_notes['dlogdp'].iloc[3:].values, axis=1).add_suffix(
            '_dlogDp')

        # Add dNdlogDp columns to df
        df_pops = pd.concat([df_pops, dNdlogDp], axis=1)
        return df_pops


pops = POPS(
    name="pops",
    dtype={
        "DateTime": "Float64",
        "Status": "Int64",
        "PartCt": "Int64",
        "PartCon": "Float64",
        "BL": "Int64",
        "BLTH": "Int64",
        "STD": "Float64",
        "P": "Float64",
        "TofP": "Float64",
        "POPS_Flow": "Float64",
        "PumpFB": "Int64",
        "LDTemp": "Float64",
        "LaserFB": "Int64",
        "LD_Mon": "Int64",
        "Temp": "Float64",
        "BatV": "Float64",
        "Laser_Current": "Float64",
        "Flow_Set": "Float64",
        "PumpLife_hrs": "Float64",
        "BL_Start": "Int64",
        "TH_Mult": "Int64",
        "nbins": "Int64",
        "logmin": "Float64",
        "logmax": "Float64",
        "Skip_Save": "Int64",
        "MinPeakPts": "Int64",
        "MaxPeakPts": "Int64",
        "RawPts": "Int64",
        "b0": "Int64",
        "b1": "Int64",
        "b2": "Int64",
        "b3": "Int64",
        "b4": "Int64",
        "b5": "Int64",
        "b6": "Int64",
        "b7": "Int64",
        "b8": "Int64",
        "b9": "Int64",
        "b10": "Int64",
        "b11": "Int64",
        "b12": "Int64",
        "b13": "Int64",
        "b14": "Int64",
        "b15": "Int64",
    },
    expected_header_value="DateTime, Status, PartCt, PartCon, BL, BLTH, STD, P, TofP, "
                          "POPS_Flow, PumpFB, LDTemp, LaserFB, LD_Mon, Temp, BatV, "
                          "Laser_Current, Flow_Set,PumpLife_hrs, BL_Start, TH_Mult, nbins, "
                          "logmin, logmax, Skip_Save, MinPeakPts,MaxPeakPts, RawPts,b0,b1,"
                          "b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15\n",
    export_order=400,
    cols_export=[
        "P",
        "PartCon_186",
        "POPS_Flow",
        "b0",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "b8",
        "b9",
        "b10",
        "b11",
        "b12",
        "b13",
        "b14",
        "b15",
    ],
    cols_housekeeping=[
        "Status",
        "PartCt",
        "PartCon_186",
        "BL",
        "BLTH",
        "STD",
        "P",
        "TofP",
        "POPS_Flow",
        "PumpFB",
        "LDTemp",
        "LaserFB",
        "LD_Mon",
        "Temp",
        "BatV",
        "Laser_Current",
        "Flow_Set",
        "PumpLife_hrs",
        "BL_Start",
        "TH_Mult",
        "nbins",
        "logmin",
        "logmax",
        "Skip_Save",
        "MinPeakPts",
        "MaxPeakPts",
        "RawPts",
        "b0",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "b8",
        "b9",
        "b10",
        "b11",
        "b12",
        "b13",
        "b14",
        "b15",
    ],
    cols_final=[f"b{i}_dlogDp_stp" for i in range(3, 16)] + ["total_conc_stp"],
    pressure_variable="P",
    temperature_variable="Temp",
    coupled_columns=[
        [f"pops_b{i}" for i in range(16)] +
        [f"pops_b{i}_dlogDp" for i in range(3, 16)] +
        ["pops_total_conc", "pops_PartCon_186"],
    ],
    rename_dict={f'pops_b{i}_dlogDp_stp': f'POPS_b{i}' for i in range(3, 16)} | {'pops_total_conc_stp': 'POPS_total_N'},
)

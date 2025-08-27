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

from helikite.instruments.base import Instrument
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcols
import matplotlib.dates as mdates
import os
from matplotlib.dates import HourLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path


class POPS(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "pops"

    def file_identifier(self, first_lines_of_csv) -> bool:
        if first_lines_of_csv[0] == (
            "DateTime, Status, PartCt, PartCon, BL, BLTH, STD, P, TofP, "
            "POPS_Flow, PumpFB, LDTemp, LaserFB, LD_Mon, Temp, BatV, "
            "Laser_Current, Flow_Set,PumpLife_hrs, BL_Start, TH_Mult, nbins, "
            "logmin, logmax, Skip_Save, MinPeakPts,MaxPeakPts, RawPts,b0,b1,"
            "b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15\n"
        ):
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        Using values in the time_offset variable, correct DateTime index
        """

        df["DateTime"] = pd.to_datetime(df["DateTime"], unit="s")

        # Round the milliseconds to the nearest second
        df["DateTime"] = pd.to_datetime(df.DateTime).dt.round("1s")

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)
        df.index = df.index.floor('s') #astype("datetime64[s]")

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
            header=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df


pops = POPS(
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
    pressure_variable="P",
)


def dNdlogDp_calculation(df_pops,dp_notes):
    
    # Adjust dN_pops and calculate dNdlogDp
    popsflow_mean = df_pops['pops_POPS_Flow'].mean()#2.9866
    dN_pops = df_pops.filter(like='pops_b') / popsflow_mean
    df_pops.loc[:,'pops_total_conc'] = dN_pops.loc[:, 'pops_b3':'pops_b15'].sum(axis=1)
    dNdlogDp = dN_pops.loc[:, 'pops_b3':'pops_b15'].div(dp_notes['dlogdp'].iloc[3:].values, axis=1).add_suffix('_dlogDp')
    
    # Add dNdlogDp columns to df
    df_pops = pd.concat([df_pops, dNdlogDp], axis=1)
    return df_pops


def plot_pops_distribution(df, time_start=None, time_end=None):
    """
    This function generates a contour plot for POPS size distribution and total concentration.

    Parameters:
    - df: DataFrame with the POPS data.
    - time_start: Optional, start time for the x-axis (datetime formatted).
    - time_end: Optional, end time for the x-axis (datetime formatted).
    """
    plt.close('all')

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

    # Define the range of columns for which you want the concentration
    start_conc = 'pops_b3_dlogDp_stp'
    end_conc = 'pops_b15_dlogDp_stp'

    # Extract the relevant columns
    counts = df.loc[:, start_conc:end_conc]
    dtimes = df.index

    # Set dataframe index to time to allow resampling if needed
    counts = counts.set_index(dtimes)

    # Ensure values are float (important for pcolormesh)
    counts = counts.astype(float)
    vmax_value = counts.values.max()
    print(vmax_value)

    # Create 2D grid from times and bin diameters
    bin_diameters = pops_dia[3:16]  # Pops diameters corresponding to b3 to b15 (13 values)
    xx, yy = np.meshgrid(counts.index.values, bin_diameters)
    Z = counts.values.T  # Shape must be (nrows = bins, ncols = time steps)

    # Start plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)

    # Color normalization
    norm = mcols.LogNorm(vmin=1, vmax=600)

    # Create the pcolormesh plot
    mesh = ax.pcolormesh(xx, yy, Z, cmap='viridis', norm=norm, shading="gouraud")

    # Add colorbar
    cb = fig.colorbar(mesh, ax=ax, orientation='vertical', location='right', pad=0.02)
    cb.set_label('dN/dlogD$_p$ (cm$^{-3}$)', fontsize=12, fontweight='bold')
    cb.ax.tick_params(labelsize=12)

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
    ax.tick_params(axis='x', rotation=90, labelsize=12)

    # Set axis labels and limits
    if time_start and time_end:
        ax.set_xlim(pd.Timestamp(time_start), pd.Timestamp(time_end))
    ax.set_ylim(180, 3370)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_yscale('log')
    ax.set_ylabel('Particle Diameter (nm)', fontsize=12, fontweight='bold')
    ax.set_title('POPS size distribution and total concentration', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold', labelpad=10)

    # Plot total concentration on a secondary y-axis
    total_conc = df['pops_total_conc_stp']
    ax2 = ax.twinx()
    ax2.plot(total_conc.index, total_conc, color='red', linewidth=2)
    ax2.set_ylabel('POPS total conc (cm$^{-3}$)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelsize=12, colors='red')
    ax2.set_ylim(-20, total_conc.max() * 1.1)

    plt.show()


def POPS_total_conc_dNdlogDp(df):
    """
    This function calculates the total concentration of POPS particles and adds it to the dataframe.
    It also plots the POPS total concentration against altitude.

    Parameters:
    - df: DataFrame with POPS data and altitude.

    Returns:
    - df: Updated DataFrame with POPS total concentration and dNdlogDp for each bin.
    """
    plt.close('all')
    
    # Define the path to the POPS DP notes file
    filenotePOPS = os.path.join(os.getcwd(), os.pardir, "helikite", "instruments", "POPS_dNdlogDp.txt")
    
    # Read the DP notes file
    dp_notes = pd.read_csv(filenotePOPS, sep="\t", skiprows=[0])

    # Select only POPS data columns
    pops_data = [col for col in df if col.startswith('pops_')]
    df_pops = df[pops_data].copy()

    # Remove duplicate column names before processing
    df_pops = df_pops.loc[:, ~df_pops.columns.duplicated()].copy()

    # Calculate dN for the POPS columns and dNdlogdP for each bin
    df_pops = dNdlogDp_calculation(df_pops, dp_notes)

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

    # PLOT
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.plot(df["pops_total_conc"], df['Altitude'], label='POPS total conc', color='teal', marker='.', linestyle='none')
    ax1.grid(ls='--')
    ax1.set_xlabel('POPS total concentration ($cm^{-3}$)')
    ax1.set_ylabel("Altitude (m)")
    ax1.legend()
    plt.tight_layout()
    
    # Show plot
    plt.show()

    return df


def POPS_STP_normalization(df):
    """
    Normalize POPS concentrations to STP conditions and plot the results.

    Parameters:
    df (pd.DataFrame): DataFrame containing POPS measurements and necessary metadata 
                       like 'flight_computer_pressure' and 'Average_Temperature'.

    Returns:
    df (pd.DataFrame): Updated DataFrame with new STP-normalized columns added.
    """
    plt.close('all')
    
    # Constants for STP
    P_STP = 1013.25  # hPa
    T_STP = 273.15   # Kelvin

    # Measured conditions
    P_measured = df["flight_computer_pressure"]
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
    pops_columns = [col for col in df.columns if col.startswith('pops_')]
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

    # PLOT
    plt.close('all')
    plt.figure(figsize=(8, 6))
    plt.plot(df['pops_total_conc'], df['Altitude'], label='Measured', color='blue')
    plt.plot(df['pops_total_conc_stp'], df['Altitude'], label='STP-normalized', color='red', linestyle='--')
    plt.xlabel('POPS total concentration (cm$^{-3}$)', fontsize=12)
    plt.ylabel('Altitude (m)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df

def apply_pops_zero_mask(df: pd.DataFrame, metadata, column_prefix='pops_b', total_col='pops_total_conc', interruption_sec=30) -> pd.DataFrame:
    """
    Masks all periods where POPS_total_conc == 0, including short interruptions (< interruption_sec).
    Sets affected columns to NaN and returns the modified DataFrame.
    Also saves the mask as a CSV.

    Parameters:
        df (pd.DataFrame): The input DataFrame with datetime index.
        metadata: An object with `flight_date` and `flight` attributes.
        column_prefix (str): Prefix of bin columns (default 'pops_b').
        total_col (str): Total concentration column name.
        interruption_sec (int): Maximum allowed gap in seconds.

    Returns:
        pd.DataFrame: Modified DataFrame with masked values.
    """
    # Step 1: Create a mask where total concentration == 0
    zero_mask = df[total_col] == 0

    # Step 2: Transition groups
    transition = (zero_mask != zero_mask.shift()).cumsum()

    # Step 3: Final mask with gap tolerance
    final_mask = pd.Series(False, index=df.index)
    interruption_tolerance = pd.Timedelta(seconds=interruption_sec)
    grouped = df.groupby(transition)
    prev_end = None

    for _, group in grouped:
        group_is_zero = zero_mask.loc[group.index[0]]
        if group_is_zero:
            final_mask.loc[group.index] = True
            prev_end = group.index[-1]
        else:
            if prev_end is not None:
                gap_duration = group.index[-1] - group.index[0]
                if gap_duration < interruption_tolerance:
                    final_mask.loc[group.index] = True
                else:
                    prev_end = None

    # Step 4: Apply mask to relevant columns
    pops_cols_to_nan = [f'{column_prefix}{i}' for i in range(16)] + [total_col]
    df.loc[final_mask, pops_cols_to_nan] = np.nan

    # Step 5: Save the mask
    pops_mask = pd.DataFrame(index=df.index)
    pops_mask['pops_zero_flag'] = final_mask
    output_path = Path(f'C:/Users/temel/Desktop/EERL/Campaigns/03_ORACLES/Neumayer_2024/Data/Processing/Level1/Level1_{metadata.flight_date}_Flight_{metadata.flight}_POPSmask.csv')
    pops_mask.to_csv(output_path)

    return df

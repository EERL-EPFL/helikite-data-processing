"""
Mini Cloud Droplet Anlayzer (mCDA)
mCDA -> mCDA_output_2025-02-12_A (has pressure)

Important variables to keep:

!!! Raspery Pie of the mCDA has an autonomous timestamp, make sur it has been corrected when creating the output file.

"""

from helikite.instruments.base import Instrument
from helikite.constants import constants
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcols
import matplotlib.dates as mdates

# Define logger
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


class mCDA(Instrument):
    """
    Instrument definition for the mcda sensor system.
    Handles timestamp creation and optional corrections.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "mcda"
        

    def file_identifier(self, first_lines_of_csv) -> bool:
        """
        Identify if the file matches the expected mCDA CSV header.
        Only checks the first few columns for matching names, ignoring the rest.
        """
        # Define the expected header as a list of the first few column names
        expected_header = [
            "DateTime", "timestamp_x", "set_flow", "actual_flow", "flow_diff", "power %", "dataB 1"
        ]
        
        # Split the first line of the CSV by commas (assuming it's CSV-formatted)
        header_columns = first_lines_of_csv[0].strip().split(',')
        
        # Compare only the first few columns
        if header_columns[:len(expected_header)] == expected_header:
            return True
        
        return False


    
    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set the DateTime as index of the dataframe and correct if needed
        Using values in the time_offset variable, correct DateTime index
        """

        df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y%m%d%H%M%S")

        # Round the milliseconds to the nearest second
        #df["DateTime"] = pd.to_datetime(df.DateTime).dt.round("1s")

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)
        df.index = df.index.floor('s') #astype("datetime64[s]")

        return df


    def read_data(self) -> pd.DataFrame:
        try:
            # First read everything as string to avoid crashing on weird values like '000A'
            df = pd.read_csv(
                self.filename,
                dtype=str,
                na_values=self.na_values,
                header=self.header,
                delimiter=self.delimiter,
                lineterminator=self.lineterminator,
                comment=self.comment,
                names=self.names,
                index_col=self.index_col,
            )
    
            # Then attempt conversion to numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
            return df
    
        except Exception as e:
            logger.error(f"Failed to read and convert mCDA data from {self.filename}: {e}")
            raise

        

    def data_corrections(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply any custom corrections here.
        For now, this is a pass-through.
        """
        return df


mcda = mCDA(
    dtype={
        "DateTime": "Float64",
        "timestamp_x": "Float64",
        "set_flow": "Float64",
        "actual_flow": "Float64",
        "flow_diff": "Float64",
        "power %": "Float64",
        **{f"dataB {i}": "Float64" for i in range(1, 513)},
        "pcount": "Float64",
        "pm 1": "Float64",
        "pm 2.5": "Float64",
        "pm 4": "Float64",
        "pm 10": "Float64",
        "pmtot": "Float64",
        "timestamp_y": "Float64",
        "Temperature": "Float64",
        "Pressure": "Float64",
        "RH": "Float64",
        "pmav": "Float64",
        "offset1": "Float64",
        "offset2": "Float64",
        "calib1": "Float64",
        "calib2": "Float64",
        "measurement_nbr": "Float64"
    },
    na_values=["NA", "-9999.00"],
    comment="#",
    cols_export=[
        "actual_flow",
        *[f"dataB {i}" for i in range(1, 513)],
        "pcount",
        "pm 1",
        "pm 2.5",
        "pm 4",
        "pm 10",
        "pmtot",
        "Temperature",
        "Pressure",
        "RH"
    ],
    cols_housekeeping=[
        "DateTime",
        "timestamp_x",
        "set_flow",
        "actual_flow",
        "flow_diff",
        "power %",
        *[f"dataB {i}" for i in range(1, 513)],
        "pcount",
        "pm 1",
        "pm 2.5",
        "pm 4",
        "pm 10",
        "pmtot",
        "timestamp_y",
        "Temperature",
        "Pressure",
        "RH",
        "pmav",
        "offset1",
        "offset2",
        "calib1",
        "calib2",
        "measurement_nbr"
    ],
    export_order=730,
    pressure_variable="Pressure",
)

# Midpoint diameters
Midpoint_diameter_list = np.array([
    0.244381, 0.246646, 0.248908, 0.251144, 0.253398, 0.255593,
    0.257846, 0.260141, 0.262561, 0.265062, 0.267712, 0.270370,
    0.273159, 0.275904, 0.278724, 0.281554, 0.284585, 0.287661,
    0.290892, 0.294127, 0.297512, 0.300813, 0.304101, 0.307439,
    0.310919, 0.314493, 0.318336, 0.322265, 0.326283, 0.330307,
    0.334409, 0.338478, 0.342743, 0.347102, 0.351648, 0.356225,
    0.360972, 0.365856, 0.371028, 0.376344, 0.382058, 0.387995,
    0.394223, 0.400632, 0.407341, 0.414345, 0.421740, 0.429371,
    0.437556, 0.446036, 0.454738, 0.463515, 0.472572, 0.481728,
    0.491201, 0.500739, 0.510645, 0.520720, 0.530938, 0.541128,
    0.551563, 0.562058, 0.572951, 0.583736, 0.594907, 0.606101,
    0.617542, 0.628738, 0.640375, 0.652197, 0.664789, 0.677657,
    0.691517, 0.705944, 0.721263, 0.736906, 0.753552, 0.770735,
    0.789397, 0.808690, 0.829510, 0.851216, 0.874296, 0.897757,
    0.922457, 0.948074, 0.975372, 1.003264, 1.033206, 1.064365,
    1.097090, 1.130405, 1.165455, 1.201346, 1.239589, 1.278023,
    1.318937, 1.360743, 1.403723, 1.446000, 1.489565, 1.532676,
    1.577436, 1.621533, 1.667088, 1.712520, 1.758571, 1.802912,
    1.847836, 1.891948, 1.937088, 1.981087, 2.027604, 2.074306,
    2.121821, 2.168489, 2.216644, 2.263724, 2.312591, 2.361099,
    2.412220, 2.464198, 2.518098, 2.571786, 2.628213, 2.685162,
    2.745035, 2.805450, 2.869842, 2.935997, 3.005175, 3.074905,
    3.148598, 3.224051, 3.305016, 3.387588, 3.476382, 3.568195,
    3.664863, 3.761628, 3.863183, 3.965651, 4.072830, 4.179050,
    4.289743, 4.400463, 4.512449, 4.621025, 4.731530, 4.839920,
    4.949855, 5.057777, 5.169742, 5.281416, 5.395039, 5.506828,
    5.621488, 5.734391, 5.849553, 5.962881, 6.081516, 6.200801,
    6.322133, 6.441786, 6.565130, 6.686935, 6.813017, 6.938981,
    7.071558, 7.205968, 7.345185, 7.483423, 7.628105, 7.774385,
    7.926945, 8.080500, 8.247832, 8.419585, 8.598929, 8.780634,
    8.973158, 9.167022, 9.372760, 9.582145, 9.808045, 10.041607,
    10.287848, 10.537226, 10.801172, 11.068405, 11.345135,
    11.621413, 11.910639, 12.200227, 12.492929, 12.780176,
    13.072476, 13.359067, 13.651163, 13.937329, 14.232032,
    14.523919, 14.819204, 15.106612, 15.402110, 15.695489,
    15.998035, 16.297519, 16.610927, 16.926800, 17.250511,
    17.570901, 17.904338, 18.239874, 18.588605, 18.938763,
    19.311505, 19.693678, 20.093464, 20.498208, 20.927653,
    21.366609, 21.827923, 22.297936, 22.802929, 23.325426,
    23.872344, 24.428708, 25.016547, 25.616663, 26.249815,
    26.888493, 27.563838, 28.246317, 28.944507, 29.626186,
    30.323440, 31.005915, 31.691752, 32.353900, 33.030123,
    33.692286, 34.350532, 34.984611, 35.626553, 36.250913,
    36.878655, 37.489663, 38.121550, 38.748073, 39.384594,
    40.008540, 40.654627, 41.292757, 41.937789, 42.578436
])

def mcda_concentration_calculations(df: pd.DataFrame) -> pd.DataFrame:
    # Select columns from 'mcda_dataB 1' to 'mcda_dataB 256'
    dataB_cols = df.loc[:, 'mcda_dataB 1':'mcda_dataB 256']

    # Compute mean flow volume (in cm3)
    mcdaflow_mean = df['mcda_actual_flow'].mean()
    mcdaflowvolume_mean = mcdaflow_mean * (1 / 60)

    # Calculate concentration and rename columns
    mcda_dN = dataB_cols / mcdaflowvolume_mean
    mcda_dN.columns = [f"{col}_dN" for col in dataB_cols.columns]

    # Compute total concentration
    mcda_dN_totalconc = mcda_dN.sum(axis=1, skipna=True).to_frame(name='mcda_dN_totalconc')

    # Midpoint diameters
    Midpoint_diameter_list = np.array([
        0.244381, 0.246646, 0.248908, 0.251144, 0.253398, 0.255593,
        0.257846, 0.260141, 0.262561, 0.265062, 0.267712, 0.270370,
        0.273159, 0.275904, 0.278724, 0.281554, 0.284585, 0.287661,
        0.290892, 0.294127, 0.297512, 0.300813, 0.304101, 0.307439,
        0.310919, 0.314493, 0.318336, 0.322265, 0.326283, 0.330307,
        0.334409, 0.338478, 0.342743, 0.347102, 0.351648, 0.356225,
        0.360972, 0.365856, 0.371028, 0.376344, 0.382058, 0.387995,
        0.394223, 0.400632, 0.407341, 0.414345, 0.421740, 0.429371,
        0.437556, 0.446036, 0.454738, 0.463515, 0.472572, 0.481728,
        0.491201, 0.500739, 0.510645, 0.520720, 0.530938, 0.541128,
        0.551563, 0.562058, 0.572951, 0.583736, 0.594907, 0.606101,
        0.617542, 0.628738, 0.640375, 0.652197, 0.664789, 0.677657,
        0.691517, 0.705944, 0.721263, 0.736906, 0.753552, 0.770735,
        0.789397, 0.808690, 0.829510, 0.851216, 0.874296, 0.897757,
        0.922457, 0.948074, 0.975372, 1.003264, 1.033206, 1.064365,
        1.097090, 1.130405, 1.165455, 1.201346, 1.239589, 1.278023,
        1.318937, 1.360743, 1.403723, 1.446000, 1.489565, 1.532676,
        1.577436, 1.621533, 1.667088, 1.712520, 1.758571, 1.802912,
        1.847836, 1.891948, 1.937088, 1.981087, 2.027604, 2.074306,
        2.121821, 2.168489, 2.216644, 2.263724, 2.312591, 2.361099,
        2.412220, 2.464198, 2.518098, 2.571786, 2.628213, 2.685162,
        2.745035, 2.805450, 2.869842, 2.935997, 3.005175, 3.074905,
        3.148598, 3.224051, 3.305016, 3.387588, 3.476382, 3.568195,
        3.664863, 3.761628, 3.863183, 3.965651, 4.072830, 4.179050,
        4.289743, 4.400463, 4.512449, 4.621025, 4.731530, 4.839920,
        4.949855, 5.057777, 5.169742, 5.281416, 5.395039, 5.506828,
        5.621488, 5.734391, 5.849553, 5.962881, 6.081516, 6.200801,
        6.322133, 6.441786, 6.565130, 6.686935, 6.813017, 6.938981,
        7.071558, 7.205968, 7.345185, 7.483423, 7.628105, 7.774385,
        7.926945, 8.080500, 8.247832, 8.419585, 8.598929, 8.780634,
        8.973158, 9.167022, 9.372760, 9.582145, 9.808045, 10.041607,
        10.287848, 10.537226, 10.801172, 11.068405, 11.345135,
        11.621413, 11.910639, 12.200227, 12.492929, 12.780176,
        13.072476, 13.359067, 13.651163, 13.937329, 14.232032,
        14.523919, 14.819204, 15.106612, 15.402110, 15.695489,
        15.998035, 16.297519, 16.610927, 16.926800, 17.250511,
        17.570901, 17.904338, 18.239874, 18.588605, 18.938763,
        19.311505, 19.693678, 20.093464, 20.498208, 20.927653,
        21.366609, 21.827923, 22.297936, 22.802929, 23.325426,
        23.872344, 24.428708, 25.016547, 25.616663, 26.249815,
        26.888493, 27.563838, 28.246317, 28.944507, 29.626186,
        30.323440, 31.005915, 31.691752, 32.353900, 33.030123,
        33.692286, 34.350532, 34.984611, 35.626553, 36.250913,
        36.878655, 37.489663, 38.121550, 38.748073, 39.384594,
        40.008540, 40.654627, 41.292757, 41.937789, 42.578436
    ])

    
    log_midpoints = np.log10(Midpoint_diameter_list)
    log_edges = np.zeros(len(log_midpoints) + 1)
    log_edges[1:-1] = (log_midpoints[:-1] + log_midpoints[1:]) / 2
    log_edges[0] = log_midpoints[0] - (log_midpoints[1] - log_midpoints[0]) / 2
    log_edges[-1] = log_midpoints[-1] + (log_midpoints[-1] - log_midpoints[-2]) / 2
    mcda_dlogDp = np.diff(log_edges)

    # Compute dNdlogDp
    mcda_dNdlogDp = mcda_dN.loc[:, 'mcda_dataB 1_dN':'mcda_dataB 256_dN'].div(mcda_dlogDp).add_suffix('_dlogDp')

    # Insert calculated DataFrames after existing mcda_ columns
    mcda_columns = [col for col in df.columns if col.startswith('mcda_')]
    last_mcda_index = df.columns.get_loc(mcda_columns[-1]) + 1 if mcda_columns else len(df.columns)

    df = pd.concat([
        df.iloc[:, :last_mcda_index],
        mcda_dN,
        mcda_dN_totalconc,
        mcda_dNdlogDp,
        df.iloc[:, last_mcda_index:]
    ], axis=1)

    # Plot mCDA total concentration vs Altitude
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(
        df['mcda_dN_totalconc'], 
        df['Altitude'], 
        label='mCDA total conc', 
        color='salmon', 
        marker='.', 
        linestyle='none'
    )
    ax.grid(ls='--')
    ax.set_ylabel('Altitude (m)', fontsize=12)
    ax.set_xlabel("mCDA total concentration (cm$^{-3}$)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return df

def mCDA_STP_normalization(df):
    """
    Normalize mCDA concentrations to STP conditions and insert the results
    right after the existing mCDA columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing mCDA measurements and metadata.

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

    # Calculate the STP correction factor
    correction_factor = (P_measured / P_STP) * (T_STP / T_measured)

    # List of columns to correct
    columns_to_normalize = [col for col in df.columns if col.endswith('_dN_dlogDp') or col.endswith('_dN')] + ['mcda_dN_totalconc']

    # Create dictionary for normalized columns
    normalized_columns = {}

    for col in columns_to_normalize:
        if col in df.columns:
            normalized_columns[col + '_stp'] = df[col] * correction_factor

    # Add recalculated total concentration from '_dN_stp' columns
    dN_stp_columns = [col for col in normalized_columns if col.endswith('_dN_stp')]
    normalized_columns['mcda_dN_totalconc_stp_recalculated'] = pd.DataFrame(normalized_columns)[dN_stp_columns].sum(axis=1, skipna=True)

    # Find where to insert (after the last mSEMS-related column)
    mcda_columns = [col for col in df.columns if col.startswith('mcda_')]
    if mcda_columns:
        last_mcda_index = df.columns.get_loc(mcda_columns[-1]) + 1
    else:
        last_mcda_index = len(df.columns)
    
    # Insert normalized columns
    df = pd.concat(
        [df.iloc[:, :last_mcda_index],
            pd.DataFrame(normalized_columns, index=df.index),
            df.iloc[:, last_mcda_index:]],
        axis=1
    )
    
    # PLOT
    plt.figure(figsize=(8, 6))
    plt.plot(df['mcda_dN_totalconc'], df['Altitude'], label='Measured', color='blue', marker='.', linestyle='none')
    plt.plot(df['mcda_dN_totalconc_stp_recalculated'], df['Altitude'], label='Recalculated', color='green', marker='.', linestyle='none')
    plt.plot(df['mcda_dN_totalconc_stp'], df['Altitude'], label='STP-normalized', color='red', marker='.', linestyle='none')
    plt.xlabel('mCDA total concentration (cm$^{-3}$)', fontsize=12)
    plt.ylabel('Altitude (m)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df


def plot_mcda_distribution(df, Midpoint_diameter_list, time_start=None, time_end=None):
    """
    Plot mCDA size distribution and total concentration.

    Parameters:
    - df (pd.DataFrame): DataFrame containing mCDA size distribution and total concentration.
    - Midpoint_diameter_list (list or np.array): List of particle diameters corresponding to the diameter bin midpoints.
    """
    plt.close('all')

    # Define the range of columns for concentration
    start_conc = 'mcda_dataB 1_dN_dlogDp_stp'
    end_conc = 'mcda_dataB 256_dN_dlogDp_stp'

    # Extract the relevant concentration data
    counts = df.loc[:, start_conc:end_conc]
    dtimes = df.index

    # Set index and prepare the data
    counts = counts.set_index(dtimes)
    counts = counts.astype(float)
    counts[counts == 0] = np.nan
    vmax_value = np.nanmax(counts.values)
    print(vmax_value)

    # Create 2D mesh grid
    bin_diameters = Midpoint_diameter_list
    xx, yy = np.meshgrid(counts.index.values, bin_diameters)
    Z = counts.values.T

    # Start plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    norm = mcols.LogNorm(vmin=1, vmax=100)
    mesh = ax.pcolormesh(xx, yy, Z, cmap='viridis', norm=norm, shading="gouraud")

    # Add colorbar
    cb = fig.colorbar(mesh, ax=ax, orientation='vertical', location='right', pad=0.02)
    cb.set_label('dN/dlogD$_p$ (cm$^{-3}$)', fontsize=12, fontweight='bold')
    cb.ax.tick_params(labelsize=12)

    # Custom x-axis formatter
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

    # Apply formatter
    custom_formatter = CustomDateFormatter(fmt="%H:%M", date_fmt="%Y-%m-%d %H:%M")
    ax.xaxis.set_major_formatter(custom_formatter)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.tick_params(axis='x', rotation=90, labelsize=12)

    # Set axis properties
    if time_start and time_end:
        ax.set_xlim(pd.Timestamp(time_start), pd.Timestamp(time_end))
    ax.set_ylim(0.4, 20)
    ax.set_yscale('log')
    ax.set_ylabel('Particle Diameter (um)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('mCDA size distribution and total concentration', fontsize=13, fontweight='bold')

    # Plot total concentration
    total_conc = df['mcda_dN_totalconc_stp']
    ax2 = ax.twinx()
    ax2.plot(total_conc.index, total_conc, color='red', linewidth=2)
    ax2.set_ylabel('mCDA total conc (cm$^{-3}$)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelsize=12, colors='red')
    ax2.set_ylim(0, total_conc.max() * 2)
    ax2.set_xlim(ax.get_xlim())  # Synchronize x-axis

    plt.show()


def plot_mcda_vertical_distribution(df, Midpoint_diameter_list):
    """
    Plots the vertical distribution of mCDA particle size distribution versus altitude.

    Parameters:
    - df: pandas DataFrame containing particle size distribution and altitude data.
    - midpoint_diameter_list: 1D NumPy array of midpoint diameters corresponding to bins.
    """
    plt.close('all')

    # Define the range of columns for which you want the concentration
    start_conc = 'mcda_dataB 1_dN_dlogDp_stp'
    end_conc = 'mcda_dataB 256_dN_dlogDp_stp'

    # Extract the relevant columns
    counts = df.loc[:, start_conc:end_conc]

    # Extract altitude data
    altitude = df['Altitude']
    counts = counts.set_index(altitude)

    # Ensure float and replace zeros with NaN
    counts = counts.astype(float)
    counts[counts == 0] = np.nan

    # Create 2D grid from altitude and bin diameters (reversed)
    bin_diameters = Midpoint_diameter_list
    yy, xx = np.meshgrid(counts.index.values, bin_diameters)
    Z = counts.values.T  # Shape must be (nrows = bins, ncols = altitude steps)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    norm = mcols.LogNorm(vmin=1, vmax=100)

    mesh = ax.pcolormesh(xx, yy, Z, cmap='viridis', norm=norm, shading="gouraud")

    # Add colorbar
    cb = fig.colorbar(mesh, ax=ax, orientation='vertical', location='right', pad=0.02)
    cb.set_label('dN/dlogD$_p$ (cm$^{-3}$)', fontsize=12, fontweight='bold')
    cb.ax.tick_params(labelsize=12)

    # Axis settings
    ax.set_xlim(0.4, 20)
    ax.set_xscale('log')
    ax.set_xlabel('Particle Diameter (µm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Altitude (m)', fontsize=12, fontweight='bold', labelpad=10)

    # Plot secondary y-axis (if necessary, e.g., for concentration)
    # If you still want to include total concentration on the secondary y-axis:
    # total_conc = df_copy['mcda_dN_totalconc_stp']
    # ax2 = ax.twinx()
    # ax2.plot(total_conc.index, total_conc, color='red', linewidth=2)
    # ax2.set_ylabel('mCDA Total Conc (cm$^{-3}$)', fontsize=12, fontweight='bold', color='red')
    # ax2.tick_params(axis='y', labelsize=12, colors='red')
    # ax2.set_ylim(0, total_conc.max() * 2)

    plt.show()

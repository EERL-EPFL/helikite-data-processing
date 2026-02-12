"""
Mini Cloud Droplet Anlayzer (mCDA)
mCDA -> mCDA_output_2025-02-12_A (has pressure)

Important variables to keep:

!!! Raspery Pie of the mCDA has an autonomous timestamp, make sur it has been corrected when creating the output file.

"""
import datetime
import logging

import matplotlib.colors as mcolors
import matplotlib.colors as mcols
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from helikite.constants import constants
from helikite.instruments.base import Instrument, filter_columns_by_instrument

# Define logger
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


# Midpoint diameters
MCDA_MIDPOINT_DIAMETER_LIST = np.array([
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


class mCDA(Instrument):
    """
    Instrument definition for the mcda sensor system.
    Handles timestamp creation and optional corrections.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def __repr__(self):
        return "mCDA"

    @property
    def has_size_distribution(self) -> bool:
        return True

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
        header_columns = first_lines_of_csv[self.header].strip().split(',')
        
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
        df.index = df.index.floor('s').astype("datetime64[s]")

        return df

    def read_data(self) -> pd.DataFrame:
        try:
            # First read everything as string to avoid crashing on weird values like '000A'
            df = pd.read_csv(
                self.filename,
                dtype=str,
                na_values=self.na_values,
                skiprows=self.header,
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

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'measurement_nbr' in df.columns:
            # Compare current value with previous value to detect changes in measurement_nbr
            is_change_mcd = df['measurement_nbr'] != df['measurement_nbr'].shift(1)
            # List of target columns where you want to nullify repetitive data
            target_columns_mcda = [
                'Temperature', 'Pressure', 'RH', 'pmav', 'offset1', 'offset2',
                'calib1', 'calib2', 'measurement_nbr', 'pressure'
            ]
            # Nullify repeated rows (set to NaN) where there's no change
            df.loc[~is_change_mcd, target_columns_mcda] = np.nan

        else:
            print(f"No 'measurement_nbr' column found in {self.name}.")

        return df

    def calculate_derived(self, df: pd.DataFrame, verbose: bool, *args, **kwargs) -> pd.DataFrame:
        # Select columns from 'mcda_dataB 1' to 'mcda_dataB 256'
        dataB_cols = df.loc[:, 'mcda_dataB 1':'mcda_dataB 256']

        # Compute mean flow volume (in cm3)
        mcdaflow_mean = df['mcda_actual_flow'].mean()
        mcdaflowvolume_mean = mcdaflow_mean * (1 / 60)

        # Calculate concentration and rename columns
        mcda_dN = dataB_cols / mcdaflowvolume_mean
        mcda_dN.columns = [f"{col}_dN" for col in dataB_cols.columns]

        # Compute total concentration
        mcda_dN_totalconc = mcda_dN.sum(axis=1, skipna=True, min_count=1).to_frame(name='mcda_dN_totalconc')

        log_midpoints = np.log10(MCDA_MIDPOINT_DIAMETER_LIST)
        log_edges = np.zeros(len(log_midpoints) + 1)
        log_edges[1:-1] = (log_midpoints[:-1] + log_midpoints[1:]) / 2
        log_edges[0] = log_midpoints[0] - (log_midpoints[1] - log_midpoints[0]) / 2
        log_edges[-1] = log_midpoints[-1] + (log_midpoints[-1] - log_midpoints[-2]) / 2
        mcda_dlogDp = np.diff(log_edges)

        # Compute dNdlogDp
        mcda_dNdlogDp = mcda_dN.loc[:, 'mcda_dataB 1_dN':'mcda_dataB 256_dN'].div(mcda_dlogDp).add_suffix('_dlogDp')

        # Insert calculated DataFrames after existing mcda_ columns
        mcda_columns = filter_columns_by_instrument(df.columns, mcda)
        last_mcda_index = df.columns.get_loc(mcda_columns[-1]) + 1 if mcda_columns else len(df.columns)

        df = pd.concat([
            df.iloc[:, :last_mcda_index],
            mcda_dN,
            mcda_dN_totalconc,
            mcda_dNdlogDp,
            df.iloc[:, last_mcda_index:]
        ], axis=1)

        return df

    def normalize(self, df: pd.DataFrame, reference_instrument: Instrument,
                  verbose: bool, *args, **kwargs) -> pd.DataFrame:
        """
        Normalize mCDA concentrations to STP conditions and insert the results
        right after the existing mCDA columns.

        Parameters:
        df (pd.DataFrame): DataFrame containing mCDA measurements and metadata.

        Returns:
        df (pd.DataFrame): Updated DataFrame with STP-normalized columns inserted.
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
        columns_to_normalize = [col for col in df.columns if col.endswith('_dN_dlogDp') or col.endswith('_dN')] + [
            'mcda_dN_totalconc']

        # Create dictionary for normalized columns
        normalized_columns = {}

        for col in columns_to_normalize:
            if col in df.columns:
                normalized_columns[col + '_stp'] = df[col] * correction_factor

        # Add recalculated total concentration from '_dN_stp' columns
        dN_stp_columns = [col for col in normalized_columns if col.endswith('_dN_stp')]
        normalized_columns['mcda_dN_totalconc_stp_recalculated'] = (
            pd.DataFrame(normalized_columns)[dN_stp_columns].sum(axis=1, skipna=True, min_count=1)
        )

        # Find where to insert (after the last mSEMS-related column)
        mcda_columns = filter_columns_by_instrument(df.columns, mcda)
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

        return df

    def plot_raw_and_normalized(self, df: pd.DataFrame, verbose: bool, *args, **kwargs):
        """Plots mCDA total concentration vs Altitude"""
        plt.figure(figsize=(8, 6))
        plt.plot(df['mcda_dN_totalconc'], df['Altitude'], label='Measured', color='blue', marker='.', linestyle='none')
        if 'mcda_dN_totalconc_stp' in df.columns:
            plt.plot(df['mcda_dN_totalconc_stp_recalculated'], df['Altitude'], label='Recalculated', color='green',
                     marker='.', linestyle='none')
            plt.plot(df['mcda_dN_totalconc_stp'], df['Altitude'], label='STP-normalized', color='red', marker='.',
                     linestyle='none')
        plt.xlabel('mCDA total concentration (cm$^{-3}$)', fontsize=12)
        plt.ylabel('Altitude (m)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_distribution(self, df: pd.DataFrame, verbose: bool,
                          time_start: datetime.datetime, time_end: datetime.datetime,
                          subplot: tuple[plt.Figure, plt.Axes] | None = None):
        """
        Plot mCDA size distribution and total concentration.

        Parameters:
        - df (pd.DataFrame): DataFrame containing mCDA size distribution and total concentration.
        - Midpoint_diameter_list (list or np.array): List of particle diameters corresponding to the diameter bin midpoints.
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

        # Define the range of columns for concentration
        start_conc = self.column_name(df, 'mcda_dataB 1_dN_dlogDp_stp')
        end_conc = self.column_name(df, 'mcda_dataB 256_dN_dlogDp_stp')

        # Extract the relevant concentration data
        counts = df.loc[:, start_conc:end_conc]
        counts = counts.set_index(df.index)
        counts = counts.astype(float)
        counts[counts == 0] = np.nan
        counts = counts.dropna(how='all') if not counts.isna().all().all() else counts

        vmax_value = np.nanmax(counts.values) if not counts.isna().all().all() else np.nan
        print(f"max value ({self.name}): {vmax_value}")

        # Create 2D mesh grid
        xx, yy = np.meshgrid(counts.index.values, MCDA_MIDPOINT_DIAMETER_LIST)
        Z = counts.values.T

        # Start plotting
        norm = mcolors.LogNorm(vmin=1, vmax=50)
        mesh = ax.pcolormesh(xx, yy, Z, cmap='viridis', norm=norm, shading="gouraud")

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = inset_axes(ax, width="1.5%", height="100%", loc='lower left',
                         bbox_to_anchor=(1.08, -0.025, 1, 1), bbox_transform=ax.transAxes)
        cb = fig.colorbar(mesh, cax=cax, orientation='vertical')
        cb.set_label('dN/dlogD$_p$ (cm$^{-3}$)', fontsize=12, fontweight='bold')
        cb.ax.tick_params(labelsize=11)

        if not is_custom_subplot:
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
            ax.tick_params(axis='x', rotation=90, labelsize=11)

        # Set axis properties
        ax.set_ylim(0.4, 20)
        ax.set_yscale('log')
        ax.set_ylabel('Part. Diameter (μm)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, linestyle='--', linewidth=0.5, axis='x')
        ax.grid(False, axis='y')
        if not is_custom_subplot:
            ax.set_xlabel('Time', fontsize=12, fontweight='bold', labelpad=10)
            ax.set_title('mCDA size distribution and total concentration', fontsize=13, fontweight='bold')

        # Plot total concentration
        total_conc = df[self.column_name(df, 'mcda_dN_totalconc_stp')]
        total_conc_max = total_conc.max() if not total_conc.isna().all() else 15
        ax2 = ax.twinx()
        ax2.plot(df.index, total_conc.ffill(limit=1), color='red', linewidth=2)
        ax2.set_ylabel('mCDA conc (cm$^{-3}$)', fontsize=12, fontweight='bold', color='red', labelpad=15)
        ax2.tick_params(axis='y', labelsize=11, colors='red')
        ax2.set_ylim(0, total_conc_max * 2)

        if not is_custom_subplot:
            plt.show()

    def plot_vertical_distribution(self, df: pd.DataFrame, verbose: bool, *args, **kwargs):
        """
        Plots the vertical distribution of mCDA particle size distribution versus altitude.

        Parameters:
        - df: pandas DataFrame containing particle size distribution and altitude data.
        - midpoint_diameter_list: 1D NumPy array of midpoint diameters corresponding to bins.
        """

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
        counts = counts.dropna(how='all') if not counts.isna().all().all() else counts

        # Create 2D grid from altitude and bin diameters (reversed)
        yy, xx = np.meshgrid(counts.index.values, MCDA_MIDPOINT_DIAMETER_LIST)
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
        # total_conc_max = total_conc.max() if not total_conc.isna().all() else 15
        # ax2 = ax.twinx()
        # ax2.plot(total_conc.index, total_conc, color='red', linewidth=2)
        # ax2.set_ylabel('mCDA Total Conc (cm$^{-3}$)', fontsize=12, fontweight='bold', color='red')
        # ax2.tick_params(axis='y', labelsize=12, colors='red')
        # ax2.set_ylim(0, total_conc_max * 2)

        plt.show()

mcda = mCDA(
    name="mcda",
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
    cols_final=[f"dataB {i}_dN_dlogDp_stp" for i in range(1, 257)] + ["dN_totalconc_stp"],
    export_order=730,
    pressure_variable="Pressure",
    temperature_variable="Temperature",
    coupled_columns=[
        [f"mcda_dataB {i}" for i in range(1, 513)] +
        [f"mcda_dataB {i}_dN" for i in range(1, 257)] +
        [f"mcda_dataB {i}_dN_dlogDp" for i in range(1, 257)] +
        [f"mcda_dN_totalconc"],
    ],
    rename_dict={f'mcda_dataB {i}_dN_dlogDp_stp': f'mCDA_dataB{i}' for i in range(1, 257)} |
                {'mcda_dN_totalconc_stp': 'mCDA_total_N'},
)
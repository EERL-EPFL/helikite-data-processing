import logging
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

import folium
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import VBox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pydantic import BaseModel

from helikite.classes.base import BaseProcessor, function_dependencies
from helikite.constants import constants
from helikite.instruments import Instrument, flight_computer_v1, flight_computer_v2, smart_tether, pops, msems_readings, \
    msems_inverted, msems_scan, mcda, filter, tapir, cpc, stap, stap_raw, co2
from helikite.instruments.co2 import process_CO2_STP
from helikite.instruments.cpc3007 import CPC
from helikite.instruments.flight_computer import FlightComputer
from helikite.instruments.mcda_instrument import mcda_concentration_calculations, mCDA_STP_normalization, \
    plot_mcda_distribution, plot_mcda_vertical_distribution
from helikite.instruments.msems import mSEMS_total_conc_dN, mSEMS_STP_normalization
from helikite.instruments.pops import POPS_total_conc_dNdlogDp, POPS_STP_normalization
from helikite.instruments.stap import STAP_STP_normalization
from helikite.processing import choose_outliers
from helikite.processing.post.TandRH import T_RH_averaging
from helikite.processing.post.altitude import altitude_calculation_barometric
from helikite.processing.post.level1 import flight_profiles_1
from helikite.processing.post.outliers import plot_outliers_check, plot_gps_on_map

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


@dataclass(frozen=True)
class OutputSchema:
    instruments: list[Instrument]
    """List of instruments whose columns should be present in the output dataframe."""
    colors: dict[Instrument, str]
    """Instrument-to-color dictionary for the consistent across a campaign plotting"""
    reference_instrument_candidates: list[Instrument]
    """Reference instrument candidates for the automatic instruments detection"""


# TODO: implement other schemas used
class OutputSchemas(OutputSchema, Enum):
    ORACLES = OutputSchema(
        instruments=[
            flight_computer_v2,
            smart_tether,
            pops,
            msems_readings,
            msems_inverted,
            msems_scan,
            mcda,
            filter,
            tapir,
            cpc,
        ],
        colors={
            flight_computer_v2: "C0",
            smart_tether: "C1",
            pops: "C2",
            msems_readings: "C3",
            msems_inverted: "C6",
            msems_scan: "C5",
            mcda: "C4",
            filter: "C7",
            tapir: "C8",
            cpc: "C9",
        },
        reference_instrument_candidates=[flight_computer_v2, smart_tether, pops]
    )

    TURTMANN = OutputSchema(
        instruments=[
            flight_computer_v1,
            smart_tether,
            pops,
            msems_readings,
            msems_inverted,
            msems_scan,
            stap,
            stap_raw,
            co2,
            filter,
        ],
        colors={
            flight_computer_v1: "C0",
            smart_tether: "C1",
            pops: "C2",
            msems_readings: "C3",
            msems_inverted: "C6",
            msems_scan: "C5",
            stap: "C4",
            stap_raw: "C8",
            co2: "C9",
            filter: "C7",
        },
        reference_instrument_candidates=[flight_computer_v2, smart_tether, pops]
    )


class DataProcessorLevel1(BaseProcessor):
    def __init__(self, output_schema: OutputSchema, df: pd.DataFrame, metadata: BaseModel) -> None:
        instruments, reference_instrument = _get_instruments(df, metadata)
        super().__init__(output_schema, instruments, reference_instrument)
        self._df = df.copy()
        self._metadata = metadata
        self._outlier_files: set[str] = set()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def instruments(self) -> list[Instrument]:
        return self._instruments

    @property
    def coupled_columns(self) -> list[tuple[str, ...]] | None:
        coupled_columns = []
        for instrument in self._instruments:
            coupled_columns += instrument.coupled_columns
        return coupled_columns

    def _data_state_info(self) -> list[str]:
        state_info = []

        for outlier_file in self._outlier_files:

            outliers = pd.read_csv(outlier_file, index_col=0, parse_dates=True)
            columns_with_outliers = outliers.columns[outliers.any()]

            state_info.append(f"{'Outliers file':<40}{outlier_file}")
            state_info.append(
                f"{'Variable':<40}{'Number of outliers':<20}"
            )
            state_info.append("-" * 60)

            for column in columns_with_outliers:
                state_info.append(
                    f"{column:<40}{outliers[column].sum():<20}"
                )

        return state_info

    @function_dependencies(required_operations=[], use_once=False)
    def choose_outliers(
        self,
        y: str,
        outlier_file: str = "outliers.csv",
        use_coupled_columns: bool = True,
    ) -> VBox:
        """Creates a plot to interactively select outliers in the data.

        A plot is generated where two variables are plotted, and the user can
        click on points to select or deselect them as outliers, or use Plotly's
        selection tools to select multiple points at once.
        If `use_coupled_columns` is True and a value in any column within a group of coupled columns
        is marked as an outlier, then the values in all other columns of that group will also be marked as outliers.

        Parameters
        ----------
            df (pandas.DataFrame): The dataframe containing the data
            y (str): The column name of the y-axis variable
            outlier_file (str): The path to the CSV file to store the outliers
            use_coupled_columns (bool): if True, use the coupled columns defined in the instruments
        """
        coupled_columns = self.coupled_columns if use_coupled_columns else []
        vbox = choose_outliers(self._df, y, coupled_columns, outlier_file)
        self._outlier_files.add(outlier_file)

        return vbox

    @function_dependencies(required_operations=[], use_once=False)
    def fillna_if_all_missing(self, values_dict: dict[str, Any]):
        for column, value in values_dict.items():
            if column not in self._df.columns:
                logger.warning(f"Column '{column}' not found in dataframe. Skipping fill.")
                continue
            if self._df[column].isna().all():
                print(f"'{column}' is all missing. Setting value to: {value}")
                self._df[column] = value
            else:
                print(f"Column '{column}' has data present. Skipping fill.")

    @function_dependencies(required_operations=["choose_outliers"], use_once=False)
    def set_outliers_to_nan(self):
        for outlier_file in self._outlier_files:
            outliers = pd.read_csv(outlier_file, index_col=0, parse_dates=True)

            self._df.loc[outliers.index] = self._df.loc[outliers.index].mask(outliers)

    @function_dependencies(required_operations=["set_outliers_to_nan"], use_once=False)
    def plot_outliers_check(self):
        plot_outliers_check(self._df, self._flight_computer)

    @function_dependencies(required_operations=[], use_once=False)
    def plot_gps_on_map(self, lat_col='flight_computer_Lat', lon_col='flight_computer_Long',
                        lat_dir='S', lon_dir='W',
                        center_coords=(-70.6587, -8.2850), zoom_start=13) -> folium.Map:
        return plot_gps_on_map(self._df, lat_col, lon_col, lat_dir, lon_dir, center_coords, zoom_start)

    @function_dependencies(required_operations=["set_outliers_to_nan"], use_once=False)
    def T_RH_averaging(self):
        self._df = T_RH_averaging(self._df, self._flight_computer)

    @function_dependencies(required_operations=["T_RH_averaging"], use_once=False)
    def altitude_calculation_barometric(self):
        self._df = altitude_calculation_barometric(self._df, self._metadata)

    @function_dependencies(required_operations=[], use_once=True)
    def add_missing_columns(self):
        for instrument in self._output_schema.instruments:
            is_reference = instrument == self._reference_instrument

            all_columns = instrument.get_expected_columns(level=0, is_reference=is_reference)
            missing_columns = [column for column in all_columns if column not in self._df.columns]

            if len(missing_columns) != 0:
                if instrument in self._instruments:
                    logger.warning(f"{instrument} is present but missing columns: {missing_columns}.")
                logger.info(f"Adding missing columns for {instrument.name}: {missing_columns}")

            self._df[missing_columns] = pd.NA

    @function_dependencies(required_operations=["altitude_calculation_barometric", "add_missing_columns"],
                           use_once=False)
    def process_CO2_STP(self, min_threshold=200, max_threshold=500):
        if self._check_schema_contains_instrument(co2):
            self._df = process_CO2_STP(self._df, min_threshold, max_threshold)

    @function_dependencies(required_operations=["altitude_calculation_barometric", "add_missing_columns"],
                           use_once=False)
    def STAP_STP_normalization(self):
        if self._check_schema_contains_instrument(stap):
            self._df = STAP_STP_normalization(self._df)

    @function_dependencies(required_operations=["altitude_calculation_barometric", "add_missing_columns"],
                           use_once=False)
    def POPS_total_conc_dNdlogDp(self):
        if self._check_schema_contains_instrument(pops):
            self._df = POPS_total_conc_dNdlogDp(self._df)

    @function_dependencies(required_operations=["POPS_total_conc_dNdlogDp"], use_once=False)
    def POPS_STP_normalization(self):
        if self._check_schema_contains_instrument(pops):
            self._df = POPS_STP_normalization(self._df)

    @function_dependencies(required_operations=["altitude_calculation_barometric", "add_missing_columns"],
                           use_once=False)
    def mSEMS_total_conc_dN(self):
        if self._check_schema_contains_instrument(msems_inverted):
            self._df = mSEMS_total_conc_dN(self._df)

    @function_dependencies(required_operations=["mSEMS_total_conc_dN"], use_once=False)
    def mSEMS_STP_normalization(self):
        if self._check_schema_contains_instrument(msems_inverted):
            self._df = mSEMS_STP_normalization(self._df)

    @function_dependencies(required_operations=["altitude_calculation_barometric", "add_missing_columns"],
                           use_once=False)
    def mcda_concentration_calculations(self):
        if self._check_schema_contains_instrument(mcda):
            self._df = mcda_concentration_calculations(self._df)

    @function_dependencies(required_operations=["mcda_concentration_calculations"], use_once=False)
    def mCDA_STP_normalization(self):
        if self._check_schema_contains_instrument(mcda):
            self._df = mCDA_STP_normalization(self._df)

    @function_dependencies(required_operations=["mCDA_STP_normalization"], use_once=False)
    def plot_mcda_distribution(self, midpoint_diameter_list, time_start=None, time_end=None):
        if self._check_schema_contains_instrument(mcda):
            plot_mcda_distribution(self._df, midpoint_diameter_list, time_start, time_end)

    @function_dependencies(required_operations=["mCDA_STP_normalization"], use_once=False)
    def plot_mcda_vertical_distribution(self, Midpoint_diameter_list):
        if self._check_schema_contains_instrument(mcda):
            plot_mcda_vertical_distribution(self._df, Midpoint_diameter_list)

    @function_dependencies(required_operations=["altitude_calculation_barometric", "add_missing_columns"],
                           use_once=False)
    def CPC_STP_normalization(self):
        if self._check_schema_contains_instrument(cpc):
            self._df = CPC.CPC_STP_normalization(self._df)

    def plot_flight_profiles(self, flight_basename: str, save_path: str | pathlib.Path):
        # Limits for x-axis (T, RH, mSEMS, CPC, POPS, mCDA, WS, WD)
        custom_xlim = {
            'ax1': (-6, 2),
            'ax2': (60, 100),
            'ax3': (0, 1200),
            'ax4': (0, 1200),
            'ax5': (0, 60),
            'ax6': (0, 60),
            'ax7': (0, 12)
        }

        custom_xticks = {
            'ax1': np.arange(-6, 3, 2),
            'ax2': np.arange(60, 101, 10),
            'ax3': np.arange(0, 1201, 200),
            'ax4': np.arange(0, 1201, 200),
            'ax5': np.arange(0, 61, 10),
            'ax6': np.arange(0, 61, 10),
            'ax7': np.arange(0, 13, 3)
        }

        # Plot title
        custom_title = f'Flight {self._metadata.flight} ({flight_basename}) [Level 1]'

        fig = flight_profiles_1(self._df, self._metadata,
                                xlims=custom_xlim, xticks=custom_xticks, fig_title=custom_title)

        # Save the figure after plotting
        print("Saving figure to:", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')


    def plot_size_distr(self, save_path: str | pathlib.Path):
        # TEMPORAL PLOT OF FLIGHT with POPS and mSEMS HEAT MAPS

        # Create figure with 3 subplots, sharing the same x-axis
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
        plt.subplots_adjust(hspace=0.1)

        """ SET THE TITLE OF THE PLOT (FLIGHT N° with DATE_X) """
        # 'i' will automatically be replaced by the set flight number
        # '_X' has to be changed manually in function of the flight index of the day (A, B, ...)
        fig.suptitle(f'Flight {self._metadata.flight} ({self._metadata.flight_date}_A) [Level 1]', fontsize=16,
                     fontweight='bold',
                     y=0.91)

        ### SUBPLOT 1: Altitude vs. Time
        ax1 = axes[0]
        ax1.plot(self._df.index, self._df['Altitude'], color='black', linewidth=2, label='Altitude')

        ax1.set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelsize=11)
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.set_ylim(-40, self._df['Altitude'].max() * 1.04)

        ### SUBPLOT 2: mSEMS heatmmap & total concentration
        ax2 = axes[1]

        # Get diameter bin averages
        start_dia = 'msems_inverted_Bin_Dia1'
        end_dia = 'msems_inverted_Bin_Dia60'
        bin_diameter_averages = self._df.loc[:, start_dia:end_dia].mean()

        # Get concentration data
        start_conc = 'msems_inverted_Bin_Conc1_stp'
        end_conc = 'msems_inverted_Bin_Conc60_stp'
        counts = self._df.loc[:, start_conc:end_conc]
        counts.index = self._df.index
        counts = counts.astype(float).dropna(how='any')
        counts = counts.clip(lower=1)

        # Create 2D grid
        xx, yy = np.meshgrid(counts.index.values, bin_diameter_averages)

        # Contour plot
        norm = mcolors.LogNorm(vmin=1, vmax=1000)
        mesh = ax2.pcolormesh(xx, yy, counts.values.T, cmap='viridis', norm=norm, shading="gouraud")

        # Colorbar
        divider = make_axes_locatable(ax2)
        cax = inset_axes(ax2, width="1.5%", height="100%", loc='lower left',
                         bbox_to_anchor=(1.08, -0.025, 1, 1), bbox_transform=ax2.transAxes)
        cb = fig.colorbar(mesh, cax=cax, orientation='vertical')
        cb.set_label('dN/dlogD$_p$ (cm$^{-3}$)', fontsize=13, fontweight='bold')
        cb.ax.tick_params(labelsize=11)

        # Add Secondary Y-axis for Total Concentration
        ax2_right = ax2.twinx()
        total_conc = self._df['msems_inverted_dN_totalconc_stp']
        ax2_right.scatter(self._df.index, total_conc, color='red', marker='.')
        ax2_right.set_ylabel('mSEMS conc (cm$^{-3}$)', fontsize=12, fontweight='bold', color='red', labelpad=8)
        ax2_right.tick_params(axis='y', labelsize=11, colors='red')
        # ax2_right.set_ylim(0, total_conc.max() * 1.1)

        # Labels and limits
        ax2.set_yscale('log')
        ax2.set_ylabel('Part. Diameter (nm)', fontsize=12, fontweight='bold')
        ax2.set_ylim(8, 236)
        ax2.grid(True, linestyle='--', alpha=0.6, axis='x')

        ### SUBPLOT 3: POPS heatmap & total concentration
        ax3 = axes[2]

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
        start_conc = 'pops_b3_dlogDp_stp'
        end_conc = 'pops_b15_dlogDp_stp'

        # Get POPS concentration data
        pops_counts = self._df.loc[:, start_conc:end_conc]
        pops_counts = pops_counts.set_index(self._df.index).astype(float)

        # Create 2D grid
        # pops_dia = np.logspace(np.log10(180), np.log10(3370), num=pops_counts.shape[1])
        bin_diameters = pops_dia[3:16]
        xx, yy = np.meshgrid(pops_counts.index.values, bin_diameters)

        # Heatmap
        norm = mcolors.LogNorm(vmin=1, vmax=300)
        mesh = ax3.pcolormesh(xx, yy, pops_counts.values.T, cmap='viridis', norm=norm, shading="gouraud")

        # Colorbar
        divider = make_axes_locatable(ax3)
        cax = inset_axes(ax3, width="1.5%", height="100%", loc='lower left',
                         bbox_to_anchor=(1.08, -0.025, 1, 1), bbox_transform=ax3.transAxes)
        cb = fig.colorbar(mesh, cax=cax, orientation='vertical')
        cb.set_label('dN/dlogD$_p$ (cm$^{-3}$)', fontsize=12, fontweight='bold')
        cb.ax.tick_params(labelsize=11)

        # Labels and grid
        ax3.set_yscale('log')
        ax3.set_ylabel('Part. Diameter (nm)', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='y', labelsize=11)
        ax3.grid(True, linestyle='--', linewidth=0.5, axis='x')
        ax3.grid(False, axis='y')
        ax3.set_ylim(180, 3370)

        # Add Secondary Y-axis for Total POPS Concentration
        ax3_right = ax3.twinx()
        ax3_right.plot(self._df.index, self._df['pops_total_conc_stp'], color='red', linewidth=2,
                       label='Total POPS Conc.')
        ax3_right.set_ylabel('POPS conc (cm$^{-3}$)', fontsize=12, fontweight='bold', color='red', labelpad=8)
        ax3_right.tick_params(axis='y', labelsize=11, colors='red')
        ax3_right.spines['right'].set_color('red')
        ax3_right.set_ylim(-20, self._df['pops_total_conc_stp'].max() * 1.1)

        ### Subplot 4: mCDA heatmap & total concentration
        ax4 = axes[3]

        # Midpoint diameters
        Midpoint_diameter_list = np.array([
            0.244381, 0.246646, 0.248908, 0.251144, 0.253398, 0.255593, 0.257846, 0.260141, 0.262561, 0.265062,
            0.267712, 0.270370, 0.273159, 0.275904, 0.278724, 0.281554, 0.284585, 0.287661, 0.290892, 0.294127,
            0.297512, 0.300813, 0.304101, 0.307439,
            0.310919, 0.314493, 0.318336, 0.322265, 0.326283, 0.330307, 0.334409, 0.338478, 0.342743, 0.347102,
            0.351648, 0.356225, 0.360972, 0.365856, 0.371028, 0.376344, 0.382058, 0.387995, 0.394223, 0.400632,
            0.407341, 0.414345, 0.421740, 0.429371,
            0.437556, 0.446036, 0.454738, 0.463515, 0.472572, 0.481728, 0.491201, 0.500739, 0.510645, 0.520720,
            0.530938, 0.541128, 0.551563, 0.562058, 0.572951, 0.583736, 0.594907, 0.606101, 0.617542, 0.628738,
            0.640375, 0.652197, 0.664789, 0.677657,
            0.691517, 0.705944, 0.721263, 0.736906, 0.753552, 0.770735, 0.789397, 0.808690, 0.829510, 0.851216,
            0.874296, 0.897757, 0.922457, 0.948074, 0.975372, 1.003264, 1.033206, 1.064365, 1.097090, 1.130405,
            1.165455, 1.201346, 1.239589, 1.278023,
            1.318937, 1.360743, 1.403723, 1.446000, 1.489565, 1.532676, 1.577436, 1.621533, 1.667088, 1.712520,
            1.758571, 1.802912, 1.847836, 1.891948, 1.937088, 1.981087, 2.027604, 2.074306, 2.121821, 2.168489,
            2.216644, 2.263724, 2.312591, 2.361099,
            2.412220, 2.464198, 2.518098, 2.571786, 2.628213, 2.685162, 2.745035, 2.805450, 2.869842, 2.935997,
            3.005175, 3.074905, 3.148598, 3.224051, 3.305016, 3.387588, 3.476382, 3.568195, 3.664863, 3.761628,
            3.863183, 3.965651, 4.072830, 4.179050,
            4.289743, 4.400463, 4.512449, 4.621025, 4.731530, 4.839920, 4.949855, 5.057777, 5.169742, 5.281416,
            5.395039, 5.506828, 5.621488, 5.734391, 5.849553, 5.962881, 6.081516, 6.200801, 6.322133, 6.441786,
            6.565130, 6.686935, 6.813017, 6.938981,
            7.071558, 7.205968, 7.345185, 7.483423, 7.628105, 7.774385, 7.926945, 8.080500, 8.247832, 8.419585,
            8.598929, 8.780634, 8.973158, 9.167022, 9.372760, 9.582145, 9.808045, 10.041607, 10.287848, 10.537226,
            10.801172, 11.068405, 11.345135,
            11.621413, 11.910639, 12.200227, 12.492929, 12.780176, 13.072476, 13.359067, 13.651163, 13.937329,
            14.232032, 14.523919, 14.819204, 15.106612, 15.402110, 15.695489, 15.998035, 16.297519, 16.610927,
            16.926800, 17.250511,
            17.570901, 17.904338, 18.239874, 18.588605, 18.938763, 19.311505, 19.693678, 20.093464, 20.498208,
            20.927653, 21.366609, 21.827923, 22.297936, 22.802929, 23.325426, 23.872344, 24.428708, 25.016547,
            25.616663, 26.249815,
            26.888493, 27.563838, 28.246317, 28.944507, 29.626186, 30.323440, 31.005915, 31.691752, 32.353900,
            33.030123, 33.692286, 34.350532, 34.984611, 35.626553, 36.250913, 36.878655, 37.489663, 38.121550,
            38.748073, 39.384594,
            40.008540, 40.654627, 41.292757, 41.937789, 42.578436
        ])

        # Prepare data
        counts = self._df.loc[:, 'mcda_dataB 1_dN_dlogDp_stp':'mcda_dataB 256_dN_dlogDp_stp']
        counts = counts.set_index(self._df.index)
        counts = counts.astype(float)
        counts[counts == 0] = np.nan

        bin_diameters = Midpoint_diameter_list
        xx, yy = np.meshgrid(counts.index.values, bin_diameters)
        Z = counts.values.T

        # Plot heatmap
        norm = mcolors.LogNorm(vmin=1, vmax=50)
        mesh = ax4.pcolormesh(xx, yy, Z, cmap='viridis', norm=norm, shading="gouraud")

        # Colorbar
        divider = make_axes_locatable(ax4)
        cax = inset_axes(ax4, width="1.5%", height="100%", loc='lower left',
                         bbox_to_anchor=(1.08, -0.025, 1, 1), bbox_transform=ax4.transAxes)
        cb = fig.colorbar(mesh, cax=cax, orientation='vertical')
        cb.set_label('dN/dlogD$_p$ (cm$^{-3}$)', fontsize=12, fontweight='bold')
        cb.ax.tick_params(labelsize=11)

        # Total concentration
        ax4_right = ax4.twinx()
        total_conc = self._df['mcda_dN_totalconc_stp']
        ax4_right.plot(self._df.index, total_conc, color='red', linewidth=2)
        ax4_right.set_ylabel('mCDA conc (cm$^{-3}$)', fontsize=12, fontweight='bold', color='red', labelpad=15)
        ax4_right.tick_params(axis='y', labelsize=11, colors='red')
        # ax4_right.set_ylim(0, total_conc.max() * 2)
        # ax4_right.set_xlim(ax4.get_xlim())

        # Axis styling
        ax4.set_yscale('log')
        ax4.set_ylabel('Part. Diameter (μm)', fontsize=12, fontweight='bold')
        ax4.set_ylim(0.4, 20)
        ax4.grid(True, linestyle='--', linewidth=0.5, axis='x')
        ax4.grid(False, axis='y')

        # Legend for secondary y-axis
        # ax2_right.legend(['mSEMS total conc.'], loc='upper right', fontsize=11, frameon=False)
        # ax3_right.legend(['POPS total conc.'], loc='upper right', fontsize=11, frameon=False)
        # ax4_right.legend(['mCDA total conc.'], loc='upper right', fontsize=11, frameon=False)

        # X-axis formatting for all subplots
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        ax4.set_xlabel('Time', fontsize=13, fontweight='bold', labelpad=10)
        ax4.tick_params(axis='x', rotation=90, labelsize=11)

        """ SET TIME RANGE (DATE + TIME) """
        # ax3.set_xlim(pd.Timestamp('2025-02-12T07:55:00'), pd.Timestamp('2025-02-12T10:20:00'))

        """ SAVE PLOT """

        print("Saving figure to:", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    def _check_schema_contains_instrument(self, instrument: Instrument) -> bool:
        if instrument not in self._output_schema.instruments:
            logger.warning(f"{self._output_schema.name} does not contain {instrument.name}. Skipping.")
            return False
        return True

    @property
    def _flight_computer(self) -> FlightComputer:
        flight_computer = next(instrument for instrument in self._instruments if isinstance(instrument, FlightComputer))
        return flight_computer


def _get_instruments(df: pd.DataFrame, metadata: BaseModel) -> tuple[list[Instrument], Instrument]:
    flight_computer_version = None
    flight_computer_outdated_name = "flight_computer"

    instruments = []
    reference_instrument = None

    for name in metadata.instruments:
        is_reference = name == metadata.reference_instrument

        if name == flight_computer_outdated_name:
            if flight_computer_version is None:
                if f"{flight_computer_outdated_name}_{flight_computer_v1.pressure_variable}" in df.columns:
                    flight_computer_version = "v1"
                elif f"{flight_computer_outdated_name}_{flight_computer_v2.pressure_variable}" in df.columns:
                    flight_computer_version = "v2"
                else:
                    raise ValueError("Could not determine flight computer version. "
                                     "Please specify `flight_computer_version` manually.")
            name += f"_{flight_computer_version}"

        instrument = Instrument.REGISTRY[name]
        if is_reference:
            reference_instrument = instrument
        instruments.append(instrument)

    return instruments, reference_instrument

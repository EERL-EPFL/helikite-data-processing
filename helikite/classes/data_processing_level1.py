import logging
import pathlib
from numbers import Number
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

from helikite.classes.base import BaseProcessor, function_dependencies, OutputSchema
from helikite.constants import constants
from helikite.instruments import Instrument, pops, msems_inverted, mcda, cpc, \
    stap, co2
from helikite.instruments.co2 import process_CO2_STP
from helikite.instruments.cpc3007 import CPC
from helikite.instruments.flight_computer import FlightComputer
from helikite.instruments.mcda_instrument import mcda_concentration_calculations, mCDA_STP_normalization, \
    plot_mcda_distribution, plot_mcda_vertical_distribution, Midpoint_diameter_list
from helikite.instruments.msems import mSEMS_total_conc_dN, mSEMS_STP_normalization, plot_msems_distribution
from helikite.instruments.pops import POPS_total_conc_dNdlogDp, POPS_STP_normalization
from helikite.instruments.stap import STAP_STP_normalization
from helikite.processing import choose_outliers
from helikite.processing.post.TandRH import T_RH_averaging, plot_T_RH
from helikite.processing.post.altitude import altitude_calculation_barometric, plot_altitude
from helikite.processing.post.level1 import flight_profiles_1
from helikite.processing.post.outliers import plot_outliers_check, plot_gps_on_map

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


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
    def T_RH_averaging(self,
                       columns_t: list[str] | None = None, columns_rh: list[str] | None = None,
                       nan_threshold: int = 400):
        columns_t = columns_t if columns_t is not None else self._build_FC_T_columns()
        columns_rh = columns_rh if columns_rh is not None else self._build_FC_RH_columns()
        self._df = T_RH_averaging(self._df, columns_t, columns_rh, nan_threshold)

    @function_dependencies(required_operations=["T_RH_averaging"], use_once=False)
    def plot_T_RH(self, save_path: str | pathlib.Path | None = None):
        plot_T_RH(self._df, self._flight_computer, save_path)

    def _build_FC_T_columns(self) -> list[str]:
        return [
            f"{self._flight_computer.name}_{self._flight_computer.T1_column}",
            f"{self._flight_computer.name}_{self._flight_computer.T2_column}",
        ]

    def _build_FC_RH_columns(self) -> list[str]:
        return [
            f"{self._flight_computer.name}_{self._flight_computer.H1_column}",
            f"{self._flight_computer.name}_{self._flight_computer.H2_column}",
        ]

    @function_dependencies(required_operations=["T_RH_averaging"], use_once=False)
    def altitude_calculation_barometric(self, offset_to_add: Number = 0):
        self._df = altitude_calculation_barometric(self._df, self._metadata, offset_to_add)

    @function_dependencies(required_operations=["altitude_calculation_barometric"], use_once=False)
    def plot_altitude(self):
        plot_altitude(self._df)

    @function_dependencies(required_operations=[], use_once=True)
    def add_missing_columns(self):
        all_missing_columns = {}
        for instrument in self._output_schema.instruments:
            is_reference = instrument == self._reference_instrument

            all_columns = instrument.get_expected_columns(level=0, is_reference=is_reference, with_dtype=True)
            missing_columns = {column: t for column, t in all_columns.items() if column not in self._df.columns}

            if len(missing_columns) != 0:
                missing_columns_names = list(missing_columns.keys())
                if instrument in self._instruments:
                    logger.warning(f"{instrument} is present but missing columns: {missing_columns_names}.")
                logger.info(f"Adding missing columns for {instrument.name}: {missing_columns_names}")

            all_missing_columns |= missing_columns

        if len(all_missing_columns) != 0:
            missing_df = pd.DataFrame({col: pd.Series(dtype=t) for col, t in all_missing_columns.items()},
                                      index=self._df.index)
            self._df = pd.concat([self._df, missing_df], axis=1)


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

    @function_dependencies(required_operations=["mSEMS_STP_normalization"], use_once=False)
    def plot_msems_distribution(self, time_start=None, time_end=None):
        if self._check_schema_contains_instrument(msems_inverted):
            plot_msems_distribution(self._df, time_start, time_end)

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

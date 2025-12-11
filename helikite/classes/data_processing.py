import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import folium
import pandas as pd
from ipywidgets import VBox
from pydantic import BaseModel

from helikite.classes.base import BaseProcessor, function_dependencies
from helikite.constants import constants
from helikite.instruments import Instrument, flight_computer_v1, flight_computer_v2, smart_tether, pops, msems_readings, \
    msems_inverted, msems_scan, mcda, filter, tapir, cpc, stap, stap_raw, co2, mcpc
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
from helikite.processing.post.outliers import plot_outliers_check, plot_gps_on_map

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


@dataclass(frozen=True)
class OutputSchema:
    instruments: list[Instrument]
    """List of instruments whose columns should be present in the output dataframe."""


# TODO: implement other schemas used
class OutputSchemas(OutputSchema, Enum):
    ORACLES = [
        flight_computer_v2,
        smart_tether,
        pops,
        msems_readings,
        msems_inverted,
        msems_scan,
        mcda,
        filter,
        tapir,
        cpc
    ]
    TURTMANN = [
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
        mcpc,
    ]


class DataProcessorLevel1(BaseProcessor):
    def __init__(
        self,
        df: pd.DataFrame,
        metadata: BaseModel,
        output_schema: OutputSchema,
    ) -> None:
        super().__init__()
        self._df = df.copy()
        self._metadata = metadata
        self._output_schema = output_schema
        self._outlier_files: set[str] = set()
        self._instruments = _get_instruments(df, metadata)

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
    def add_missing_columns(self, only_if_instrument_missing: bool = True):
        for instrument in self._output_schema.instruments:
            if not only_if_instrument_missing and instrument in self._instruments:
                continue  # columns should be already present
            else:
                all_columns = self.get_expected_columns_for(instrument)
                missing_columns = [column for column in all_columns if column not in self._df.columns]

                self._df[missing_columns] = pd.NA

    @function_dependencies(required_operations=["altitude_calculation_barometric"], use_once=False)
    def process_CO2_STP(self, min_threshold=200, max_threshold=500):
        if self._check_schema_contains_instrument(co2):
            self._df = process_CO2_STP(self._df, min_threshold, max_threshold)

    @function_dependencies(required_operations=["altitude_calculation_barometric"], use_once=False)
    def STAP_STP_normalization(self):
        if self._check_schema_contains_instrument(stap):
            self._df = STAP_STP_normalization(self._df)

    @function_dependencies(required_operations=["altitude_calculation_barometric"], use_once=False)
    def POPS_total_conc_dNdlogDp(self):
        if self._check_schema_contains_instrument(pops):
            self._df = POPS_total_conc_dNdlogDp(self._df)

    @function_dependencies(required_operations=["POPS_total_conc_dNdlogDp"], use_once=False)
    def POPS_STP_normalization(self):
        if self._check_schema_contains_instrument(pops):
            self._df = POPS_STP_normalization(self._df)

    @function_dependencies(required_operations=["altitude_calculation_barometric"], use_once=False)
    def mSEMS_total_conc_dN(self):
        if self._check_schema_contains_instrument(msems_inverted):
            self._df = mSEMS_total_conc_dN(self._df)

    @function_dependencies(required_operations=["mSEMS_total_conc_dN"], use_once=False)
    def mSEMS_STP_normalization(self):
        if self._check_schema_contains_instrument(msems_inverted):
            self._df = mSEMS_STP_normalization(self._df)

    @function_dependencies(required_operations=["altitude_calculation_barometric"], use_once=False)
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

    @function_dependencies(required_operations=["altitude_calculation_barometric"], use_once=False)
    def CPC_STP_normalization(self):
        if self._check_schema_contains_instrument(cpc):
            self._df = CPC.CPC_STP_normalization(self._df)

    def _check_schema_contains_instrument(self, instrument: Instrument) -> bool:
        if instrument not in self._output_schema.instruments:
            logger.warning(f"{self._output_schema.name} does not contain {instrument.name}. Skipping.")
            return False
        return True

    @property
    def _flight_computer(self) -> FlightComputer:
        flight_computer = next(instrument for instrument in self._instruments if isinstance(instrument, FlightComputer))
        return flight_computer

    @staticmethod
    def get_expected_columns_for(instrument: Instrument) -> list[str]:
        """Returns a list of column names expected for the given instrument in the final dataframe."""
        column_pressure = f"{instrument.name}_{constants.HOUSEKEEPING_VAR_PRESSURE}"

        expected_columns = [f"{instrument.name}_{column}" for column in instrument.dtype.keys()]
        if instrument.pressure_variable is not None:
            expected_columns.append(column_pressure)

        return expected_columns


def _get_instruments(df: pd.DataFrame, metadata: BaseModel) -> list[Instrument]:
    flight_computer_version = None
    flight_computer_outdated_name = "flight_computer"

    instruments = []
    for name in metadata.instruments:
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

        instruments.append(Instrument.REGISTRY[name])
    return instruments

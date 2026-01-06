import logging
import pathlib
from datetime import datetime
from numbers import Number
from typing import Any

import folium
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import VBox
from pydantic import BaseModel

from helikite import Cleaner
from helikite.classes.base import BaseProcessor, function_dependencies, get_instruments_from_cleaned_data, \
    launch_operations_changing_df
from helikite.classes.output_schemas import OutputSchema
from helikite.constants import constants
from helikite.instruments import Instrument
from helikite.instruments.base import filter_columns_by_instrument
from helikite.metadata.models import Level0
from helikite.processing import choose_outliers
from helikite.processing.post.TandRH import T_RH_averaging, plot_T_RH
from helikite.processing.post.altitude import altitude_calculation_barometric, plot_altitude
from helikite.processing.post.level1 import flight_profiles_1, plot_size_distributions
from helikite.processing.post.outliers import plot_outliers_check, plot_gps_on_map, convert_gps_coordinates

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


class DataProcessorLevel1(BaseProcessor):
    def __init__(self, output_schema: OutputSchema, df: pd.DataFrame, metadata: BaseModel) -> None:
        instruments, reference_instrument = get_instruments_from_cleaned_data(df, metadata)
        super().__init__(output_schema, instruments, reference_instrument)
        self._df = df.copy()
        self._metadata = metadata
        self._outliers_files: set[str] = set()

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

        for outliers_file in self._outliers_files:
            state_info += self._outliers_file_state_info(outliers_file)

        return state_info

    @function_dependencies(required_operations=[], changes_df=False, use_once=False)
    def choose_outliers(
        self,
        y: str = "flight_computer_pressure",
        outliers_file: str = "outliers.csv",
        use_coupled_columns: bool = True,
        instruments: list[Instrument] | None = None,
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
        plt.close("all")

        if instruments is None:
            columns = self._df.columns
            coupled_columns = self.coupled_columns
        else:
            columns = [y]
            coupled_columns = []
            for instrument in instruments:
                columns += filter_columns_by_instrument(self._df.columns, instrument)
                coupled_columns += instrument.coupled_columns

        coupled_columns = coupled_columns if use_coupled_columns else []

        vbox = choose_outliers(self._df[columns], y, coupled_columns, outliers_file)
        self._outliers_files.add(outliers_file)

        return vbox

    @function_dependencies(required_operations=[], changes_df=True, use_once=False)
    def fillna_if_all_missing(self, values_dict: dict[str, Any] | None = None):
        if values_dict is None:
            values_dict = {}

        for column, value in values_dict.items():
            if column not in self._df.columns:
                logger.warning(f"Column '{column}' not found in dataframe. Skipping fill.")
                continue
            if self._df[column].isna().all():
                print(f"'{column}' is all missing. Setting value to: {value}")
                self._df[column] = value
            else:
                print(f"Column '{column}' has data present. Skipping fill.")

    @function_dependencies(required_operations=["choose_outliers"], changes_df=True, use_once=False)
    def set_outliers_to_nan(self):
        for outlier_file in self._outliers_files:
            outliers = pd.read_csv(outlier_file, index_col=0, parse_dates=True)
            columns = [col for col in outliers.columns if col in self._df.columns]
            self._df.loc[outliers.index, columns] = self._df.loc[outliers.index, columns].mask(outliers)

    @function_dependencies(required_operations=["set_outliers_to_nan"], changes_df=False, use_once=False)
    def plot_outliers_check(self):
        plot_outliers_check(self._df, self._flight_computer)

    @function_dependencies(required_operations=["set_outliers_to_nan"], changes_df=True, use_once=False)
    def convert_gps_coordinates(self,
                                lat_col='flight_computer_Lat', lon_col='flight_computer_Long',
                                lat_dir='S', lon_dir='W'):
        self._df = convert_gps_coordinates(self._df, lat_col, lon_col, lat_dir, lon_dir)

    @function_dependencies(required_operations=["convert_gps_coordinates"], changes_df=False, use_once=False)
    def plot_gps_on_map(self, center_coords=(-70.6587, -8.2850), zoom_start=13) -> folium.Map:
        return plot_gps_on_map(self._df, center_coords, zoom_start)

    @function_dependencies(required_operations=["set_outliers_to_nan"], changes_df=True, use_once=False)
    def T_RH_averaging(self,
                       columns_t: list[str] | None = None, columns_rh: list[str] | None = None,
                       nan_threshold: int = 400):
        columns_t = columns_t if columns_t is not None else self._build_FC_T_columns()
        columns_rh = columns_rh if columns_rh is not None else self._build_FC_RH_columns()
        self._df = T_RH_averaging(self._df, columns_t, columns_rh, nan_threshold)

    @function_dependencies(required_operations=["T_RH_averaging"], changes_df=False, use_once=False)
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

    @function_dependencies(required_operations=["T_RH_averaging"], changes_df=True, use_once=False)
    def altitude_calculation_barometric(self, offset_to_add: Number = 0):
        self._df = altitude_calculation_barometric(self._df, self._metadata, offset_to_add)

    @function_dependencies(required_operations=["altitude_calculation_barometric"], changes_df=False, use_once=False)
    def plot_altitude(self):
        plot_altitude(self._df)

    @function_dependencies(required_operations=[], changes_df=True, use_once=True)
    def add_missing_columns(self):
        all_missing_columns = {}
        expected_columns = Cleaner.get_expected_columns(self.output_schema, with_dtype=True)

        for instrument in self._output_schema.instruments:
            instrument_expected_columns = filter_columns_by_instrument(list(expected_columns.keys()), instrument)
            missing_columns = {
                column: t for column, t in expected_columns.items()
                if column not in self._df.columns and column in instrument_expected_columns
            }

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

        if self._df.index.name not in self._df.columns:
            self._df.insert(0, self._df.index.name, self._df.index)

    @function_dependencies(required_operations=["altitude_calculation_barometric", "add_missing_columns"],
                           changes_df=True, use_once=False,
                           complete_with_arg="instrument")
    def calculate_derived(self, instrument: Instrument, verbose: bool = True, *args, **kwargs):
        if self._check_schema_contains_instrument(instrument):
            self._df = instrument.calculate_derived(self._df, verbose, *args, **kwargs)

    @function_dependencies(required_operations=["altitude_calculation_barometric", "add_missing_columns"],
                           changes_df=True, use_once=False,
                           complete_with_arg="instrument")
    def normalize(self, instrument: Instrument, verbose: bool = True, *args, **kwargs):
        if self._check_schema_contains_instrument(instrument):
            self._df = instrument.normalize(self._df, verbose, *args, **kwargs)

    @function_dependencies(required_operations=["altitude_calculation_barometric", "add_missing_columns"],
                           changes_df=False, use_once=False,
                           complete_with_arg="instrument")
    def plot_raw_and_normalized_data(self, instrument: Instrument, verbose: bool = True, *args, **kwargs):
        if self._check_schema_contains_instrument(instrument):
            plt.close("all")
            instrument.plot_raw_and_normalized(self._df, verbose, *args, **kwargs)

    @function_dependencies(required_operations=["normalize"],
                           changes_df=False, use_once=False,
                           complete_with_arg="instrument", complete_req=True)
    def plot_distribution(self, instrument: Instrument, verbose: bool = True,
                          time_start: datetime | None = None, time_end: datetime | None = None, *args, **kwargs):
        if self._check_schema_contains_instrument(instrument):
            plt.close("all")
            instrument.plot_distribution(self._df, verbose, time_start, time_end, *args, **kwargs)

    @function_dependencies(required_operations=["normalize"],
                           changes_df=False, use_once=False,
                           complete_with_arg="instrument", complete_req=True)
    def plot_vertical_distribution(self, instrument: Instrument, verbose: bool = True, *args, **kwargs):
        if self._check_schema_contains_instrument(instrument):
            plt.close("all")
            instrument.plot_vertical_distribution(self._df, verbose, *args, **kwargs)

    @function_dependencies(required_operations=[], changes_df=False, use_once=False)
    def plot_flight_profiles(self, flight_basename: str, save_path: str | pathlib.Path,
                             xlims: dict | None = None, xticks: dict | None = None):
        title = f'Flight {self._metadata.flight} ({flight_basename}) [Level 1]'
        fig = flight_profiles_1(self._df, xlims, xticks, fig_title=title)

        # Save the figure after plotting
        print("Saving figure to:", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    @function_dependencies(required_operations=[], changes_df=False, use_once=False)
    def plot_size_distr(self, flight_basename: str, save_path: str | pathlib.Path,
                        time_start: datetime | None = None, time_end: datetime | None = None,):
        title = f'Flight {self._metadata.flight} ({flight_basename}) [Level 1]'
        fig = plot_size_distributions(self._df, self._output_schema, title, time_start, time_end)

        # Save the figure after plotting
        print("Saving figure to:", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    @classmethod
    def get_expected_columns(cls,
                             output_schema: OutputSchema,
                             reference_instrument: Instrument,
                             with_dtype: bool) -> list[str] | dict[str, str]:
        data = Cleaner.get_expected_columns(output_schema, with_dtype=True)

        df_level0 = pd.DataFrame(data)
        metadata = Level0.mock(reference_instrument.name,
                               instruments=[instrument.name for instrument in output_schema.instruments])

        df_level0 = df_level0.reindex(pd.date_range(metadata.takeoff_time, metadata.landing_time, freq="1s"))
        df_level0.index = df_level0.index.astype("datetime64[s]")

        data_processor = cls(output_schema, df_level0, metadata)
        launch_operations_changing_df(data_processor)
        expected_columns = {column: str(t) for column, t in data_processor.df.dtypes.to_dict().items()}

        return list(expected_columns.keys()) if not with_dtype else expected_columns

    @staticmethod
    def read_data(level1_filepath: str | pathlib.Path) -> pd.DataFrame:
        df_level1 = pd.read_csv(level1_filepath, index_col='DateTime', parse_dates=['DateTime'])
        df_level1.rename(columns={"DateTime.1": "DateTime"}, inplace=True)

        return df_level1

    @function_dependencies(required_operations=[], changes_df=False, use_once=False)
    def export_data(self, filepath: str | pathlib.Path | None = None):
        self._df.to_csv(filepath, index=True)

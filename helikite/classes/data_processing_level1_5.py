import logging
import pathlib
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

from helikite.classes.base import BaseProcessor, get_instruments_from_cleaned_data, function_dependencies, \
    launch_operations_changing_df
from helikite.classes.data_processing_level1 import DataProcessorLevel1
from helikite.classes.output_schemas import OutputSchema, FlightProfileVariable, Level, flag_pollution_cpc, \
    Flag
from helikite.instruments import Instrument
from helikite.metadata.models import Level0
from helikite.processing import choose_outliers
from helikite.processing.post.fda import FDAParameters, FDA
from helikite.processing.post.level1 import fill_msems_takeoff_landing, create_level1_dataframe, rename_columns, \
    round_flightnbr_campaign, flight_profiles, plot_size_distributions


class DataProcessorLevel1_5(BaseProcessor):
    @property
    def level(self) -> Level:
        return Level.LEVEL1_5


    def __init__(self, output_schema: OutputSchema, df: pd.DataFrame, metadata: BaseModel,
                 flight_computer_version: str | None = None) -> None:
        instruments, reference_instrument = get_instruments_from_cleaned_data(df, metadata, flight_computer_version)
        super().__init__(output_schema, instruments, reference_instrument)
        self._df = df.copy()
        self._metadata = metadata
        self._automatic_flags_files: set[str] = set()
        self._final_flags_files: set[str] = set()

    def _data_state_info(self) -> list[str]:
        state_info = []

        for flag_file in self._automatic_flags_files:
            state_info += self._outliers_file_state_info(flag_file, add_all=True)

        for flag_file in self._final_flags_files:
            state_info += self._outliers_file_state_info(flag_file, add_all=True)

        return state_info

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @function_dependencies(required_operations=[], changes_df=True, use_once=False)
    def fill_msems_takeoff_landing(self, time_window_seconds=90):
        fill_msems_takeoff_landing(self._df, self._metadata, time_window_seconds)

    @function_dependencies(required_operations=["fill_msems_takeoff_landing"], changes_df=True, use_once=False)
    def remove_before_takeoff_and_after_landing(self):
        self._df = self._df.loc[self._metadata.takeoff_time: self._metadata.landing_time]

    @function_dependencies(required_operations=[], changes_df=True, use_once=True)
    def filter_columns(self):
        self._df = create_level1_dataframe(self._df, self._output_schema)

    @function_dependencies(required_operations=["filter_columns"], changes_df=True, use_once=True)
    def rename_columns(self):
        self._df = rename_columns(self._df, self._output_schema, self._reference_instrument)

    @function_dependencies(required_operations=["filter_columns"], changes_df=True, use_once=True)
    def round_flightnbr_campaign(self, decimals=2):
        self._df = round_flightnbr_campaign(self._df, self._metadata, self._output_schema, decimals)

    @function_dependencies(required_operations=["rename_columns"], changes_df=True, use_once=False,
                           complete_with_arg="flag")
    def detect_flag(self,
                    flag: Flag = flag_pollution_cpc,
                    auto_path: str | Path = "flag_auto.csv",
                    plot_detection: bool = True):
        params = FDAParameters(inverse=False) if flag.params is None else flag.params

        fda = FDA(self._df, flag.column_name, gt_flag_column_name=None, params=params)
        flag_values = fda.detect()
        if plot_detection:
            fda.plot_detection(yscale=flag.y_scale)

        if isinstance(auto_path, str):
            auto_path = Path(auto_path)
        auto_path.parent.mkdir(parents=True, exist_ok=True)

        flag_df = pd.DataFrame(data={"datetime": True}, index=self._df.index[(~flag_values.isna()) & flag_values])
        flag_df.to_csv(auto_path, index=True)

        self._automatic_flags_files.add(str(auto_path))

    @function_dependencies(required_operations=["rename_columns"], changes_df=True, use_once=False,
                           complete_with_arg="flag")
    def choose_flag(self,
                    flag: Flag = flag_pollution_cpc,
                    auto_path: str | Path | None = None,
                    corr_path: str | Path = "flag_corr.csv"):
        if auto_path is not None:
            try:
                shutil.copy(auto_path, corr_path)
            except shutil.SameFileError:
                logging.warning(f"{auto_path} and {corr_path} are the same file. Skipping copy.")

        vbox = choose_outliers(self._df[["datetime", flag.column_name]], y=flag.column_name,
                               coupled_columns=[], outlier_file=str(corr_path),
                               yscale=flag.y_scale, colorscale=None)
        self._final_flags_files.add(str(corr_path))

        return vbox

    @function_dependencies(required_operations=["choose_flag"], changes_df=True, use_once=False,
                           complete_with_arg="flag", complete_req=True)
    def set_flag(self,
                 flag: Flag = flag_pollution_cpc,
                 corr_path: str | Path = "flag_corr.csv",
                 mask: pd.Series | None = None):
        flag_values = pd.read_csv(corr_path, index_col=0, parse_dates=True)["datetime"]

        self._df[flag.flag_name] = 0
        self._df.loc[flag_values[flag_values].index, flag.flag_name] = 1
        self._df[flag.flag_name] = self._df[flag.flag_name].where(~self._df[flag.column_name].isna(), pd.NA)

        if mask is not None:
            full_flag = self._df[flag.flag_name].where(mask, 0)
            full_flag = full_flag.where(~self._df[flag.flag_name].isna(), pd.NA)
            self._df[flag.flag_name] = full_flag

    @function_dependencies(required_operations=["rename_columns"], changes_df=False, use_once=False)
    def plot_flight_profiles(self, flight_basename: str, save_path: str | pathlib.Path,
                             variables: list[FlightProfileVariable] | None = None):
        plt.close("all")
        title = f'Flight {self._metadata.flight} ({flight_basename}) [Level {self.level.value}]'
        fig = flight_profiles(self._df, self._reference_instrument,
                              self.level, self._output_schema, variables, fig_title=title)

        # Save the figure after plotting
        print("Saving figure to:", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    @function_dependencies(required_operations=["rename_columns"], changes_df=False, use_once=False)
    def plot_size_distr(self, flight_basename: str, save_path: str | pathlib.Path,
                        time_start: datetime | None = None, time_end: datetime | None = None):
        plt.close("all")
        title = f'Flight {self._metadata.flight} ({flight_basename}) [Level {self.level.value}]'
        fig = plot_size_distributions(self._df, self.level, self._output_schema, title, time_start, time_end)

        # Save the figure after plotting
        print("Saving figure to:", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    @classmethod
    def get_expected_columns(cls,
                             output_schema: OutputSchema,
                             reference_instrument: Instrument,
                             with_dtype: bool) -> list[str] | dict[str, str]:
        columns_level1 = DataProcessorLevel1.get_expected_columns(output_schema, reference_instrument, with_dtype=True)
        df_level1 = pd.DataFrame({c: pd.Series(dtype=t) for c, t in columns_level1.items()},
                                 index=pd.DatetimeIndex([], name="DateTime"))
        metadata = Level0.mock(reference_instrument.name,
                               instruments=[instrument.name for instrument in output_schema.instruments])
        data_processor = cls(output_schema, df_level1, metadata)
        launch_operations_changing_df(data_processor)
        expected_columns = {column: str(t) for column, t in data_processor.df.dtypes.to_dict().items()}

        return list(expected_columns.keys()) if not with_dtype else expected_columns

    @staticmethod
    def read_data(level1_5_filepath: str | pathlib.Path) -> pd.DataFrame:
        df_level1_5 = pd.read_csv(level1_5_filepath, index_col='datetime', parse_dates=['datetime'])
        df_level1_5 = df_level1_5.convert_dtypes()

        return df_level1_5

    @function_dependencies(
        required_operations=["rename_columns", "round_flightnbr_campaign"],
        changes_df=False,
        use_once=False
    )
    def export_data(self, filepath: str | pathlib.Path | None = None):
        self._df.to_csv(filepath, index=False)

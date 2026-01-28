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
from helikite.classes.output_schemas import OutputSchema, FlightProfileVariable, Level, OutputSchemas
from helikite.config import Config
from helikite.constants import constants
from helikite.instruments import Instrument
from helikite.metadata.models import Level0
from helikite.metadata.utils import load_parquet
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
        self._df = rename_columns(self._df, self._output_schema)

    @function_dependencies(required_operations=["filter_columns"], changes_df=True, use_once=True)
    def round_flightnbr_campaign(self, decimals=2):
        self._df = round_flightnbr_campaign(self._df, self._metadata, self._output_schema, decimals)

    @function_dependencies(required_operations=["rename_columns"], changes_df=True, use_once=False,
                           complete_with_arg="flag_name")
    def detect_flag(self,
                    flag_name: str = "flag_pollution",
                    column_name: str = "CPC_total_N",
                    params: FDAParameters | None = None,
                    auto_path: str | Path = "flag_auto.csv",
                    plot_detection: bool = True,
                    yscale: str = "log"):
        if params is None:
            params = FDAParameters(inverse=False)

        fda = FDA(self._df, column_name, flag_column_name=None, params=params)
        flag = fda.detect()
        if plot_detection:
            fda.plot_detection(yscale=yscale)
        self._df[flag_name] = flag

        if isinstance(auto_path, str):
            auto_path = Path(auto_path)
        auto_path.parent.mkdir(parents=True, exist_ok=True)

        flag_df = pd.DataFrame(data={"datetime": True}, index=self._df.index[(~flag.isna()) & flag])
        flag_df.to_csv(auto_path, index=True)

        self._automatic_flags_files.add(str(auto_path))

    @function_dependencies(required_operations=["rename_columns"], changes_df=True, use_once=False,
                           complete_with_arg="flag_name")
    def choose_flag(self,
                    flag_name: str = "flag_pollution",  # do not remove, it is used to complete the operation name
                    column_name: str = "CPC_total_N",
                    auto_path: str | Path | None = None,
                    corr_path: str | Path = "flag_corr.csv",
                    yscale: str = "linear"):
        if auto_path is not None:
            shutil.copy(auto_path, corr_path)

        vbox = choose_outliers(self._df[["datetime", column_name]], y=column_name,
                               coupled_columns=[], outlier_file=str(corr_path),
                               yscale=yscale, colorscale=None)
        self._final_flags_files.add(str(corr_path))

        return vbox

    @function_dependencies(required_operations=["choose_flag"], changes_df=True, use_once=False,
                           complete_with_arg="flag_name", complete_req=True)
    def set_flag(self,
                 flag_name: str = "flag_pollution",
                 column_name: str = "CPC_total_N",
                 corr_path: str | Path = "flag_corr.csv",
                 mask: pd.Series | None = None):
        flag = pd.read_csv(corr_path, index_col=0)["datetime"]

        self._df.loc[flag[flag].index, flag_name] = 1

        if mask is not None:
            full_flag = self._df[flag_name].where(mask, 0)
            full_flag = full_flag.where(~self._df[flag_name].isna(), pd.NA)
            self._df[flag_name] = full_flag

    @function_dependencies(required_operations=["rename_columns"], changes_df=False, use_once=False)
    def plot_flight_profiles(self, flight_basename: str, save_path: str | pathlib.Path,
                             variables: list[FlightProfileVariable] | None = None):
        plt.close("all")
        title = f'Flight {self._metadata.flight} ({flight_basename}) [Level {self.level.value}]'
        fig = flight_profiles(self._df, self.level, self._output_schema, variables, fig_title=title)

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
        df_level1 = pd.DataFrame({c: pd.Series(dtype=t) for c, t in columns_level1.items()})
        metadata = Level0.mock(reference_instrument.name,
                               instruments=[instrument.name for instrument in output_schema.instruments])
        data_processor = cls(output_schema, df_level1, metadata)
        launch_operations_changing_df(data_processor)
        expected_columns = {column: str(t) for column, t in data_processor.df.dtypes.to_dict().items()}

        return list(expected_columns.keys()) if not with_dtype else expected_columns

    @staticmethod
    def read_csv(level1_5_filepath: str | pathlib.Path) -> pd.DataFrame:
        df_level1_5 = pd.read_csv(level1_5_filepath, index_col='datetime', parse_dates=['datetime'])

        return df_level1_5

    @function_dependencies(
        required_operations=["rename_columns", "round_flightnbr_campaign"],
        changes_df=False,
        use_once=False
    )
    def export_data(self, filepath: str | pathlib.Path | None = None):
        self._df.to_csv(filepath, index=False)


def execute_level1_5(config: Config):
    input_dir = constants.OUTPUTS_FOLDER / "Processing"
    output_level1_5_dir = constants.OUTPUTS_FOLDER / "Processing" / "Level1.5"
    output_level1_5_dir.mkdir(parents=True, exist_ok=True)

    df_level1 = DataProcessorLevel1.read_data(input_dir / "Level1" / f"level1_{config.flight_basename}.csv")

    _, metadata = load_parquet(input_dir / "Level0" / f"level0_{config.flight_basename}.parquet")

    data_processor = DataProcessorLevel1_5(getattr(OutputSchemas, config.output_schema), df_level1, metadata)
    data_processor.fill_msems_takeoff_landing(time_window_seconds=90)
    data_processor.remove_before_takeoff_and_after_landing()
    data_processor.filter_columns()
    data_processor.rename_columns()
    data_processor.round_flightnbr_campaign(decimals=2)

    output_flags_dir = output_level1_5_dir / "flags"

    for flag_name, column_name, params in data_processor.output_schema.flags:
        auto_file = output_flags_dir / f"level1.5_{config.flight_basename}_{flag_name}_auto.csv"
        data_processor.detect_flag(flag_name, column_name, params, auto_file, plot_detection=True)
        data_processor.set_flag(flag_name, column_name, auto_file)

    save_path = output_level1_5_dir / f'Level1.5_{config.flight_basename}_SizeDistr_Flight_{config.flight}.png'
    data_processor.plot_size_distr(config.flight_basename, save_path, time_start=None, time_end=None)

    data_processor.export_data(output_level1_5_dir / f"level1.5_{config.flight_basename}.csv")
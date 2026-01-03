import shutil
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from helikite.classes.base import BaseProcessor, OutputSchema, get_instruments_from_cleaned_data, function_dependencies, \
    launch_operations_changing_df
from helikite.classes.data_processing_level1 import DataProcessorLevel1
from helikite.instruments import Instrument
from helikite.metadata.models import Level0
from helikite.processing import choose_outliers
from helikite.processing.post.fda import FDAParameters, FDA
from helikite.processing.post.level1 import fill_msems_takeoff_landing, create_level1_dataframe, rename_columns, \
    round_flightnbr_campaign


class DataProcessorLevel1_5(BaseProcessor):
    def __init__(self, output_schema: OutputSchema, df: pd.DataFrame, metadata: BaseModel) -> None:
        instruments, reference_instrument = get_instruments_from_cleaned_data(df, metadata)
        super().__init__(output_schema, instruments, reference_instrument)
        self._df = df.copy()
        self._metadata = metadata
        self._automatic_flags_file: str | None = None
        self._final_flags_file: str | None = None

    def _data_state_info(self) -> list[str]:
        state_info = []

        if self._automatic_flags_file is not None:
            state_info += self._outliers_file_state_info(self._automatic_flags_file, add_all=True)

        if self._final_flags_file is not None:
            state_info += self._outliers_file_state_info(self._final_flags_file, add_all=True)

        return state_info

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @function_dependencies(required_operations=[], changes_df=True, use_once=False)
    def fill_msems_takeoff_landing(self, time_window_seconds=90):
        fill_msems_takeoff_landing(self._df, self._metadata, time_window_seconds)

    @function_dependencies(required_operations=["fill_msems_takeoff_landing"], changes_df=True, use_once=False)
    def remove_before_takeoff_and_after_landing(self):
        self._df = self._df.loc[self._metadata.takeoff_time : self._metadata.landing_time]

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
                    plot_detection: bool = True):
        if params is None:
            params = FDAParameters(inverse=False)

        fda = FDA(self._df, column_name, flag_column_name=None, params=params)
        flag = fda.detect()
        if plot_detection:
            fda.plot_detection()
        self._df[flag_name] = flag

        if isinstance(auto_path, str):
            auto_path = Path(auto_path)
        auto_path.parent.mkdir(parents=True, exist_ok=True)

        flag_df = pd.DataFrame(data={"datetime": True}, index=self._df.index[(~flag.isna()) & flag])
        flag_df.to_csv(auto_path, index=True)

    @function_dependencies(required_operations=["rename_columns"], changes_df=True, use_once=False,
                           complete_with_arg="flag_name")
    def choose_flag(self,
                    flag_name: str = "flag_pollution",
                    column_name: str = "CPC_total_N",
                    auto_path: str | Path | None = None,
                    corr_path: str | Path = "flag_corr.csv",
                    yscale: str = "linear"):
        if auto_path is not None:
            shutil.copy(auto_path, corr_path)

        vbox = choose_outliers(self._df[["datetime", column_name]], y=column_name,
                               coupled_columns=[], outlier_file=str(corr_path),
                               yscale=yscale, colorscale=None)

        return vbox

    @function_dependencies(required_operations=["choose_flag"], changes_df=True, use_once=False,
                           complete_with_arg="flag_name", complete_req=True)
    def set_flag(self,
                 flag_name: str = "flag_pollution",
                 column_name: str = "CPC_total_N",
                 corr_path: str | Path = "flag_corr.csv"):
        flag = pd.read_csv(corr_path, index_col=0)["datetime"]

        self._df.loc[flag[flag].index, flag_name] = True

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

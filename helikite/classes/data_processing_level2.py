import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
from scipy.stats import circmean

from helikite.classes.base import BaseProcessor, get_instruments_from_cleaned_data, function_dependencies
from helikite.classes.output_schemas import OutputSchema, FlightProfileVariable
from helikite.processing.post.level1 import flight_profiles, plot_size_distributions


class DataProcessorLevel2(BaseProcessor):
    def __init__(self, output_schema: OutputSchema, df: pd.DataFrame, metadata: BaseModel) -> None:
        instruments, reference_instrument = get_instruments_from_cleaned_data(df, metadata)
        super().__init__(output_schema, instruments, reference_instrument)
        self._df = df.copy()
        self._metadata = metadata

    def _data_state_info(self) -> list[str]:
        return []

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @function_dependencies(required_operations=[], changes_df=True, use_once=True)
    def average(self, rule="10s"):
        agg_dict = {
            **{
                col: "mean"
                for col in self._df.columns
                if col not in ["campaign", "WindDir"]
            },
            "campaign": "first",
        }
        if "WindDir" in self._df.columns:
            agg_dict["WindDir"] = lambda x: circmean(x.dropna(), high=360, low=0)

        self._df = self._df.resample(rule).agg(agg_dict)

        # Convert flags to binary
        flag_cols = self._output_schema.flag_columns
        self._df[flag_cols] = (self._df[flag_cols] >= 0.5).astype(int)

        # Round
        self._df['Lat'] = self._df['Lat'].round(4)
        self._df['Long'] = self._df['Long'].round(4)
        self._df['flight_nr'] = self._df['flight_nr'].round(0).astype(int)
        cols_to_round_2 = self._df.select_dtypes(include='number').columns.difference(['Lat', 'Long', 'flight_nr'])
        self._df[cols_to_round_2] = self._df[cols_to_round_2].round(2)

        # Convert filter position to binary
        if "Filter_position" in self._df.columns:
            self._df.loc[self._df['Filter_position'] <= 1.5, 'Filter_position'] = 0
            self._df.loc[self._df['Filter_position'] > 1.5, 'Filter_position'] = 1

    @function_dependencies(required_operations=["average"], changes_df=False, use_once=False)
    def plot_flight_profiles(self, flight_basename: str, save_path: str | pathlib.Path,
                             variables: list[FlightProfileVariable] | None = None):
        plt.close("all")
        title = f'Flight {self._metadata.flight} ({flight_basename}) [Level 2]'
        fig = flight_profiles(self._df, self._output_schema, variables, fig_title=title)

        # Save the figure after plotting
        print("Saving figure to:", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    @function_dependencies(required_operations=["average"], changes_df=False, use_once=False)
    def plot_size_distr(self, flight_basename: str, save_path: str | pathlib.Path,
                        time_start: datetime | None = None, time_end: datetime | None = None):
        plt.close("all")
        title = f'Flight {self._metadata.flight} ({flight_basename}) [Level 2]'
        fig = plot_size_distributions(self._df, self._output_schema, title, time_start, time_end)

        # Save the figure after plotting
        print("Saving figure to:", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    @staticmethod
    def read_csv(level2_filepath: str | pathlib.Path) -> pd.DataFrame:
        df_level2 = pd.read_csv(level2_filepath, index_col='datetime', parse_dates=['datetime'])

        return df_level2

    @function_dependencies(required_operations=["average"], changes_df=False, use_once=False)
    def export_data(self, filepath: str | pathlib.Path | None = None):
        self._df.to_csv(filepath, index=True)

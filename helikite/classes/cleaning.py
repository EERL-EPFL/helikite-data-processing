import datetime
import logging
import os
import pathlib
from itertools import cycle
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
import psutil
import pyarrow
import pyarrow.parquet as pq
from ipywidgets import Output, VBox

from helikite.classes.base import BaseProcessor, function_dependencies
from helikite.classes.output_schemas import OutputSchema, Level, OutputSchemas
from helikite.config import Config
from helikite.constants import constants
from helikite.instruments import msems_inverted, msems_scan, flight_computer_v2
from helikite.instruments.base import Instrument, filter_columns_by_instrument
from helikite.metadata.models import Level0
from helikite.processing.helpers import temporary_attr
from helikite.processing.post import crosscorrelation

parent_process = psutil.Process().parent().cmdline()[-1]

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)

class Cleaner(BaseProcessor):
    @property
    def level(self) -> Level:
        return Level.LEVEL0

    def __init__(
        self,
        output_schema: OutputSchema,
        input_folder: str | pathlib.Path,
        flight_date: datetime.date,
        instruments: list[Instrument] | None = None,
        reference_instrument: Instrument | None = None,
        reference_instrument_shift: str | None = None,
        flight: str | None = None,
        time_takeoff: datetime.datetime | None = None,
        time_landing: datetime.datetime | None = None,
        time_offset: datetime.time = datetime.time(0, 0),
        interactive: bool = True,
    ) -> None:
        if instruments is None:
            instruments = Cleaner.detect_instruments(output_schema, input_folder)
        if reference_instrument is None:
            reference_instrument = Cleaner.choose_reference_instrument(output_schema, instruments)

        super(Cleaner, self).__init__(output_schema, instruments, reference_instrument)

        self._instruments: list[Instrument] = []  # For managing in batches
        self.input_folder: str = input_folder if isinstance(input_folder, str) else str(input_folder)
        self.flight = flight
        self.flight_date: datetime.date = flight_date
        self.time_takeoff: datetime.datetime | None = time_takeoff
        self.time_landing: datetime.datetime | None = time_landing
        self.time_offset: datetime.time = time_offset
        self.pressure_column: str = constants.HOUSEKEEPING_VAR_PRESSURE
        self.master_df: pd.DataFrame | None = None
        self.housekeeping_df: pd.DataFrame | None = None
        self._reference_instrument: Instrument = reference_instrument
        self._reference_instrument_shift: str | None = reference_instrument_shift

        # Create an attribute from each instrument.name
        for instrument in instruments:
            instrument.df_raw = instrument.read_from_folder(
                self.input_folder, quiet=True, interactive=interactive,
            )
            instrument.df = instrument.df_raw.copy(deep=True)
            instrument.df_before_timeshift = pd.DataFrame()
            instrument.date = flight_date
            instrument.pressure_column = self.pressure_column
            instrument.time_offset = {}
            instrument.time_offset["hour"] = time_offset.hour
            instrument.time_offset["minute"] = time_offset.minute
            instrument.time_offset["second"] = time_offset.second

            # Add the instrument to the Cleaner object and the list
            setattr(self, instrument.name, instrument)
            self._instruments.append(instrument)

        print(
            f"Helikite Cleaner has been initialised with "
            f"{len(self._instruments)} instruments. Use the .state() method "
            "to see the current state, and the .help() method to see the "
            "available methods."
        )

    @property
    def df(self) -> pd.DataFrame | None:
        return self.master_df

    def _data_state_info(self) -> list[str]:
        state_info = []

        # Add instrument information
        state_info.append(
            f"{'Instrument':<20}{'Records':<10}{'Reference':<10}"
        )
        state_info.append("-" * 40)

        for instrument in self._instruments:
            reference = (
                "Yes" if instrument == self._reference_instrument else "No"
            )
            state_info.append(
                f"{instrument.name:<20}{len(instrument.df):<10}{reference:<10}"
            )

        # Add general settings
        state_info.append("\n")
        state_info.append(f"{'Property':<25}{'Value':<30}")
        state_info.append("-" * 55)
        state_info.append(f"{'Input folder':<25}{self.input_folder:<30}")
        state_info.append(f"{'Flight':<25}{self.flight}")
        state_info.append(f"{'Flight date':<25}{self.flight_date}")
        state_info.append(
            f"{'Time trim from':<25}{str(self.time_takeoff):<30}"
        )
        state_info.append(f"{'Time trim to':<25}{str(self.time_landing):<30}")
        state_info.append(f"{'Time offset':<25}{str(self.time_offset):<30}")
        state_info.append(f"{'Pressure column':<25}{self.pressure_column:<30}")

        # Add dataframe information
        master_df_status = (
            f"{len(self.master_df)} records"
            if self.master_df is not None and not self.master_df.empty
            else "Not available"
        )
        housekeeping_df_status = (
            f"{len(self.housekeeping_df)} records"
            if self.housekeeping_df is not None
               and not self.housekeeping_df.empty
            else "Not available"
        )

        state_info.append(f"{'Master dataframe':<25}{master_df_status:<30}")
        state_info.append(
            f"{'Housekeeping dataframe':<25}{housekeeping_df_status:<30}"
        )

        # Add selected pressure points info
        selected_points_status = (
            f"{len(self.selected_pressure_points)}"
            if hasattr(self, "selected_pressure_points")
            else "Not available"
        )
        state_info.append(
            f"{'Selected pressure points':<25}{selected_points_status:<30}"
        )
        return state_info

    @function_dependencies(required_operations=[], changes_df=True, use_once=True)
    def set_pressure_column(
        self,
        column_name_override: str | None = None,
    ) -> None:
        """Set the pressure column for each instrument's dataframe"""

        success = []
        errors = []
        if (
            column_name_override
            and column_name_override != self.pressure_column
        ):
            print("Updating pressure column to", column_name_override)
            self.pressure_column = column_name_override

        for instrument in self._instruments:
            try:
                instrument.pressure_column = self.pressure_column
                instrument.df = (
                    instrument.set_housekeeping_pressure_offset_variable(
                        instrument.df, instrument.pressure_column
                    )
                )
                success.append(instrument.name)
            except Exception as e:
                errors.append((instrument.name, e))

        self._print_success_errors("pressure column", success, errors)

    @function_dependencies([], changes_df=True, use_once=True)
    def set_time_as_index(self) -> None:
        """Set the time column as the index for each instrument dataframe"""

        success = []
        errors = []

        for instrument in self._instruments:
            try:
                instrument.df = instrument.set_time_as_index(instrument.df)
                instrument.df = instrument.df[~instrument.df.index.duplicated(keep="first")]
                instrument.df.index = instrument.df.index.astype("datetime64[s]")
                if instrument.df.index.name not in instrument.df.columns:
                    instrument.df.insert(0, instrument.df.index.name, instrument.df.index)
                success.append(instrument.name)
                assert instrument.df.index.dtype == "datetime64[s]", f"Unexpected index type: {instrument.df.index.dtype}"
            except Exception as e:
                errors.append((instrument.name, e))

        self._print_success_errors("time as index", success, errors)

    @function_dependencies(["set_time_as_index"], changes_df=True, use_once=True)
    def data_corrections(
        self,
        start_altitude: float = None,
        start_pressure: float = None,
        start_temperature: float = None,
    ) -> None:
        success = []
        errors = []

        for instrument in self._instruments:
            try:
                instrument.df = instrument.data_corrections(
                    instrument.df,
                    start_altitude=start_altitude,
                    start_pressure=start_pressure,
                    start_temperature=start_temperature,
                )
                success.append(instrument.name)
            except Exception as e:
                errors.append((instrument.name, e))

        if self._reference_instrument_shift is not None:
            self._reference_instrument.df = self._reference_instrument.df.shift(
                periods=self._reference_instrument_shift, freq="s"
            )

        self._print_success_errors("data corrections", success, errors)

    @function_dependencies(["set_time_as_index", "set_pressure_column"], changes_df=False, use_once=False)
    def plot_pressure(self) -> None:
        """Creates a plot with the pressure measurement of each instrument

        Assumes the pressure column has been set for each instrument
        """
        fig = go.Figure()

        # Use Plotly's default color sequence
        color_cycle = cycle(plotly.colors.qualitative.Plotly)

        for instrument in self._instruments:
            # Check that the pressure column exists
            if instrument.pressure_column not in instrument.df.columns:
                print(
                    f"Note: {instrument.name} does not have a pressure column"
                )
                continue

            # Get the next color from the cycle
            color = next(color_cycle)

            # Plot the main pressure data
            fig.add_trace(
                go.Scatter(
                    x=instrument.df.index,
                    y=instrument.df[instrument.pressure_column],
                    name=instrument.name,
                    line=dict(color=color),
                )
            )

            # Plot the df_before_timeshift if it exists and is not empty
            if (
                hasattr(instrument, "df_before_timeshift")
                and not instrument.df_before_timeshift.empty
                and instrument.pressure_variable
                in instrument.df_before_timeshift.columns
            ):
                fig.add_trace(
                    go.Scatter(
                        x=instrument.df_before_timeshift.index,
                        y=instrument.df_before_timeshift[
                            instrument.pressure_column
                        ],
                        name=f"{instrument.name} (before timeshift)",
                        line=dict(color=color, dash="dash"),  # Dashed line
                    )
                )

        fig.update_layout(
            title="Pressure Measurements",
            xaxis_title="Time",
            yaxis_title="Pressure (hPa)",
            legend=dict(
                title="Instruments",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05,
                orientation="v",
            ),
            margin=dict(l=40, r=150, t=50, b=40),
            height=800,
            width=1200,
        )

        fig.show()

    @function_dependencies(
        required_operations=[("correct_time_and_pressure", "pressure_based_time_synchronization")],
        changes_df=False,
        use_once=False
    )
    def plot_time_sync(self, save_path: str | pathlib.Path, skip: list[Instrument]):
        plt.close('all')
        fig, (ax) = plt.subplots(1, 1, figsize=(10, 8))

        for instrument in self._instruments:
            if instrument in skip:
                continue

            if instrument.pressure_column not in instrument.df.columns:
                print(f"Note: {instrument.name} does not have a pressure column")
                continue

            color = self._output_schema.colors[instrument]
            if instrument != self._reference_instrument:
                # Initial (before timeshift)
                ax.plot(
                    instrument.df_before_timeshift.index,
                    instrument.df_before_timeshift['pressure'],
                    linestyle='dashed',
                    color=color,
                    label=f'{instrument} init'
                )

            # Corrected
            label = f'{instrument} corr' if instrument != self._reference_instrument else str(instrument)
            pressure = instrument.df['pressure'].ffill()
            ax.plot(instrument.df.index, pressure, color=color, label=label)

        ax.set_xlabel("Time", fontsize=10, labelpad=15, fontweight='bold')
        ax.set_ylabel("Pressure (hPa)", fontsize=10, labelpad=15, fontweight='bold')
        ax.set_title(f'Flight {self.flight} ({self.flight_date}_B) [Level {self.level.value}]', fontsize=12, fontweight='bold',
                     pad=15)
        ax.grid(ls='--')
        ax.legend(ncols=2)

        # Show the plot
        plt.show()

        print("Saving figure to:", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    @function_dependencies(["set_time_as_index"], changes_df=True, use_once=False)
    def remove_duplicates(self) -> None:
        """Remove duplicate rows from each instrument based on time index,
        and clear repeated values in 'msems_scan_', 'msems_inverted_' columns,
        and specific 'mcda_*' columns, keeping only the first instance.
        """

        success = []
        errors = []

        for instrument in self._instruments:
            try:
                # Step 1: Remove duplicate rows based on the time index
                # instrument.df = instrument.remove_duplicates(instrument.df)

                # Step 2: Handle repeated values in msems_scan
                if 'scan_direction' in self.msems_scan.df.columns:
                    # Compare current value with previous value to detect changes
                    is_change = self.msems_scan.df['scan_direction'] != self.msems_scan.df['scan_direction'].shift(1)
                    # Nullify repeated rows (set to NaN) where there's no change
                    self.msems_scan.df.loc[~is_change, self.msems_scan.df.columns != self.msems_scan.df.index.name] = np.nan

                else:
                    print(f"No 'scan_direction' column found in {self.msems_scan.name}.")

                # Step 3: Handle repeated values in msems_inverted
                if 'scan_direction' in self.msems_inverted.df.columns:
                    # Compare current value with previous value to detect changes
                    is_change_inverted = self.msems_inverted.df['scan_direction'] != self.msems_inverted.df['scan_direction'].shift(1)
                    # Nullify repeated rows (set to NaN) where there's no change
                    self.msems_inverted.df.loc[~is_change_inverted, self.msems_inverted.df.columns != self.msems_inverted.df.index.name] = np.nan
                else:
                    print(f"No 'scan_direction' column found in {self.msems_inverted.name}.")

                # Step 4: Handle repeated values in mcda
                if 'measurement_nbr' in self.mcda.df.columns:
                    # Compare current value with previous value to detect changes in measurement_nbr
                    is_change_mcd = self.mcda.df['measurement_nbr'] != self.mcda.df['measurement_nbr'].shift(1)
                    # List of target columns where you want to nullify repetitive data
                    target_columns_mcda = [
                        'Temperature', 'Pressure', 'RH', 'pmav', 'offset1', 'offset2',
                        'calib1', 'calib2', 'measurement_nbr', 'pressure'
                    ]
                    # Nullify repeated rows (set to NaN) where there's no change
                    self.mcda.df.loc[~is_change_mcd, target_columns_mcda] = np.nan

                else:
                    print(f"No 'measurement_nbr' column found in {self.mcda.name}.")

                success.append(instrument.name)

            except Exception as e:
                errors.append((instrument.name, e))

        self._print_success_errors("duplicate removal", success, errors)

    @function_dependencies(
        required_operations=[("correct_time_and_pressure", "pressure_based_time_synchronization"), "remove_duplicates"],
        changes_df=True,
        use_once=False
    )
    def merge_instruments(
        self, tolerance_seconds: int = 0, remove_duplicates: bool = True
    ) -> None:
        """Merges all the dataframes from the instruments into one dataframe.

        All columns from all instruments are included in the merged dataframe,
        with unique prefixes to avoid column name collisions.

        Parameters
        ----------
        tolerance_seconds: int
            The tolerance in seconds for merging dataframes.
        remove_duplicates: bool
            If True, removes duplicate times and keeps the first result.
        """

        # Ensure all dataframes are sorted by their index
        for instrument in self._instruments:
            instrument.df.sort_index(inplace=True)

            # Use same time resolution as reference instrument
            instrument.df.index = instrument.df.index.astype(
                self._reference_instrument.df.index.dtype
            )

        print("Using merge_asof to align and merge instrument dataframes.")

        # Create a full 1s-spaced datetime index to preserve in the final master_df
        start = self._reference_instrument.df.index.min()
        end = self._reference_instrument.df.index.max()
        full_index = pd.date_range(start=start, end=end, freq="1s")

        # Start with the reference instrument dataframe
        self.master_df = self._reference_instrument.df.copy()
        self.master_df.columns = [
            f"{self.reference_instrument.name}_{col}"
            for col in self.master_df.columns
        ]

        # Merge all other dataframes with merge_asof
        for instrument in self._instruments:
            if instrument == self._reference_instrument:
                continue

            temp_df = instrument.df.copy()
            temp_df.columns = [
                f"{instrument.name}_{col}" for col in temp_df.columns
            ]

            self.master_df = pd.merge_asof(
                self.master_df,
                temp_df,
                left_index=True,
                right_index=True,
                tolerance=pd.Timedelta(seconds=tolerance_seconds),
                direction="nearest",
            )

        # Remove duplicates in the merged dataframe if flag is set
        if remove_duplicates:
            self.master_df = self.master_df[
                ~self.master_df.index.duplicated(keep="first")
            ]

        print(
            "Master dataframe created using merge_asof. "
            "Available at Cleaner.master_df."
        )

    @function_dependencies(["define_flight_times", "merge_instruments"], changes_df=False, use_once=False)
    def export_data(
        self,
        filepath: str | pathlib.Path | None = None,
    ) -> None:
        """Export all data columns from all instruments to local files

        The function will export a CSV and a Parquet file with all columns
        from all instruments. The files will be saved in the current working
        directory unless a filename is provided.

        The Parquet file will include the metadata from the class.

        """

        # Raise error if the dataframes have not been merged
        if self.master_df is None or self.master_df.empty:
            raise ValueError(
                "Dataframes have not been merged. Please run the "
                "merge_instruments() method."
            )

        if filepath is None:
            # Include the date and time of the first row of the reference
            # instrument in the filename
            time = (
                self.master_df.index[0]
                .to_pydatetime()
                .strftime("%Y-%m-%dT%H-%M")
            )
            filepath = f"level0_{time}"  # noqa


        metadata = Level0(
            flight=self.flight,
            flight_date=self.flight_date,
            takeoff_time=self.time_takeoff,
            landing_time=self.time_landing,
            reference_instrument=self.reference_instrument.name,
            instruments=[instrument.name for instrument in self._instruments],
        ).model_dump()

        all_columns = list(self.master_df.columns)

        # Convert the master dataframe to a PyArrow Table
        table = pyarrow.Table.from_pandas(
            self.master_df[all_columns], preserve_index=True
        )

        # We can only replace metadata so we need to merge with existing
        level0_metadata = orjson.dumps(metadata)
        existing_metadata = table.schema.metadata
        merged_metadata = {
            **{"level0": level0_metadata},
            **existing_metadata,
        }

        dirpath = pathlib.Path(filepath).parent
        dirpath.mkdir(parents=True, exist_ok=True)

        # Save the metadata to the Parquet file
        table = table.replace_schema_metadata(merged_metadata)
        pq.write_table(table, f"{filepath}.parquet")

        self.master_df[all_columns].to_csv(f"{filepath}.csv")

        print(
            f"\nDone. The file '{filepath}'.{{csv|parquet}} contains all "
            "instrument data. The metadata is stored in the Parquet file."
        )

    def _apply_rolling_window_to_pressure(
        self,
        instrument,
        window_size: int = 20,
    ):
        """Apply rolling window to the pressure measurements of instrument

        Then plot the pressure measurements with the rolling window applied
        """
        if instrument.pressure_column not in instrument.df.columns:
            raise ValueError(
                f"Note: {instrument.name} does not have a pressure column"
            )

        instrument.df[instrument.pressure_column] = (
            instrument.df[instrument.pressure_column]
            .rolling(window=window_size)
            .mean()
        )

        print(
            f"Applied rolling window to pressure for {instrument.name}"
            f" on column '{instrument.pressure_column}'"
        )

    @function_dependencies(["set_pressure_column", "data_corrections"], changes_df=False, use_once=False)
    def define_flight_times(self):
        """Creates a plot to select the start and end of the flight

        Uses the pressure measurements of the reference instrument to select
        the start and end of the flight. The user can click on the plot to
        select the points.
        """

        # Create a figure widget for interactive plotting
        fig = go.FigureWidget()
        out = Output()
        # out.append_stdout('Output appended with append_stdout')
        out.append_stdout(f"\nStart time: {self.time_takeoff}\n")
        out.append_stdout(f"End time: {self.time_landing}\n")
        out.append_stdout("Click to set the start time.\n")

        # Initialize the list to store selected pressure points
        self.selected_pressure_points = []

        @out.capture(clear_output=True)
        def select_point_callback(trace, points, selector):
            # Callback function for click events to select points
            if points.point_inds:
                point_index = points.point_inds[0]
                selected_x = trace.x[point_index]

                # Add a message if the start/end time has not been satisfied.
                # As we are clicking on a point to define it, the next click
                # should be the end time. If both are set, then it will be
                # reset.
                if (self.time_takeoff is None) or (
                    self.time_takeoff is not None
                    and self.time_landing is not None
                ):
                    # Set the start time, and reset the end time
                    self.time_takeoff = selected_x
                    self.time_landing = None
                    print(f"Start time: {self.time_takeoff}")
                    print(f"End time: {self.time_landing}")
                    print("Click to set the end time.")
                elif (
                    self.time_takeoff is not None and self.time_landing is None
                ):
                    # Set the end time
                    self.time_landing = selected_x
                    print(f"Start time: {self.time_takeoff}")
                    print(f"End time: {self.time_landing}")
                    print(
                        "Click again if you wish to reset the times and set "
                        "a new start time"
                    )
                else:
                    print("Something went wrong with the time selection.")

            # Update the plot if self.time_takeoff and self.time_landing
            # have been set or modified
            if self.time_takeoff is not None and self.time_landing is not None:
                # If there is a vrect, delete it and add a new one. First,
                # find the vrect shape
                shapes = [
                    shape
                    for shape in fig.layout.shapes
                    if shape["type"] == "rect"
                ]

                # If there is a vrect, delete it
                if shapes:
                    fig.layout.shapes = []

                # Add a new vrect
                fig.add_vrect(
                    x0=self.time_takeoff,
                    x1=self.time_landing,
                    fillcolor="rgba(0, 128, 0, 0.25)",
                    layer="below",
                    line_width=0,
                )

        # Add the initial time range to the plot
        if self.time_takeoff is not None and self.time_landing is not None:
            # Add a new vrect
            fig.add_vrect(
                x0=self.time_takeoff,
                x1=self.time_landing,
                fillcolor="rgba(0, 128, 0, 0.25)",
                layer="below",
                line_width=0,
            )
        # Iterate through instruments to plot pressure data
        for instrument in self._instruments:
            # Check if the pressure column exists in the instrument dataframe
            if instrument.pressure_column not in instrument.df.columns:
                print(
                    f"Note: {instrument.name} does not have a pressure column"
                )
                continue

            # Add pressure trace to the plot. If it is the reference
            # instrument, plot it with a thicker/darker line, otherwise,
            # plot it lightly with some transparency.
            if instrument == self._reference_instrument:
                fig.add_trace(
                    go.Scatter(
                        x=instrument.df.index,
                        y=instrument.df[instrument.pressure_column],
                        name=instrument.name,
                        line=dict(width=2, color="red"),
                        opacity=1,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=instrument.df.index,
                        y=instrument.df[instrument.pressure_column],
                        name=instrument.name,
                        line=dict(width=1, color="grey"),
                        opacity=0.25,
                        hoverinfo="skip",
                    )
                )

        # Attach the callback to all traces
        for trace in fig.data:
            # Only allow the reference instrument to be clickable
            if trace.name == self._reference_instrument.name:
                trace.on_click(select_point_callback)

        # Customize plot layout
        fig.update_layout(
            title="Select flight times",
            xaxis_title="Time",
            yaxis_title="Pressure (hPa)",
            hovermode="closest",
            showlegend=True,
            height=600,
            width=800,
        )

        # Show plot with interactive click functionality
        return VBox([fig, out])  # Use VBox to stack the plot and output

    @function_dependencies(["set_pressure_column", "data_corrections"], changes_df=True, use_once=False)
    def pressure_based_time_synchronization(self, max_lag: int | None = None):
        """
        Time-synchronizes instruments by maximizing cross-correlation between the pressure data of the reference
        instrument and the other instruments. Runs multiple iterations using a coarse-to-fine scheme,
        starting with a coarse adjustment and refining until the final lags are found.

        Parameters
        ----------
        max_lag : int, optional
            Maximum allowed time lag in seconds used at the first (coarsest) level.
            If None, it defaults to half of the flight duration.
        """
        self._build_df_pressure()

        df_pressure = self.df_pressure.copy()
        ref_index = self.reference_instrument.df.index

        instrument_lags = {}
        final_lags = {}

        for instrument in self._instruments:
            # skip reference or instruments without pressure data
            if instrument == self._reference_instrument or instrument.name not in df_pressure.columns:
                continue

            instrument_lags[instrument] = 0
            final_lags[instrument] = 0

            # preliminary coarse adjustment if indices don't overlap at all
            instr_index = instrument.df.index
            if instr_index.max() < ref_index.min() or instr_index.min() > ref_index.max():
                instrument_lags[instrument] = int((ref_index.mean() - instr_index.mean()).total_seconds())
                df_pressure[instrument.name] = df_pressure[instrument.name].shift(instrument_lags[instrument], freq="s")
                final_lags[instrument] = instrument_lags[instrument]

        if max_lag is None:
            max_lag = (ref_index.max() - ref_index.min()).total_seconds() // 2

        max_lag_count = 200

        # iteratively narrow down the best lag using a decreasing step size
        while max_lag >= max_lag_count:
            step = max_lag // max_lag_count
            max_lag = step * max_lag_count

            lags = np.arange(-max_lag, max_lag + 1, step, dtype=int)

            instrument_lags = self._get_best_instrument_lags(df_pressure, lags)
            for instrument, lag in instrument_lags.items():
                col = instrument.name
                shifted = df_pressure[col].shift(lag, freq="s")

                # ensure we don't shift data too far out of bounds (keep at least 80% coverage)
                if shifted.reindex(df_pressure.index).count() / df_pressure[col].count() < 0.8:
                    continue

                new_index = df_pressure.index.union(shifted.index)
                df_pressure = df_pressure.reindex(new_index)
                df_pressure[col] = shifted
                final_lags[instrument] += lag

            # reduce the search range for the next iteration
            max_lag = (max_lag + 1) // 2

        # apply the final calculated lags to the actual instrument dataframes
        for instrument, final_lag in final_lags.items():
            print(f"Shifting {instrument.name} by {final_lag} seconds")
            # if time correction was already performed, then we don't need to set `df_before_timeshift` again
            if len(instrument.df_before_timeshift) == 0:
                instrument.df_before_timeshift = instrument.df.copy()
            instrument.df.index = instrument.df.index.shift(final_lag, freq="s")

    def _get_best_instrument_lags(self, df_pressure, lags: np.ndarray) -> dict[Instrument, Number]:
        best_instrument_lags = {}

        df_new = crosscorrelation.df_derived_by_shift(df_pressure, lags, NON_DER=[self.reference_instrument.name])

        with np.errstate(invalid='ignore', divide='ignore'):
            self.df_corr = df_new.corrwith(df_new.iloc[:, 0]).to_frame().T
            self.df_corr.index = [df_new.columns[0]]

        for instrument in self._instruments:
            if instrument == self._reference_instrument:
                continue

            if instrument.pressure_column in instrument.df.columns:
                instrument.corr_df = crosscorrelation.df_findtimelag(
                    self.df_corr, lags, instrument,
                )

                instrument.corr_max_val = max(instrument.corr_df)
                instrument.corr_max_idx = instrument.corr_df.idxmax(axis=0)

                best_instrument_lags[instrument] = instrument.corr_max_idx

        return best_instrument_lags

    @function_dependencies(["set_pressure_column", "data_corrections"], changes_df=True, use_once=False)
    def correct_time_and_pressure(
        self,
        max_lag=180,
        walk_time_seconds: int | None = None,
        apply_rolling_window_to: list[Instrument] = [],
        rolling_window_size: int = constants.ROLLING_WINDOW_DEFAULT_SIZE,
        reference_pressure_thresholds: tuple[float, float] | None = None,
        detrend_pressure_on: list[Instrument] = [],
        offsets: list[tuple[Instrument, int]] = [],
        match_adjustment_with: list[tuple[Instrument, Instrument]] = [],
    ):
        """Correct time and pressure for each instrument based on time lag.

        Parameters
        ----------
        max_lag: int
            The maximum time lag to consider for cross-correlation.
        walk_time_seconds: int
            The time in seconds to walk the pressure data to match the
            reference instrument.
        apply_rolling_window_to: list[Instrument]
            A list of instruments to apply a rolling window to the pressure
            data.
        rolling_window_size: int
            The size of the rolling window to apply to the pressure data.
        reference_pressure_thresholds: tuple[float, float]
            A tuple with two values (low, high) to apply a threshold to the
            reference instrument's pressure data.
        detrend_pressure_on: list[Instrument]
            A list of instruments to detrend the pressure data.
        offsets: list[tuple[Instrument, int]]
            A list of tuples with an instrument and an offset in seconds to
            apply to the time index.
        match_adjustment_with: dict[Instrument, list[Instrument]]
            A list of tuples with two instruments, in order to be able to
            to match the same time adjustment. This can be used,
            for example, if an instrument does not have a pressure column,
            and as such, can use the time adjustment from another instrument.
            The first instrument is the one that has the index adjustment, and
            the second instrument is the one that will be adjusted.

        """
        # Apply manual offsets before cross-correlation
        if offsets:
            print("Applying manual offsets:")
        for instrument, offset_seconds in offsets:
            print(f"\t{instrument.name}: {offset_seconds} seconds")

            # Adjust the index (DateTime) by the specified offset
            instrument.df.index = instrument.df.index + pd.Timedelta(
                seconds=offset_seconds,
                unit="s",
            )
            instrument.df.index = instrument.df.index.floor('s')

        if reference_pressure_thresholds:
            # Assert the tuple has two values (low, high)
            assert len(reference_pressure_thresholds) == 2, (
                "The reference_pressure_threshold must be a tuple with two "
                "values (low, high)"
            )
            assert (
                reference_pressure_thresholds[0]
                < reference_pressure_thresholds[1]
            ), (
                "The first value of the reference_pressure_threshold must be "
                "lower than the second value"
            )

            # Apply the threshold to the reference instrument
            self._reference_instrument.df.loc[
                (
                    self._reference_instrument.df[
                        self._reference_instrument.pressure_column
                    ]
                    > reference_pressure_thresholds[1]
                )
                | (
                    self._reference_instrument.df[
                        self._reference_instrument.pressure_column
                    ]
                    < reference_pressure_thresholds[0]
                ),
                self._reference_instrument.pressure_column,
            ] = np.nan
            self._reference_instrument.df[
                self._reference_instrument.pressure_column
            ] = (
                self._reference_instrument.df[
                    self._reference_instrument.pressure_column
                ]
                .interpolate()
                .rolling(window=rolling_window_size)
                .mean()
            )
            print(
                f"Applied threshold of {reference_pressure_thresholds} to "
                f"{self.reference_instrument.name} on "
                f"column '{self.reference_instrument.pressure_column}'"
            )

        # Apply rolling window to pressure
        if apply_rolling_window_to:
            for instrument in apply_rolling_window_to:
                self._apply_rolling_window_to_pressure(
                    instrument,
                    window_size=rolling_window_size,
                )

        lags = np.arange(-max_lag, max_lag + 1)

        self._build_df_pressure()

        takeofftime = self.df_pressure.index.asof(
            pd.Timestamp(self.time_takeoff)
        )
        landingtime = self.df_pressure.index.asof(
            pd.Timestamp(self.time_landing)
        )

        if detrend_pressure_on:
            if takeofftime is None or landingtime is None:
                raise ValueError(
                    "Could not find takeoff or landing time in the pressure "
                    "data. Check the time range and the pressure data. "
                    f"The takeoff time is {takeofftime} @ "
                    f"{self.time_takeoff} and the landing time "
                    f"is {landingtime} @ {self.time_landing}."
                )

            if detrend_pressure_on:
                print("Detrending pressure:")

            for instrument in detrend_pressure_on:
                print(f"\t{instrument.name}")
                if instrument.name not in self.df_pressure.columns:
                    raise ValueError(
                        f"\t{instrument.name} not in the df_pressure column. "
                        f"Available columns: {self.df_pressure.columns}"
                    )
                self.df_pressure[instrument.name] = (
                    crosscorrelation.presdetrend(
                        self.df_pressure[instrument.name],
                        takeofftime,
                        landingtime,
                    )
                )
                print(
                    f"\tDetrended pressure for {instrument.name} on column "
                    f"'{instrument.pressure_column}'\n"
                )

            if walk_time_seconds:
                # Apply matchpress to correct pressure
                print(
                    "Walk time adjustment is not available and will be "
                    "skipped."
                )
                # pd_walk_time = pd.Timedelta(seconds=walk_time_seconds)
                # refpresFC = (
                #     self.df_pressure[self.reference_instrument.name]
                #     .loc[takeofftime - pd_walk_time : takeofftime]
                #     .mean()
                # )

                # print("Applying match pressure correction:")
                # for instrument in self._instruments:
                #     print(f"\tWorking on instrument: {instrument.name}")
                #     if instrument == self._reference_instrument:
                #         print("\tSkipping reference instrument")
                #         continue
                #     if instrument.pressure_column not in instrument.df.columns:
                #         print(
                #             f"\tNote: {instrument.name} does not have a "
                #             "pressure column"
                #         )
                #         continue
                #     try:
                #         df_press_corr = crosscorrelation.matchpress(
                #             instrument.df[instrument.pressure_column],
                #             refpresFC,
                #             takeofftime,
                #             pd_walk_time,
                #         )
                #         instrument.df[f"{instrument.pressure_column}_corr"] = (
                #             df_press_corr
                #         )
                #     except (TypeError, AttributeError, NameError) as e:
                #         print(f"\tError in match pressure: {e}")

                #     print(
                #         "\tApplied match pressure correction for "
                #         f"{instrument.name}\n"
                #     )

        df_new = crosscorrelation.df_derived_by_shift(
            self.df_pressure,
            lags,
            NON_DER=[self.reference_instrument.name],
        )

        with np.errstate(invalid='ignore', divide='ignore'):
            self.df_corr = df_new.corrwith(df_new.iloc[:, 0]).to_frame().T
            self.df_corr.index = [df_new.columns[0]]

        print("Cross correlation:")
        for instrument in self._instruments:
            print("\tWorking on instrument:", instrument.name)
            instrument_is_matched_with = None
            for (
                    primary_instrument,
                    secondary_instrument,
            ) in match_adjustment_with:
                # If the instrument is in the match_adjustment_with list,
                # then it will be matched with the match_with instrument
                if instrument == secondary_instrument:
                    instrument_is_matched_with = primary_instrument
                    break

            if instrument == self._reference_instrument:
                print("\tSkipping reference instrument\n")
                continue
            if instrument.pressure_column in instrument.df.columns:
                instrument.corr_df = crosscorrelation.df_findtimelag(
                    self.df_corr, lags, instrument,
                )
                if instrument.corr_df.isna().all():
                    raise ValueError(
                        f"All correlation values are NaN for instrument {instrument.name}. Please check "
                        f"that time ranges of {instrument.name} and {self.reference_instrument.name} overlap."
                    )

                instrument.corr_max_val = max(instrument.corr_df)
                instrument.corr_max_idx = instrument.corr_df.idxmax(axis=0)

                print(
                    f"\tInstrument: {instrument.name} | Max val "
                    f"{instrument.corr_max_val} "
                    f"@ idx: {instrument.corr_max_idx}"
                )
                instrument.df_before_timeshift, instrument.df = (
                    crosscorrelation.df_lagshift(
                        instrument.df,
                        self._reference_instrument.df,
                        instrument.corr_max_idx,
                        instrument.name,
                    )
                )

                print()
            else:
                if instrument_is_matched_with:
                    # If the instrument is matched with another instrument,
                    # it will use the time adjustment from the matched
                    # instrument to adjust its own time index.
                    print(
                        f"\tInstrument: {instrument.name} will be matched "
                        f"with {instrument_is_matched_with.name} "
                        "after all other instruments are adjusted.\n"
                    )
                else:
                    print(
                        f"\tERROR: No pressure column in {instrument.name}\n"
                    )

        if match_adjustment_with:
            print("Applying time adjustment from primary to secondary:")
        for primary_instrument, secondary_instrument in match_adjustment_with:
            # Apply the time adjustment from the primary instrument to the
            # secondary instrument

            print(
                f"\t{primary_instrument.name} to {secondary_instrument.name}"
            )
            (
                secondary_instrument.df_before_timeshift,
                secondary_instrument.df,
            ) = crosscorrelation.df_lagshift(
                secondary_instrument.df,
                self._reference_instrument.df,
                primary_instrument.corr_max_idx,
                secondary_instrument.name,
            )
            print(
                f"\tApplied time adjustment from {primary_instrument.name} to "
                f"{secondary_instrument.name}\n"
            )

        print("Time and pressure corrections applied.")

        # Plot the corr_df for each instrument on one plot
        fig = go.Figure()
        for instrument in self._instruments:
            if hasattr(instrument, "corr_df"):
                fig.add_trace(
                    go.Scatter(
                        x=instrument.corr_df.index,
                        y=instrument.corr_df,
                        name=instrument.name,
                    )
                )

        fig.update_layout(
            title="Cross-correlation",
            xaxis_title="Lag (s)",
            yaxis_title="Correlation",
            height=800,
            width=1000,
        )

        print("Note: Cross correlation df available at Cleaner.df_corr")
        print("Note: Pressure data available at Cleaner.df_pressure")

        # Show the figure if using a jupyter notebook
        if (
            "jupyter-lab" in parent_process
            or "jupyter-notebook" in parent_process
        ):
            fig.show()

    def _build_df_pressure(self):
        self.df_pressure = self._reference_instrument.df[
            [self.reference_instrument.pressure_column]
        ].copy()
        self.df_pressure.rename(
            columns={
                self._reference_instrument.pressure_column: self._reference_instrument.name  # noqa
            },
            inplace=True,
        )

        for instrument in self._instruments:
            if instrument == self._reference_instrument:
                # We principally use the ref for this, don't merge with itself
                continue

            if instrument.pressure_column in instrument.df.columns:
                df = instrument.df[[instrument.pressure_column]].copy()
                df.index = df.index.astype(self._reference_instrument.df.index.dtype)
                df.rename(
                    columns={instrument.pressure_column: instrument.name},
                    inplace=True,
                )
                self.df_pressure = pd.merge(
                    self.df_pressure,
                    df,
                    left_index=True,
                    right_index=True,
                    how="outer",
                )
                assert self.df_pressure[instrument.name].count() == df[instrument.name].count()

        full_index = pd.date_range(start=self.df_pressure.index.min(), end=self.df_pressure.index.max(), freq="1s")
        self.df_pressure = self.df_pressure.reindex(full_index)

    @function_dependencies(["merge_instruments"], changes_df=True, use_once=True)
    def shift_msems_columns_by_90s(self):
        """
        Shift all 'msems_inverted_' and 'msems_scan_' columns by 90 seconds in time.
        """
        if not isinstance(self.master_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DateTimeIndex to apply a time-based shift.")

        cols_to_shift = (filter_columns_by_instrument(self.master_df, msems_inverted) +
                         filter_columns_by_instrument(self.master_df, msems_scan))

        if not cols_to_shift:
            print("No msems_inverted_ or msems_scan_ columns found to shift.")
            return

        self.master_df[cols_to_shift] = self.master_df[cols_to_shift].shift(freq="90s")

        print("Shifted msems_inverted and msems_scan columns by 90 seconds.")

    @function_dependencies(["set_time_as_index"], changes_df=True, use_once=False)
    def fill_missing_timestamps(
        self,
        instrument: Instrument,
        freq: str = "1S",
        fill_method: str | None = None  # Optional: "ffill", "bfill", or None
    ):
        """
        Reindex the DataFrame of the instrument to fill in missing timestamps at the specified frequency.
        Optionally forward- or backward-fill missing values.
        Prints the number of timestamps added.

        Parameters
        ----------
        instrument : Instrument
            The input DataFrame with a DateTimeIndex.
        freq : str
            The desired frequency for the DateTimeIndex (e.g., "1S" for 1 second).
        fill_method : str or None
            Method to fill missing values: "ffill", "bfill", or None (default: None).
        """
        df = instrument.df

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DateTimeIndex.")

        # Create full time range
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)  # .astype("datetime64[s]")
        full_index.name = "DateTime"

        num_missing = len(full_index.difference(df.index))
        print(f"Added {num_missing} missing timestamps.")

        # Reindex
        df_full = df.reindex(full_index)

        # Optionally fill
        if fill_method == "ffill":
            df_full = df_full.ffill()
        elif fill_method == "bfill":
            df_full = df_full.bfill()

        instrument.df = df_full

    @staticmethod
    def detect_instruments(output_schema: OutputSchema, input_folder: str | pathlib.Path) -> list[Instrument]:
        """
        Automatically detect instruments from the files available in the input folder.
        """
        all_matched_files = set()
        flight_instruments = []

        for instrument in output_schema.instruments:
            matched_files = instrument.detect_from_folder(input_folder, quiet=False, interactive=False)

            if matched_files:
                flight_instruments.append(instrument)
                for matched_file in matched_files:
                    all_matched_files.add(os.path.basename(matched_file))

        for file in os.listdir(input_folder):
            if file not in all_matched_files:
                logger.warning(f"{file} was not matched")

        return flight_instruments

    @staticmethod
    def choose_reference_instrument(output_schema: OutputSchema,
                                    instruments: list[Instrument]) -> Instrument | None:
        for candidate in output_schema.reference_instrument_candidates:
            if candidate in instruments:
                return candidate

        return None

    @classmethod
    def get_expected_columns(cls, output_schema: OutputSchema, with_dtype: bool) -> list[str] | dict[str, str]:
        expected_columns = {}
        for instrument in output_schema.instruments:
            df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in instrument.dtype.items()})
            with temporary_attr(instrument, "date", datetime.date(year=2000, month=1, day=1)):
                df = instrument.set_time_as_index(df)
                if df.index.name not in df.columns:
                    df.insert(0, df.index.name, df.index)

            # TODO: Remove this once addition of `scan_direction` is integrated in the cleaning pipeline
            if instrument.name == "msems_inverted":
                df.insert(len(df.columns), "scan_direction", pd.Series([], dtype="Int64"))

            df = instrument.data_corrections(df)

            if instrument.pressure_variable is not None:
                df = instrument.set_housekeeping_pressure_offset_variable(df, constants.HOUSEKEEPING_VAR_PRESSURE)

            expected_columns |= {f"{instrument.name}_{column}": str(t) for column, t in df.dtypes.to_dict().items()}

        if not with_dtype:
            return list(expected_columns.keys())

        return expected_columns

def execute_level0(config: Config):
    input_dir = config.campaign_data_dirpath / config.flight_basename
    output_level0_dir = constants.OUTPUTS_FOLDER / "Processing" / "Level0"
    output_level0_dir.mkdir(parents=True, exist_ok=True)

    cleaner = Cleaner(
        output_schema=OutputSchemas.from_name(config.output_schema),
        input_folder=input_dir,
        flight_date=config.flight_date,
        flight=config.flight,
        interactive=False,
    )
    reference_instrument = cleaner.reference_instrument

    if flight_computer_v2 not in cleaner.instruments:
        return

    cleaner.set_time_as_index()
    cleaner.fill_missing_timestamps(reference_instrument, freq="1s", fill_method="ffill")
    cleaner.data_corrections()
    cleaner.set_pressure_column()

    cleaner.pressure_based_time_synchronization()

    cleaner.define_flight_times()
    if cleaner.time_takeoff is None:
        cleaner.time_takeoff = reference_instrument.df.index[0].to_pydatetime() + datetime.timedelta(seconds=120)
    if cleaner.time_landing is None:
        cleaner.time_landing = reference_instrument.df.index[-1].to_pydatetime() - datetime.timedelta(seconds=120)

    cleaner.remove_duplicates()
    cleaner.merge_instruments()

    save_path = output_level0_dir / f"Level0_{config.flight_basename}_Flight_{config.flight}_TimeSync.png"
    cleaner.plot_time_sync(save_path, skip=[msems_scan])
    cleaner.shift_msems_columns_by_90s()

    cleaner.export_data(filepath=output_level0_dir / f"level0_{config.flight_basename}")
import pandas as pd
from typing import Any
import datetime
from helikite.instruments.base import Instrument
from helikite.constants import constants
import plotly.graph_objects as go
import numpy as np
import inspect
from functools import wraps
from ipywidgets import Output, VBox


def function_dependencies(required_operations: list[str] = [], use_once=False):
    """A decorator to enforce that a method can only run if the required
    operations have been completed and not rerun.

    If used without a list, the function can only run once.
    """

    def decorator(func):
        @wraps(func)  # This will preserve the original docstring and signature
        def wrapper(self, *args, **kwargs):
            # Check if the function has already been completed
            if use_once and func.__name__ in self._completed_operations:
                print(
                    f"The operation '{func.__name__}' has already been "
                    "completed and cannot be run again."
                )
                return

            functions_required = []
            # Ensure all required operations have been completed
            for operation in required_operations:
                if operation not in self._completed_operations:
                    functions_required.append(operation)

            if functions_required:
                print(
                    f"This function '{func.__name__}()' requires the "
                    "following operations first: "
                    f"{'(), '.join(functions_required)}()."
                )
                return  # Halt execution of the function if dependency missing

            # Run the function
            result = func(self, *args, **kwargs)

            # Mark the function as completed
            self._completed_operations.append(func.__name__)

            return result

        # Store dependencies and use_once information in the wrapper function
        wrapper.__dependencies__ = required_operations
        wrapper.__use_once__ = use_once

        return wrapper

    return decorator


class Cleaner:
    def __init__(
        self,
        instruments: list[Instrument],
        reference_instrument: Instrument,
        input_folder: str,
        flight_date: datetime.date,
        time_trim_from: datetime.datetime | None = None,
        time_trim_to: datetime.datetime | None = None,
        time_offset: datetime.time = datetime.time(0, 0),
    ) -> None:
        self._instruments: list[Instrument] = []  # For managing in batches
        self.input_folder: str = input_folder
        self.flight_date: datetime.date = flight_date
        self.time_trim_from: datetime.datetime | None = time_trim_from
        self.time_trim_to: datetime.datetime | None = time_trim_to
        self.time_offset: datetime.time = time_offset
        self.pressure_column: str = constants.HOUSEKEEPING_VAR_PRESSURE
        self.master_df: pd.DataFrame | None = None
        self.housekeeping_df: pd.DataFrame | None = None
        self.rolling_window_pressure_column_name: str = "pressure_rn"
        self.reference_instrument: Instrument = reference_instrument

        self._completed_operations: list[str] = []

        # Create an attribute from each instrument.name
        for instrument in instruments:
            instrument.df_raw = instrument.read_from_folder(
                input_folder, quiet=True
            )
            instrument.df = instrument.df_raw.copy(deep=True)
            instrument.date = flight_date
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

    def state(self):
        """Prints the current state of the Cleaner class in a tabular format"""

        # Create a list to store the state in a formatted way
        state_info = []

        # Add instrument information
        state_info.append(
            f"{'Instrument':<20}{'Records':<10}{'Reference':<10}"
        )
        state_info.append("-" * 40)

        for instrument in self._instruments:
            reference = (
                "Yes" if instrument == self.reference_instrument else "No"
            )
            state_info.append(
                f"{instrument.name:<20}{len(instrument.df):<10}{reference:<10}"
            )

        # Add general settings
        state_info.append("\n")
        state_info.append(f"{'Property':<25}{'Value':<30}")
        state_info.append("-" * 55)
        state_info.append(f"{'Input folder':<25}{self.input_folder:<30}")
        state_info.append(f"{'Flight date':<25}{self.flight_date}")
        state_info.append(
            f"{'Time trim from':<25}{str(self.time_trim_from):<30}"
        )
        state_info.append(f"{'Time trim to':<25}{str(self.time_trim_to):<30}")
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

        # Add rolling window pressure column info
        state_info.append(
            f"{'Rolling window column':<25}"
            f"{self.rolling_window_pressure_column_name:<30}"
        )

        # Add the functions that have been called and completed
        state_info.append("\nCompleted operations")
        state_info.append("-" * 30)

        if len(self._completed_operations) == 0:
            state_info.append("No operations have been completed.")

        for operation in self._completed_operations:
            state_info.append(f"{operation:<25}")

        # Print all the collected info in a nicely formatted layout
        print("\n".join(state_info))
        print()

    def help(self):
        """Prints available methods in a clean format"""

        print("\nThere are several methods available to clean the data:")

        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        for name, method in methods:
            if not name.startswith("_"):
                # Get method signature (arguments)
                signature = inspect.signature(method)
                func_wrapper = getattr(self.__class__, name)

                # Extract function dependencies and use_once details from the decorator
                dependencies = getattr(func_wrapper, "__dependencies__", [])
                use_once = getattr(func_wrapper, "__use_once__", False)

                # Print method name and signature
                print(f"- {name}{signature}")

                # Get the first line of the method docstring
                docstring = inspect.getdoc(method)
                if docstring:
                    first_line = docstring.splitlines()[
                        0
                    ]  # Get only the first line
                    print(f"\t{first_line}")
                else:
                    print("\tNo docstring available.")

                # Print function dependencies and use_once details
                if dependencies:
                    print(f"\tDependencies: {', '.join(dependencies)}")
                if use_once:
                    print(f"\tNote: Can only be run once")

    def _print_instruments(self) -> None:
        print(
            f"Helikite Cleaner has been initialised with "
            f"{len(self._instruments)} instruments."
        )
        for instrument in self._instruments:
            print(
                f"- Cleaner.{instrument.name}.df ({len(instrument.df)} records)",
                end="",
            )
            if instrument == self.reference_instrument:
                print(" (reference)")
            else:
                print()

    @function_dependencies(use_once=True)
    def set_pressure_column(
        self,
        column_name: str = constants.HOUSEKEEPING_VAR_PRESSURE,
    ) -> None:
        """Set the pressure column for each instrument's dataframe"""

        success = []
        errors = []
        if column_name != constants.HOUSEKEEPING_VAR_PRESSURE:
            print("Updating pressure column to", column_name)
            self.pressure_column = column_name

        for instrument in self._instruments:
            try:
                instrument.df = (
                    instrument.set_housekeeping_pressure_offset_variable(
                        instrument.df, self.pressure_column
                    )
                )
                success.append(instrument.name)
            except Exception as e:
                errors.append((instrument.name, e))

        self._print_success_errors("pressure column", success, errors)

    @function_dependencies([], use_once=True)
    def set_time_as_index(self) -> None:
        """Set the time column as the index for each instrument dataframe"""

        success = []
        errors = []

        for instrument in self._instruments:
            try:
                instrument.df = instrument.set_time_as_index(instrument.df)
                success.append(instrument.name)
            except Exception as e:
                errors.append((instrument.name, e))

        self._print_success_errors("time as index", success, errors)

    @function_dependencies(["set_time_as_index"], use_once=True)
    def correct_time(
        self,
        trim_start: pd.Timestamp | None = None,
        trim_end: pd.Timestamp | None = None,
    ) -> None:
        """Corrects the time of each instrument based on the time offset"""

        success = []
        errors = []

        # If no trim start or end, use the class's time_trim_from/to values
        if trim_start is None:
            trim_start = self.time_trim_from
        if trim_end is None:
            trim_end = self.time_trim_to

        for instrument in self._instruments:
            try:
                if self.time_trim_from is None or self.time_trim_to is None:
                    temp_df = instrument.correct_time_from_config(
                        instrument.df, trim_start, trim_end
                    )
                    if len(instrument.df) == 0:
                        print(
                            f"Warning {instrument.name}: No data in time range! "
                            "No changes have been made."
                        )
                        continue

                    instrument.df = temp_df
                    success.append(instrument.name)

                elif (
                    self.time_trim_from is not None
                    and self.time_trim_to is not None
                ):
                    # Cut each instrument's data to the selected time range
                    instrument.df = instrument.df[
                        (instrument.df.index >= self.time_trim_from)
                        & (instrument.df.index <= self.time_trim_to)
                    ]
                    print("Time trimmed for", instrument.name)
                    success.append(instrument.name)
            except Exception as e:
                errors.append((instrument.name, e))

        self._print_success_errors("time corrections", success, errors)

    @function_dependencies(["set_time_as_index"], use_once=True)
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

        self._print_success_errors("data corrections", success, errors)

    @function_dependencies(
        [
            "set_time_as_index",
            "set_pressure_column",
        ],
        use_once=False,
    )
    def plot_pressure(self) -> None:
        """Creates a plot with the pressure measurement of each instrument

        Assumes the pressure column has been set for each instrument
        """

        fig = go.Figure()

        for instrument in self._instruments:
            # Check that the column exists
            if self.pressure_column not in instrument.df.columns:
                print(
                    f"Note: {instrument.name} does not have a pressure column"
                )
                continue
            fig.add_trace(
                go.Scatter(
                    x=instrument.df.index,
                    y=instrument.df[self.pressure_column],
                    name=instrument.name,
                )
            )

        fig.update_layout(
            title="Pressure measurements",
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
        )

        fig.show()

    @function_dependencies(["set_time_as_index"], use_once=True)
    def remove_duplicates(self) -> None:
        """Remove duplicate rows from each instrument based on time index"""

        success = []
        errors = []
        for instrument in self._instruments:
            try:
                instrument.df = instrument.remove_duplicates(instrument.df)
                success.append(instrument.name)
            except Exception as e:
                errors.append((instrument.name, e))

        self._print_success_errors("duplicate removal", success, errors)

    def _print_success_errors(
        self,
        operation: str,
        success: list[str],
        errors: list[tuple[str, Any]],
    ) -> None:
        print(
            f"Set {operation} for "
            f"({len(success)}/{len(self._instruments)}): {', '.join(success)}"
        )
        print(f"Errors ({len(errors)}/{len(self._instruments)}):")
        for error in errors:
            print(f"Error ({error[0]}): {error[1]}")

    @function_dependencies(
        [
            "set_time_as_index",
            "data_corrections",
            "correct_time",
        ],
        use_once=False,
    )
    def merge_instruments(self):
        """Merges all of the dataframes from the instruments and exports them

        The dataframes are merged based on the DateTime index. The order of the
        columns is determined by the export_order attribute of the Instrument
        objects. If this attribute does not exist, the columns are placed at
        the end of the dataframe.
        """
        # Create a list of copies of the instrument dataframes with device
        # names added
        instrument_dfs = [
            instrument.add_device_name_to_columns(instrument.df.copy())
            for instrument in self._instruments
        ]

        print(
            "Added instrument names to columns in their respective dataframes "
            "(without modifying originals)."
        )

        # Sort the export columns in their numerical hierarchy order and log
        self._instruments.sort(key=lambda x: x.export_order)
        print("Instruments will be merged together with this column order:")
        for instrument in self._instruments:
            for column in instrument.export_columns:
                print(f"\t{column}")

        # Merge all the dataframes together, first df is the master
        self.master_df = instrument_dfs[0]
        for df in instrument_dfs[1:]:
            self.master_df = self.master_df.merge(
                df,
                how="outer",
                left_index=True,
                right_index=True,
            )

        # Sort rows by the date index
        self.master_df.index = pd.to_datetime(self.master_df.index)
        self.master_df.sort_index(inplace=True)

        print("The master dataframe has been created at Cleaner.master_df.")

    @function_dependencies(
        [
            "merge_instruments",
            "set_pressure_column",
            "set_time_as_index",
            "correct_time",
            "remove_duplicates",
        ],
        use_once=False,
    )
    def export_data(
        self,
        master_filename: str = "master.csv",
        housekeeping_filename: str = "housekeeping.csv",
        export_master: bool = True,
        export_housekeeping: bool = True,
    ) -> None:
        """Export the merged dataframes to CSV files"""

        # Raise error if the dataframes have not been merged
        if hasattr(self, "master_df") is False or self.master_df.empty:
            raise ValueError(
                "Dataframes have not been merged. Please run the "
                "merge_instruments() method."
            )

        # Export data and housekeeping CSV files using the column names from
        # the instruments
        if export_master:
            master_cols = [
                f"{col}"
                for instrument in self._instruments
                for col in instrument.export_columns
            ]
            self.master_df[master_cols].to_csv(master_filename)

        if export_housekeeping:
            housekeeping_cols = [
                f"{col}"
                for instrument in self._instruments
                for col in instrument.housekeeping_columns
            ]
            self.master_df[housekeeping_cols].to_csv(housekeeping_filename)

        print("\nDone.")
        print("\tDataframes have been exported to CSV files.")

        if export_master:
            print(f"\tThe file '{master_filename}' contains the data columns")
        if export_housekeeping:
            print(
                f"\tThe file '{housekeeping_filename}' contains the "
                "housekeeping columns."
            )
        print(
            "\tThe column order is determined by the export_order attribute."
        )
        print(
            "\tThe order of the instruments is determined by the export "
            "order defined in each instrument's class."
        )

    @function_dependencies(
        [
            "merge_instruments",
            "set_pressure_column",
            "set_time_as_index",
            "correct_time",
            "remove_duplicates",
        ],
        use_once=True,
    )
    def apply_rolling_window_to_pressure(self, window_size: int = 20):
        """Apply rolling window to the pressure measurements of each instrument

        Then plot the pressure measurements with the rolling window applied
        """

        fig = go.Figure()
        for instrument in self._instruments:
            if self.pressure_column not in instrument.df.columns:
                print(
                    f"Note: {instrument.name} does not have a pressure column"
                )
                continue
            instrument.df[self.rolling_window_pressure_column_name] = (
                instrument.df[self.pressure_column]
                .rolling(window=window_size)
                .mean()
            )

            fig.add_trace(
                go.Scatter(
                    x=instrument.df.index,
                    y=instrument.df[self.rolling_window_pressure_column_name],
                    name=f"{instrument.name}_rolling_window",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=instrument.df.index,
                    y=instrument.df[self.pressure_column],
                    name=instrument.name,
                )
            )

        fig.update_layout(
            title="Pressure measurements",
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
        )

        fig.show()

    @function_dependencies(
        [
            "set_pressure_column",
            "set_time_as_index",
            "data_corrections",
        ],
        use_once=False,
    )
    def define_flight_times(self):
        """Creates a plot to select the start and end of the flight

        Uses the pressure measurements of the reference instrument to select
        the start and end of the flight. The user can click on the plot to
        select the points.

        The time of the selected points will be used to trim the dataframes.
        """

        # Create a figure widget for interactive plotting
        fig = go.FigureWidget()
        out = Output()
        # out.append_stdout('Output appended with append_stdout')
        out.append_stdout(f"\nStart time: {self.time_trim_from}\n")
        out.append_stdout(f"End time: {self.time_trim_to}\n")
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
                if (self.time_trim_from is None) or (
                    self.time_trim_from is not None
                    and self.time_trim_to is not None
                ):
                    # Set the start time, and reset the end time
                    self.time_trim_from = selected_x
                    self.time_trim_to = None
                    print(f"Start time: {self.time_trim_from}")
                    print(f"End time: {self.time_trim_to}")
                    print("Click to set the end time.")
                elif (
                    self.time_trim_from is not None
                    and self.time_trim_to is None
                ):
                    # Set the end time
                    self.time_trim_to = selected_x
                    print(f"Start time: {self.time_trim_from}")
                    print(f"End time: {self.time_trim_to}")
                    print(
                        "Click again if you wish to reset the times and set "
                        "a new start time"
                    )
                else:
                    print("Something went wrong with the time selection.")

            # Update the plot if self.time_trim_from and self.time_trim_to
            # have been set or modified
            if (
                self.time_trim_from is not None
                and self.time_trim_to is not None
            ):
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
                    x0=self.time_trim_from,
                    x1=self.time_trim_to,
                    fillcolor="rgba(0, 128, 0, 0.25)",
                    layer="below",
                    line_width=0,
                )

        # Add the initial time range to the plot
        if self.time_trim_from is not None and self.time_trim_to is not None:
            # Add a new vrect
            fig.add_vrect(
                x0=self.time_trim_from,
                x1=self.time_trim_to,
                fillcolor="rgba(0, 128, 0, 0.25)",
                layer="below",
                line_width=0,
            )
        # Iterate through instruments to plot pressure data
        for instrument in self._instruments:
            # Check if the pressure column exists in the instrument dataframe
            if self.pressure_column not in instrument.df.columns:
                print(
                    f"Note: {instrument.name} does not have a pressure column"
                )
                continue

            # Add pressure trace to the plot. If it is the reference
            # instrument, plot it with a thicker/darker line, otherwise,
            # plot it lightly with some transparency.
            if instrument == self.reference_instrument:
                fig.add_trace(
                    go.Scatter(
                        x=instrument.df.index,
                        y=instrument.df[self.pressure_column],
                        name=instrument.name,
                        line=dict(width=2, color="red"),
                        opacity=1,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=instrument.df.index,
                        y=instrument.df[self.pressure_column],
                        name=instrument.name,
                        line=dict(width=1, color="grey"),
                        opacity=0.25,
                        hoverinfo="skip",
                    )
                )

        # Attach the callback to all traces
        for trace in fig.data:
            # Only allow the reference instrument to be clickable
            if trace.name == self.reference_instrument.name:
                trace.on_click(select_point_callback)
                print(f"Callback attached to trace: {trace.name}")

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

    def _crosscorr(self, datax, datay, lag=0):
        """Lag-N cross correlation."""
        return datax.corr(datay.shift(lag))

    def _find_time_lag(self, max_lag=180):
        """Find time lag between instrument and reference pressure"""
        if self.pressure_column not in self.reference_instrument.df.columns:
            raise KeyError(
                f"Pressure column '{self.pressure_column}' not found in "
                "reference instrument '{self.reference_instrument.name}'"
            )

        ref_pressure = self.reference_instrument.df[
            self.pressure_column
        ].dropna()
        ref_pressure = ref_pressure.sort_index()

        lag_results = {}

        for instrument in self._instruments:
            if instrument == self.reference_instrument:
                continue

            # Check if the pressure column exists in this instrument
            if self.pressure_column not in instrument.df.columns:
                print(
                    f"Skipping {instrument.name}: '{self.pressure_column}' "
                    "column not found."
                )
                continue

            instrument_pressure = instrument.df[self.pressure_column].dropna()
            instrument_pressure = instrument_pressure.sort_index()

            # Align the indexes between the reference and current instrument
            common_index = ref_pressure.index.intersection(
                instrument_pressure.index
            )
            if len(common_index) == 0:
                print(
                    f"No common data between {self.reference_instrument.name} "
                    "and {instrument.name}. Skipping..."
                )
                continue

            ref_pressure_aligned = ref_pressure.loc[common_index]
            inst_pressure_aligned = instrument_pressure.loc[common_index]

            if (
                len(ref_pressure_aligned) == 0
                or len(inst_pressure_aligned) == 0
            ):
                print(
                    f"No valid overlapping data for {instrument.name}. "
                    "Skipping..."
                )
                continue

            # Compute cross-correlation for different lags
            lags = range(-max_lag, max_lag + 1)
            corrs = [
                self._crosscorr(
                    ref_pressure_aligned, inst_pressure_aligned, lag
                )
                for lag in lags
            ]
            best_lag = lags[
                np.nanargmax(corrs)
            ]  # Find the lag with max correlation, ignoring NaNs

            print(f"Best lag for {instrument.name}: {best_lag} seconds")

            # Store the lag
            lag_results[instrument.name] = best_lag

        return lag_results

    @function_dependencies(
        [
            "set_time_as_index",
            "data_corrections",
            "set_pressure_column",
        ],
        use_once=False,
    )
    def correct_time_and_pressure(self, max_lag=180):
        """Correct time and pressure for each instrument based on time lag."""

        lag_results = self._find_time_lag(max_lag=max_lag)
        print("Lag results:", lag_results)

        ref_pressure = self.reference_instrument.df[
            self.pressure_column
        ].dropna()

        for instrument in self._instruments:
            if instrument.name in lag_results:
                best_lag = lag_results[instrument.name]

                # Shift time index
                instrument.df.index = instrument.df.index.shift(
                    -best_lag, freq="s"
                )
                print(f"Shifted {instrument.name} by {best_lag} seconds")

                # Align pressures based on the reference instrument
                inst_pressure = instrument.df[self.pressure_column].dropna()

                # Get the common time index again after shifting
                common_index = ref_pressure.index.intersection(
                    instrument.df.index
                )

                if len(common_index) == 0:
                    print(
                        "No overlapping data between "
                        f"{self.reference_instrument.name} and "
                        f"{instrument.name} after shifting. Skipping pressure "
                        "alignment."
                    )
                    continue

                # Subtract the mean difference in pressure during the common
                # time window
                ref_pressure_aligned = ref_pressure.loc[common_index]
                inst_pressure_aligned = inst_pressure.loc[common_index]

                # Calculate the mean difference in pressure
                mean_diff = (
                    inst_pressure_aligned.mean() - ref_pressure_aligned.mean()
                )

                # Adjust the instrument's pressure by subtracting the mean
                # difference
                instrument.df[self.pressure_column] = (
                    instrument.df[self.pressure_column] - mean_diff
                )
                print(
                    f"Adjusted pressure for {instrument.name} by "
                    f"{mean_diff:.2f} hPa"
                )

import pandas as pd
from typing import Any
import datetime
from helikite.instruments.base import Instrument
from helikite.constants import constants
from helikite.processing.post import crosscorrelation
import plotly.graph_objects as go
import numpy as np
import inspect
from functools import wraps
from ipywidgets import Output, VBox
import psutil
from itertools import cycle
import plotly.colors

parent_process = psutil.Process().parent().cmdline()[-1]


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
        time_takeoff: datetime.datetime | None = None,
        time_landing: datetime.datetime | None = None,
        time_offset: datetime.time = datetime.time(0, 0),
    ) -> None:
        self._instruments: list[Instrument] = []  # For managing in batches
        self.input_folder: str = input_folder
        self.flight_date: datetime.date = flight_date
        self.time_takeoff: datetime.datetime | None = time_takeoff
        self.time_landing: datetime.datetime | None = time_landing
        self.time_offset: datetime.time = time_offset
        self.pressure_column: str = constants.HOUSEKEEPING_VAR_PRESSURE
        self.master_df: pd.DataFrame | None = None
        self.housekeeping_df: pd.DataFrame | None = None
        self.reference_instrument: Instrument = reference_instrument

        self._completed_operations: list[str] = []

        # Create an attribute from each instrument.name
        for instrument in instruments:
            instrument.df_raw = instrument.read_from_folder(
                input_folder, quiet=True
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

                # Extract func dependencies and use_once details from decorator
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
                    print("\tNote: Can only be run once")

    def _print_instruments(self) -> None:
        print(
            f"Helikite Cleaner has been initialised with "
            f"{len(self._instruments)} instruments."
        )
        for instrument in self._instruments:
            print(
                f"- Cleaner.{instrument.name}.df "
                f"({len(instrument.df)} records)",
                end="",
            )
            if instrument == self.reference_instrument:
                print(" (reference)")
            else:
                print()

    @function_dependencies(use_once=True)
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
            "set_pressure_column",
        ],
        use_once=False,
    )
    def merge_instruments(
        self, tolerance_seconds: int = 1, remove_duplicates: bool = True
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
                self.reference_instrument.df.index.dtype
            )

        print("Using merge_asof to align and merge instrument dataframes.")

        # Start with the reference instrument dataframe
        self.master_df = self.reference_instrument.df.copy()
        self.master_df.columns = [
            f"{self.reference_instrument.name}_{col}"
            for col in self.master_df.columns
        ]

        # Merge all other dataframes with merge_asof
        for instrument in self._instruments:
            if instrument == self.reference_instrument:
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

    @function_dependencies(
        [
            "merge_instruments",
            "remove_duplicates",
        ],
        use_once=False,
    )
    def export_data(
        self,
        filename: str | None = None,
    ) -> None:
        """Export all data columns from all instruments to a single CSV file"""

        # Raise error if the dataframes have not been merged
        if self.master_df is None or self.master_df.empty:
            raise ValueError(
                "Dataframes have not been merged. Please run the "
                "merge_instruments() method."
            )

        if filename is None:
            # Include the date and time of the first row of the reference
            # instrument in the filename
            filename = f'level0_{self.master_df.index[0].to_pydatetime().strftime("%Y-%m-%dT%H-%M")}.csv'  # noqa

        # Combine all columns from all instruments
        all_columns = list(self.master_df.columns)
        self.master_df[all_columns].to_csv(filename)

        print(f"\nDone. The file '{filename}' contains all instrument data.")

    @function_dependencies(
        [
            "set_pressure_column",
            "set_time_as_index",
        ],
        use_once=False,
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
            if instrument == self.reference_instrument:
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
            if trace.name == self.reference_instrument.name:
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

    @function_dependencies(
        [
            "set_time_as_index",
            "data_corrections",
            "set_pressure_column",
        ],
        use_once=False,
    )
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
            instrument.df.index = instrument.df.index.astype("datetime64[s]")

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
            self.reference_instrument.df.loc[
                (
                    self.reference_instrument.df[
                        self.reference_instrument.pressure_column
                    ]
                    > reference_pressure_thresholds[1]
                )
                | (
                    self.reference_instrument.df[
                        self.reference_instrument.pressure_column
                    ]
                    < reference_pressure_thresholds[0]
                ),
                self.reference_instrument.pressure_column,
            ] = np.nan
            self.reference_instrument.df[
                self.reference_instrument.pressure_column
            ] = (
                self.reference_instrument.df[
                    self.reference_instrument.pressure_column
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

        # 0 is ignore because it's at the beginning of the df_corr, not
        # in the range
        rangelag = [i for i in range(-max_lag, max_lag + 1) if i != 0]

        self.df_pressure = self.reference_instrument.df[
            [self.reference_instrument.pressure_column]
        ].copy()
        self.df_pressure.rename(
            columns={
                self.reference_instrument.pressure_column: self.reference_instrument.name  # noqa
            },
            inplace=True,
        )

        for instrument in self._instruments:
            if instrument == self.reference_instrument:
                # We principally use the ref for this, don't merge with itself
                continue

            if instrument.pressure_column in instrument.df.columns:
                df = instrument.df[[instrument.pressure_column]].copy()
                df.index = df.index.astype(
                    self.reference_instrument.df.index.dtype
                )

                df.rename(
                    columns={instrument.pressure_column: instrument.name},
                    inplace=True,
                )

                self.df_pressure = pd.merge_asof(
                    self.df_pressure,
                    df,
                    left_index=True,
                    right_index=True,
                )

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
                #     if instrument == self.reference_instrument:
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
            lag=max_lag,
            NON_DER=[self.reference_instrument.name],
        )
        df_new = df_new.dropna()
        self.df_corr = df_new.corr()

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

            if instrument == self.reference_instrument:
                print("\tSkipping reference instrument\n")
                continue
            if instrument.pressure_column in instrument.df.columns:
                instrument.corr_df = crosscorrelation.df_findtimelag(
                    self.df_corr, rangelag, instname=f"{instrument.name}_"
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
                        self.reference_instrument.df,
                        instrument.corr_max_idx,
                        f"{instrument.name}_",
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
                self.reference_instrument.df,
                primary_instrument.corr_max_idx,
                f"{secondary_instrument.name}_",
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

import dataclasses
import datetime
from numbers import Number

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pydantic import BaseModel

from helikite.classes.output_schemas import OutputSchema, FlightProfileVariable, FlightProfileVariableShade, Level
from helikite.instruments.base import filter_columns_by_instrument, Instrument


def create_level1_dataframe(df: pd.DataFrame, output_schema: OutputSchema):
    selected_columns = []
    processed_columns = set()

    for instrument in output_schema.instruments:
        instr_cols_all = filter_columns_by_instrument(df, instrument)
        if instrument.final_columns is not None:
            instr_cols_final = [f"{instrument.name}_{col}" for col in instrument.final_columns]
        else:
            instr_cols_final = instr_cols_all

        selected_columns += instr_cols_final
        processed_columns = processed_columns.union(instr_cols_all)

    non_instr_columns_to_skip = ["Pressure_ground", "Temperature_ground"]
    non_instr_columns = [col for col in df.columns
                         if col not in processed_columns and col not in non_instr_columns_to_skip]

    selected_columns = non_instr_columns + selected_columns

    return df[selected_columns].copy()


def rename_columns(df: pd.DataFrame, output_schema: OutputSchema, reference_instrument: Instrument):
    """
    Renames columns of the input DataFrame according to predefined rules and instrument-specific rules.
    See `Instrument.rename_dict`

    Parameters:
        df (pd.DataFrame): The DataFrame with columns to be renamed.
        output_schema (OutputSchema): The OutputSchema object containing the instruments.
        reference_instrument (Instrument): Reference instrument to take pressure values from.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    df_renamed = df.rename(columns=_build_rename_dict(output_schema, reference_instrument))

    return df_renamed


def _build_rename_dict(output_schema: OutputSchema, reference_instrument: Instrument):
    rename_dict = {
        'DateTime': 'datetime',
        'latitude_dd': 'Lat',
        'longitude_dd': 'Long',
        f'{reference_instrument.name}_pressure': 'P',
        'Average_Temperature': 'TEMP',
        'Average_RH': 'RH',
    }
    for instrument in output_schema.instruments:
        rename_dict = rename_dict | instrument.rename_dict

    return rename_dict


def round_flightnbr_campaign(df: pd.DataFrame, metadata: BaseModel, output_schema: OutputSchema, decimals: int):
    """
    Round numeric columns of the DataFrame with special handling for 'Lat' and 'Long',
    and add columns for flight number and campaign.

    Parameters:
        df (pd.DataFrame): The DataFrame to be rounded and modified.
        metadata (object): Metadata object containing the 'flight' attribute.
        output_schema (OutputSchema): The OutputSchema object containing the campaign name.
        decimals (int, optional): The number of decimal places to round to (default is 2).

    Returns:
        pd.DataFrame: The rounded and modified DataFrame with additional columns.
    """
    # Columns to exclude from default rounding
    exclude_cols = ['Lat', 'Long']

    # Round all numeric columns except 'Lat' and 'Long'
    numeric_cols = df.select_dtypes(include='number').columns
    round_cols = [col for col in numeric_cols if col not in exclude_cols]
    df[round_cols] = df[round_cols].round(decimals)

    # Now round 'Lat' and 'Long' to 4 decimals
    for col in exclude_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    # Convert 'WindDir' to integer if it exists
    if 'WindDir' in df.columns:
        df['WindDir'] = df['WindDir'].astype('Int64')

    # Add metadata columns
    df = pd.concat([
        df,
        pd.DataFrame({'flight_nr': metadata.flight, 'campaign': output_schema.campaign}, index=df.index)
    ], axis=1)


    return df



def fill_msems_takeoff_landing(df, metadata, time_window_seconds):
    """
    Fill missing values in mSEMS columns at takeoff and landing times using nearby values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DateTimeIndex where filling should occur.
    metadata : object
        An object containing `takeoff_time` and `landing_time` attributes.
    time_window_seconds : int, optional
        Number of seconds before/after to search for replacement values (default: 90).
    """

    # Convert to Timestamps
    takeoff_time = pd.to_datetime(metadata.takeoff_time)
    landing_time = pd.to_datetime(metadata.landing_time)

    # Select relevant columns
    msems_cols = [
        col for col in df.columns
        if (col.startswith("msems_scan_") or col.startswith("msems_inverted_"))
        and col not in ["msems_scan_DateTime", "msems_inverted_DateTime"]
    ]

    for event_time, label in [(takeoff_time, "takeoff_time"), (landing_time, "landing_time")]:
        if event_time in df.index:
            row = df.loc[event_time, msems_cols]

            if row.isna().any():
                window_start = event_time - pd.Timedelta(seconds=time_window_seconds)
                window_end = event_time + pd.Timedelta(seconds=time_window_seconds)
                window_df = df.loc[window_start:window_end, msems_cols]

                filled_from_indices = []

                for col in msems_cols:
                    if pd.isna(df.at[event_time, col]):
                        valid_values = window_df[col].dropna()
                        if not valid_values.empty:
                            df.at[event_time, col] = valid_values.iloc[0]
                            filled_from_indices.append(valid_values.index[0])

                if filled_from_indices:
                    earliest_fill = min(filled_from_indices)
                    print(f"Filled mSEMS columns at {label} ({event_time}) using values from {earliest_fill}")
                else:
                    print(f"No valid values found in ±{time_window_seconds} seconds of {label} ({event_time}).")
            else:
                print(f"No fill needed at {label} ({event_time}).")
        else:
            print(f"{label} ({event_time}) not in index.")


def flight_profiles(df: pd.DataFrame, reference_instrument: Instrument, level: Level, output_schema: OutputSchema,
                    variables: list[FlightProfileVariable] | None, fig_title=None):
    # Columns in df might have been already renamed
    rename_dict = _build_rename_dict(output_schema, reference_instrument)
    variables = variables if variables is not None else output_schema.flight_profile_variables
    variables = variables.copy()
    for i, variable in enumerate(variables):
        if variable.column_name not in df.columns:
            variable = dataclasses.replace(variable, column_name=rename_dict[variable.column_name])
            variables[i] = variable

    # Find the index of the maximum altitude
    max_altitude_index = df['Altitude'].idxmax()
    max_altitude_pos = df.index.get_loc(max_altitude_index)

    # Split DataFrame into ascent and descent
    df_up = df.iloc[:max_altitude_pos + 1]
    df_down = df.iloc[max_altitude_pos + 1:]

    num_subplots = (len(variables) + 1) // 2
    fig, axes = plt.subplots(1, num_subplots + 1,
                             figsize=(4 * num_subplots, 6),
                             gridspec_kw={'width_ratios': [1] * num_subplots + [0.3]})
    plt.subplots_adjust(wspace=0.3)

    # repeat each axis twice, excluding the last one
    axes_doubled = sum(([ax, ax] for ax in axes[:-1]), [])

    for i in range(len(axes_doubled)):
        if i % 2 == 1:
            ax = axes_doubled[i - 1].twiny()
            axes_doubled[i] = ax

            # Position second x-axis ticks below the axis for all twinned axes_doubled
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_label_position('bottom')
            ax.spines['bottom'].set_position(('outward', 40))

    legend_lines = [
        Line2D([0], [0], color='darkgrey', lw=4, label='Ascent'),
        Line2D([0], [0], color='lightgrey', lw=4, label='Descent'),
    ]

    for shade_config in output_schema.flight_profile_shades:
        if shade_config.name in df.columns:
            axes_doubled_to_shade = [ax for ax, variable in zip(axes_doubled, variables)
                                     if shade_config.name in variable.shade_flags]
            if len(axes_doubled_to_shade) == 0:
                continue

            for i, df_partial in enumerate([df_up, df_down]):
                shade_flagged(
                    shade_config, axes_doubled_to_shade, df_partial, level,
                    shade_coord="x",
                    other_coord_name="Altitude",
                )

                if i == 0:
                    legend_lines.append(Patch(label=shade_config.label, **shade_config.span_kwargs))

    # Plot ascent data
    for ax, variable in zip(axes_doubled, variables):
        ax.plot(df_up[variable.column_name], df_up["Altitude"], alpha=variable.alpha_ascent, **variable.plot_kwargs)
        ax.plot(df_down[variable.column_name], df_down["Altitude"], alpha=variable.alpha_descent, **variable.plot_kwargs)

    for ax, var in zip(axes_doubled, variables):
        divider = var.x_divider or 1
        (x_min, x_max), divider = _get_series_bounds(df[var.column_name], divider, var.x_min, var.x_max)

        ax.set_xlim(x_min, x_max)
        if divider is not None and x_min is not None and x_max is not None:
            ax.set_xticks(np.arange(x_min, x_max + min(divider, 1), divider))

    # Y axis minor ticks, grid and limits on main axes
    for j in range(num_subplots):
        axes[j].yaxis.set_minor_locator(ticker.MultipleLocator(base=50))
        axes[j].set_axisbelow(True)
        axes[j].grid(which='major', linestyle='--', linewidth=0.5, color='gray')
        axes[j].grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        axes[j].set_ylim(-10, df['Altitude'].max() + 10)

    # Label settings
    axes[0].set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
    for i in range(1, num_subplots):
        axes[i].set_yticklabels('')

    for ax, variable in zip(axes_doubled, variables):
        if variable.x_label is not None:
            color = variable.plot_kwargs.get("color", "gray")
            ax.set_xlabel(variable.x_label, color=color, fontweight='bold')
            ax.tick_params(axis='x', labelcolor=color, labelsize=11)

    # Hide last subplot axis and add legend
    axes[-1].axis('off')
    axes[-1].legend(
        handles=legend_lines,
        loc='upper right',
        bbox_to_anchor=(1, 1.02),
        frameon=True,
        fancybox=True,
        fontsize=12
    )

    # Figure title
    if fig_title is None:
        fig_title = 'Flight X [Level 1]'

    fig.suptitle(
        fig_title,
        fontsize=16,
        fontweight='bold',
        y=0.98,
        x=0.51
    )

    plt.tight_layout()
    plt.show()
    return fig


def _get_series_bounds(
    x: pd.Series,
    default_divider: Number,
    default_min: Number | None,
    default_max: Number | None
) -> tuple[tuple[Number, Number], Number]:
    if x.isna().all().all():
        return (default_min, default_max), default_divider

    min_bound = x.min() if default_min is None else default_min
    max_bound = x.quantile(0.99) * 1.1 if default_max is None else default_max

    divider = default_divider
    while max_bound - min_bound > divider * 6:
        divider = ((max_bound - min_bound) // 6)
        divider = divider - (divider % default_divider) + default_divider

    min_bound = min_bound - (min_bound % divider)
    max_bound = max_bound + (-max_bound % divider)

    return (min_bound, max_bound), divider


def plot_size_distributions(df: pd.DataFrame, level: Level, output_schema: OutputSchema, fig_title: str,
                            time_start: datetime.datetime | None, time_end: datetime.datetime | None,
                            freq: str = "1s", min_loc_interval: int = 10):
    # TEMPORAL PLOT OF FLIGHT with POPS and mSEMS HEAT MAPS
    instruments_with_size_distr = [instr for instr in output_schema.instruments if instr.has_size_distribution]
    nrows = len(instruments_with_size_distr) + 1

    # Create figure with 3 subplots, sharing the same x-axis
    fig, axes = plt.subplots(nrows, ncols=1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    plt.subplots_adjust(hspace=0.1)

    """ SET THE TITLE OF THE PLOT (FLIGHT N° with DATE_X) """
    # 'i' will automatically be replaced by the set flight number
    # '_X' has to be changed manually in function of the flight index of the day (A, B, ...)
    fig.suptitle(fig_title, fontsize=16, fontweight='bold', y=0.91)

    ### SUBPLOT 1: Altitude vs. Time
    ax1 = axes[0]
    ax1.plot(df.index, df['Altitude'], color='black', linewidth=2, label='Altitude')

    ax1.set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=11)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.set_ylim(-40, df['Altitude'].max() * 1.04)

    freq_detected = pd.infer_freq(df.index)
    if freq_detected is None:
        freq_detected = freq
    timedelta = pd.to_timedelta(pd.tseries.frequencies.to_offset(freq_detected))

    for shade_config in output_schema.flight_profile_shades:
        if shade_config.name in df.columns:
            shade_flagged(shade_config, [ax1], df, level, timedelta=timedelta)

            if shade_config.line_name is not None:
                values = df[shade_config.line_name]
                v_max = values.max() if not values.isna().all() else 1.0
                line_label, line_color = shade_config.line_kwargs["label"], shade_config.line_kwargs.get("color", None)

                ax1_twin = ax1.twinx()
                ax1_twin.plot(df.index, values, **shade_config.line_kwargs)
                ax1_twin.tick_params(axis="y", labelsize=11, colors=line_color)
                ax1_twin.set_ylabel(line_label, color=line_color, fontsize=12, fontweight='bold')
                ax1_twin.set_ylim(0.0, v_max * 1.1)

    # Optional: Clean legend (avoid duplicates)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=10)

    verbose = True
    for i, instrument in enumerate(instruments_with_size_distr):
        instrument.plot_distribution(df, verbose, time_start, time_end, subplot=(fig, axes[i + 1]))

    # X-axis formatting for all subplots
    span = df.index.max() - df.index.min()

    ax_last = axes[-1]
    if span <= pd.Timedelta(days=1):
        ax_last.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_last.xaxis.set_major_locator(mdates.MinuteLocator(interval=min_loc_interval))
    else:
        ax_last.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax_last.xaxis.set_major_locator(mdates.HourLocator(interval=min_loc_interval // 24))

    ax_last.set_xlabel('Time', fontsize=13, fontweight='bold', labelpad=10)
    ax_last.tick_params(axis='x', rotation=90, labelsize=11)

    xlim = ax_last.get_xlim()
    xlim = (time_start if time_start is not None else xlim[0], time_end if time_end is not None else xlim[1])
    ax_last.set_xlim(xlim)

    plt.show()
    return fig


def shade_flagged(shade_config: FlightProfileVariableShade, axes: list[plt.Axes], df: pd.DataFrame, level: Level,
                  shade_coord: str = "y", other_coord_name: str | None = None,
                  timedelta: pd.Timedelta = pd.Timedelta(seconds=1)):
    name = shade_config.name
    mask = df[[name]].copy()
    mask[name] = shade_config.condition(level, mask[name]).astype("Int64")

    # If NaNs are between shaded areas, fill them in, otherwise, do not shade
    nans_between_shaded = mask[name].ffill().eq(1) & mask[name].bfill().eq(1)
    nans_between_shaded = nans_between_shaded.fillna(False)
    mask[name] = mask[name].where(~nans_between_shaded, 1)
    mask[name] = mask[name].fillna(0)

    if mask.empty:
        return

    starts = mask[(mask[name] != mask.shift(1)[name]).fillna(True)].index
    ends = mask[(mask[name] != mask.shift(-1)[name]).fillna(True)].index

    for start, end in zip(starts, ends):
        if end - start <= timedelta:
            continue

        if mask.loc[start, name] == 1:
            v_min = start if other_coord_name is None else df.loc[start:end, other_coord_name].min()
            v_max = end if other_coord_name is None else df.loc[end, other_coord_name].max()

            if pd.isna(v_min) or pd.isna(v_max):
                continue

            for ax in axes:
                if shade_coord == "y":
                    ax.axvspan(v_min, v_max, label=shade_config.label, **shade_config.span_kwargs)
                elif shade_coord == "x":
                    ax.axhspan(v_min, v_max, label=shade_config.label, **shade_config.span_kwargs)
                else:
                    raise ValueError(f"Invalid `flag_coord`: {shade_coord}")


def filter_data(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Filter_position
    ax1.plot(df.index, df['filter_cur_pos'], color='tab:blue', label='Filter Position')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Filter Position', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for Filter_flow
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['filter_pump_pw'], color='tab:red', label='Pump Power')
    ax2.set_ylabel('Pump Power', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Title and legend
    #fig.suptitle('Filter Position and Flow vs Time')
    fig.tight_layout()
    plt.show()

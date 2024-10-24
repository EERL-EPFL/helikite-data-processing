import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from helikite.constants import constants
import numpy as np
from helikite.processing.helpers import reduce_column_to_single_unique_value
from helikite import instruments
import os


logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


def plot_scatter_from_variable_list_by_index(
    df: pd.DataFrame,
    title: str,
    variables: List[str],
) -> go.Figure:

    fig = go.Figure()
    for var in variables:
        if var is not None:
            fig.add_trace(
                go.Scattergl(
                    x=df.index,
                    y=df[var],
                    name=var,
                    mode="markers",
                    marker=dict(
                        size=constants.PLOT_MARKER_SIZE,
                    ),
                )
            )
    fig.update_yaxes(
        mirror=True, showline=True, linecolor="black", linewidth=2
    )
    fig.update_xaxes(
        mirror=True, showline=True, linecolor="black", linewidth=2
    )
    fig.update_layout(
        **constants.PLOT_LAYOUT_COMMON, title=title, showlegend=True
    )

    return fig


def generate_grid_plot(
    df: pd.DataFrame,
    all_instruments: List[instruments.Instrument],
    altitude_col: str = "flight_computer_Altitude",
    resample_seconds: int | None = None,
) -> go.Figure:
    """Generates a 4x3 plot of quicklooks variables from several instruments

    Resample values to a time interval in seconds is possible, to decrease
    the total count of points in the plot. This is useful for large datasets
    where the plot is too slow to render, or too many points to be useful.

    The resampling will not happen on the wind direction due to the nature
    of the data. The wind direction is a circular variable, and resampling
    will not work as expected.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data to plot
    all_instruments : List[instruments.Instrument]
        A list of all instruments to plot
    altitude_col : str, optional
        The column name of the altitude variable, by default
        "flight_computer_Altitude"
    resample_seconds : int, optional
        The number of seconds to resample the data to, by default None
    """

    colors = generate_normalised_colours(df)
    fig = make_subplots(rows=3, cols=4, shared_yaxes=False)

    # Always plot wind degrees with original data (not resampled)
    if instruments.smart_tether in all_instruments:
        fig.add_trace(
            go.Scattergl(
                x=df["smart_tether_Wind (degrees)"],
                y=df[altitude_col],
                name="Wind direction (Smart Tether)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    showscale=False,
                ),
            ),
            row=1,
            col=4,
        )

    # Resample the data, and replace the dataframe with the resampled version
    if resample_seconds is not None:
        logger.info(
            "Resampling grid-plot data at the mean of "
            f"{resample_seconds} seconds "
        )
        df = df.resample(f"{int(resample_seconds)}S").mean(numeric_only=True)
        colors = generate_normalised_colours(df)

    # Add the same temperature plots to all three rows
    for row_id in range(1, 4):

        fig.add_trace(
            go.Scattergl(
                x=df["flight_computer_TEMP1"],
                y=df[altitude_col],
                name="Temperature1 (Flight Computer)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="circle-open",
                ),
            ),
            row=row_id,
            col=1,
        )

        fig.add_trace(
            go.Scattergl(
                x=df["flight_computer_TEMP2"],
                y=df[altitude_col],
                name="Temperature2 (Flight Computer)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="x-open",
                ),
            ),
            row=row_id,
            col=1,
        )

        fig.add_trace(
            go.Scattergl(
                x=df["flight_computer_TEMPsamp"],
                y=df[altitude_col],
                name="TemperatureSamp (Flight Computer)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="square-x-open",
                ),
            ),
            row=row_id,
            col=1,
        )

        if instruments.smart_tether in all_instruments:
            fig.add_trace(
                go.Scattergl(
                    x=df["smart_tether_T (deg C)"],
                    y=df[altitude_col],
                    name="Temperature (Smart Tether)",
                    mode="markers",
                    marker=dict(
                        color=colors,
                        size=constants.PLOT_MARKER_SIZE,
                        line_width=2,
                        showscale=False,
                        symbol="diamond-open",
                    ),
                ),
                row=row_id,
                col=1,
            )

    fig.add_trace(
        go.Scattergl(
            x=df["flight_computer_RH1"],
            y=df[altitude_col],
            name="Relative Humidity1 (Flight Computer)",
            mode="markers",
            marker=dict(
                color=colors,
                size=constants.PLOT_MARKER_SIZE,
                line_width=2,
                showscale=False,
                symbol="circle-open",
            ),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scattergl(
            x=df["flight_computer_RH2"],
            y=df[altitude_col],
            name="Relative Humidity2 (Flight Computer)",
            mode="markers",
            marker=dict(
                color=colors,
                size=constants.PLOT_MARKER_SIZE,
                line_width=2,
                showscale=False,
                symbol="x-open",
            ),
        ),
        row=1,
        col=2,
    )

    if instruments.pops in all_instruments:
        fig.add_trace(
            go.Scattergl(
                x=df["pops_PartCon_186"],
                y=df[altitude_col],
                name="PartCon_186 (POPS)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    showscale=False,
                ),
            ),
            row=2,
            col=2,
        )

    fig.add_trace(
        go.Scattergl(
            x=df["flight_computer_CO2"],
            y=df[altitude_col],
            name="CO2 (Flight Computer)",
            mode="markers",
            marker=dict(
                color=colors, size=constants.PLOT_MARKER_SIZE, showscale=False
            ),
        ),
        row=2,
        col=3,
    )

    # Add smart tether data if it exists
    if instruments.smart_tether in all_instruments:
        fig.add_trace(
            go.Scattergl(
                x=df["smart_tether_Wind (m/s)"],
                y=df[altitude_col],
                name="Wind speed (Smart Tether)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    showscale=False,
                ),
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scattergl(
                x=df["smart_tether_%RH"],
                y=df[altitude_col],
                name="Relative Humidity (Smart Tether)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="diamond-open",
                ),
            ),
            row=1,
            col=2,
        )

    # Add STAP data if it exists
    if instruments.stap in all_instruments:
        fig.add_trace(
            go.Scattergl(
                x=df["stap_sigmab_smth"],
                y=df[altitude_col],
                name="sigmab_smth (STAP)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="circle-open",
                ),
            ),
            row=2,
            col=4,
        )

        fig.add_trace(
            go.Scattergl(
                x=df["stap_sigmag_smth"],
                y=df[altitude_col],
                name="sigmag_smth (STAP)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="x-open",
                ),
            ),
            row=2,
            col=4,
        )

        fig.add_trace(
            go.Scattergl(
                x=df["stap_sigmar_smth"],
                y=df[altitude_col],
                name="sigmar_smth (STAP)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="diamond-open",
                ),
            ),
            row=2,
            col=4,
        )

    # Add *RAW* STAP data if it exists )
    if instruments.stap_raw in all_instruments:
        fig.add_trace(
            go.Scattergl(
                x=df["stap_raw_invmm_b"],
                y=df[altitude_col],
                name="invmm_b (STAP Raw)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="circle-open",
                ),
            ),
            row=2,
            col=4,
        )

        fig.add_trace(
            go.Scattergl(
                x=df["stap_raw_invmm_g"],
                y=df[altitude_col],
                name="invmm_g (STAP Raw)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="x-open",
                ),
            ),
            row=2,
            col=4,
        )

        fig.add_trace(
            go.Scattergl(
                x=df["stap_raw_invmm_r"],
                y=df[altitude_col],
                name="invmm_r (STAP Raw)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    line_width=2,
                    showscale=False,
                    symbol="diamond-open",
                ),
            ),
            row=2,
            col=4,
        )

    if instruments.pico in all_instruments:
        fig.add_trace(
            go.Scattergl(
                x=df["pico_CO (ppm)"],
                y=df[altitude_col],
                name="CO (Pico)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    showscale=False,
                ),
            ),
            row=3,
            col=2,
        )

        fig.add_trace(
            go.Scattergl(
                x=df["pico_N2O (ppm)"],
                y=df[altitude_col],
                name="N2O (Pico)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    showscale=False,
                ),
            ),
            row=3,
            col=3,
        )

    if instruments.ozone_monitor in all_instruments:
        fig.add_trace(
            go.Scattergl(
                x=df["ozone_ozone"],
                y=df[altitude_col],
                name="Ozone (Ozone)",
                mode="markers",
                marker=dict(
                    color=colors,
                    size=constants.PLOT_MARKER_SIZE,
                    showscale=False,
                ),
            ),
            row=3,
            col=4,
        )

    # Give each plot a border and white background
    for row in [1, 2, 3]:
        for col in [1, 2, 3, 4]:
            fig.update_yaxes(
                row=row,
                col=col,
                mirror=True,
                showline=True,
                linecolor="black",
                linewidth=2,
            )
            fig.update_xaxes(
                row=row,
                col=col,
                mirror=True,
                showline=True,
                linecolor="black",
                linewidth=2,
            )

    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Relative Humidity (%)", row=1, col=2)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=3)
    fig.update_xaxes(title_text="Wind Direction (degrees)", row=1, col=4)
    fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Particle Concentration 186", row=2, col=2)
    fig.update_xaxes(title_text="CO2", row=2, col=3)
    fig.update_xaxes(title_text="STAP", row=2, col=4)
    fig.update_xaxes(title_text="Temperature (°C)", row=3, col=1)
    fig.update_xaxes(title_text="CO", row=3, col=2)
    fig.update_xaxes(title_text="N2O", row=3, col=3)
    fig.update_xaxes(title_text="Ozone", row=3, col=4)

    layout = constants.PLOT_LAYOUT_COMMON
    layout["height"] = 1000

    if resample_seconds is None:
        title = "Measurements"
    else:
        title = f"Measurements (resampled to {resample_seconds} seconds)"

    fig.update_layout(
        **layout,
        title=title,
        coloraxis=dict(colorbar=dict(orientation="h", y=-0.15)),
    )

    return fig


def generate_particle_heatmap(
    df: pd.DataFrame,
    props_msems_inverted: Dict[str, Any],
    props_msems_scan: Dict[str, Any],
) -> go.Figure:

    figlist = []

    # Get number of bins
    bins = reduce_column_to_single_unique_value(df, "msems_inverted_NumBins")

    z = df[[f"msems_inverted_Bin_Conc{x}" for x in range(1, bins)]].dropna()
    y = df[[f"msems_inverted_Bin_Lim{x}" for x in range(1, bins)]].dropna()
    x = df[["msems_inverted_StartTime"]].dropna()
    fig = go.Figure(
        data=go.Heatmap(
            z=z.T,
            x=x.index.values,
            y=y.mean(),
            colorscale="Viridis",
            **props_msems_inverted,
        )
    )
    fig.update_yaxes(
        type="log",
        range=(np.log10(y.mean().iloc[0]), np.log10(y.mean().iloc[-1])),
        tickformat="f",
        nticks=4,
    )
    fig.update_layout(
        **constants.PLOT_LAYOUT_COMMON,
        title="Bin concentrations (msems_inverted)",
    )

    fig.update_yaxes(title_text="D<sub>p</sub> [nm]")
    fig.update_xaxes(title_text="Time")
    fig.update_layout(height=600)

    figlist.append(fig)

    z = df[[f"msems_scan_bin{x}" for x in range(1, bins)]].dropna()
    y = df[[f"msems_inverted_Bin_Lim{x}" for x in range(1, bins)]].dropna()
    x = df[["msems_inverted_StartTime"]].dropna()

    fig = go.Figure(
        data=go.Heatmap(
            z=z.T,
            x=x.index.values,
            y=y.mean(),
            colorscale="Viridis",
            **props_msems_scan,
        )
    )
    fig.update_yaxes(
        type="log",
        range=(np.log10(y.mean().iloc[0]), np.log10(y.mean().iloc[-1])),
        tickformat="f",
        nticks=4,
    )
    fig.update_layout(
        **constants.PLOT_LAYOUT_COMMON, title="Bin readings (msems_scan)"
    )

    fig.update_yaxes(title_text="D<sub>p</sub> [nm]")
    fig.update_xaxes(title_text="Time")
    fig.update_layout(height=600)

    figlist.append(fig)

    return figlist


def generate_average_bin_concentration_plot(
    df: pd.DataFrame,
    title: str,
    timestamp_start: pd.Timestamp,
    timestamp_end: pd.Timestamp,
    bin_limit_col_prefix: str = "msems_inverted_Bin_Lim",
    bin_concentration_col_prefix: str = "msems_inverted_Bin_Conc",
    bin_quantity_col: str = "msems_inverted_NumBins",
    y_logscale: bool = False,
) -> go.Figure:
    """With a given timestamp, generate an average of MSEM bin concentrations

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data
    title : str
        Title of plot
    timestamp_start : str
        Start timestamp of period to average
    timestamp_end : str
        End timestamp of period to average
    bin_limit_col_prefix : str, optional
        Prefix of column containing bin limits, default: msems_inverted_Bin_Lim
    bin_concentration_col_prefix : str, optional
        Prefix of column containing bin concentrations,
        default 'msems_inverted_Bin_Conc'
    bin_quantity_col : str, optional
        Column containing number of bins, default 'msems_inverted_NumBins'
    y_logscale : bool, optional
        Whether to use a log scale for the y axis, default False


    Returns
    -------
    go.Figure
        Plotly figure
    """

    # Take sample of dataframe of the given time period
    df = df[timestamp_start:timestamp_end]

    # Get number of bins
    bins = reduce_column_to_single_unique_value(df, bin_quantity_col)

    x = df[[f"{bin_limit_col_prefix}{x}" for x in range(1, bins)]]
    y = df[[f"{bin_concentration_col_prefix}{x}" for x in range(1, bins)]]

    fig = go.Figure()

    num_records = min(len(x), len(y))
    logger.info(
        f"Generating mean MSEMS plot for {title} with {num_records} " "records"
    )
    for i in range(0, num_records):
        recordx = x.iloc[i]
        recordy = y.iloc[i]
        fig.add_trace(
            go.Scattergl(
                x=recordx.to_numpy().flatten(),
                y=recordy.to_numpy().flatten(),
                name=str(recordx.name),
                line={"color": "rgba(143, 82, 244 ,0.2)", "width": 2.5},
            )
        )

    # Plot the mean
    fig.add_trace(
        go.Scattergl(
            x=x.mean(numeric_only=True).to_numpy().flatten(),
            y=y.mean(numeric_only=True).to_numpy().flatten(),
            line={"width": 5, "color": "rgba(255, 0, 0, 1)"},
            name="Mean",
        )
    )

    fig.update_layout(
        title=f"{title}: ({num_records} records; "
        f"{timestamp_start} to {timestamp_end})"
    )
    fig.update_layout(height=600)
    fig.update_xaxes(
        title_text="Bin size",
        type="log",
        range=(np.log10(x.mean()[0]), np.log10(x.mean()[-1])),
        tickformat="f",
        nticks=4,
    )
    fig.update_yaxes(title_text="Particle concentration")

    if y_logscale:
        fig.update_yaxes(type="log")

    return fig


def write_plots_to_html(figures: List[go.Figure], filename: str) -> None:

    # Remove all None items in figures list. These are None because an
    # instrument may not have a figure to create, None is default
    figures = [i for i in figures if i is not None]

    logger.info(f"Writing {len(figures)} figures to {filename}")
    # Write out all of the figures to HTML
    with open(filename, "w") as f:
        # Write figures. They'll be sorted by the order they were added
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs=True))


def generate_altitude_plot(
    df: pd.DataFrame,
    at_ground_level: bool,
    altitude_col: str = "flight_computer_Altitude",
) -> go.Figure:

    colors = generate_normalised_colours(df)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df[altitude_col],
            name="smart_tether_Wind",
            mode="markers",
            marker=dict(
                color=colors, size=constants.PLOT_MARKER_SIZE, showscale=False
            ),
        )
    )

    if at_ground_level:
        title = "Altitude (ground level)"
    else:
        title = "Altitude (sea level)"

    # Update background to white and add black border
    fig.update_layout(
        **constants.PLOT_LAYOUT_COMMON,
        title=title,
        xaxis=dict(
            title="Time",
            mirror=True,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=(
            dict(
                title="Altitude (m)",
                mirror=True,
                showline=True,
                linecolor="black",
                linewidth=2,
            )
        ),
    )

    return fig


def generate_altitude_concentration_plot(
    df: pd.DataFrame,
    bins: List[Tuple[str, str, str]],
    at_ground_level: bool,
    height: int = 400,
    altitude_col: str = "flight_computer_Altitude",
) -> go.Figure:
    """Generate a plot that sits above MSEMS with selected bins

    The bins represent the periods of time that the averaged plots are
    calculated from.
    """

    fig = generate_altitude_plot(df, at_ground_level, altitude_col)

    # Set line markers to single solid colour
    fig.update_traces(marker=dict(color="rgb(31, 119, 180)"))

    for title, time_start, time_end in bins:
        # Ignore any that don't have a start or end time
        if time_start is None or time_end is None:
            continue

        fig.add_vrect(
            x0=time_start,
            x1=time_end,
            annotation_text=title,
            annotation_position="top left",
            fillcolor="green",
            opacity=0.25,
            line_width=0,
        )

    fig.update_layout(height=height)

    return fig


def generate_normalised_colours(
    df: pd.DataFrame, convert_nan_to: int = 0
) -> List[str]:
    """Generate a list of colours for a plot based on index of dataframe"""

    color_scale = px.colors.sequential.Rainbow
    normalized_index = (df.index - df.index.min()) / (
        df.index.max() - df.index.min()
    )

    # If there are NaN values in the index, convert them to the given value
    normalized_index = normalized_index.fillna(convert_nan_to)
    colors = [
        color_scale[int(x * (len(color_scale) - 1))] for x in normalized_index
    ]

    return colors


def campaign_2023(
    df: pd.DataFrame,
    plot_props: Dict[str, Any],
    all_instruments: List[instruments.Instrument],
    output_path_with_time: str,
) -> None:
    """Defines all the plots for the 2023 campaigns

    Due to the plot specifications being bespoke according to plans defined
    in 2023, this function attemps to isolate these decisions from further
    plotting requirements in the future

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing all the merged data to be plotted
    plot_props : Dict[str, Any]
        Dictionary containing all the properties for the plots originating from
        the runtime YAML
    all_instruments : List[instruments.Instrument]
        List of all instruments that are available after merging all the data
    output_path_with_time : str
        Path to the output directory for the plots, most likely the one
        generated in helikite.py with the current output time
    """

    # Set altitude plots based on ground station altitude or calculated
    if plot_props["altitude_ground_level"] is True:
        altitude_col = constants.ALTITUDE_GROUND_LEVEL_COL
        logger.info("Plotting altitude relative to ground level")
    else:
        altitude_col = constants.ALTITUDE_SEA_LEVEL_COL
        logger.info("Plotting altitude relative to sea level")

    # List to add plots to that will end up being exported
    figures_quicklook = []
    figures_qualitycheck = []

    figures_quicklook.append(
        generate_altitude_plot(
            df,
            at_ground_level=plot_props["altitude_ground_level"],
            altitude_col=altitude_col,
        )
    )
    figures_quicklook.append(
        generate_grid_plot(
            df,
            all_instruments,
            altitude_col=altitude_col,
            resample_seconds=plot_props["grid"]["resample_seconds"],
        )
    )

    # Create a list of instruments with pressure housekeeping variables
    # This allows automatic addition of instrument to pressure plots
    pressure_housekeeping = []
    pressure_quicklook = []

    for instrument in all_instruments:
        if instrument.pressure_variable is not None:
            pressure_housekeeping.append(
                f"{instrument.name}_{constants.HOUSEKEEPING_VAR_PRESSURE}"
            )
            pressure_quicklook.append(
                f"{instrument.name}_{instrument.pressure_variable}"
            )

    # Housekeeping pressure vars as qualitychecks
    figures_qualitycheck.append(
        plot_scatter_from_variable_list_by_index(
            df, "Housekeeping pressure variables", pressure_housekeeping
        )
    )

    # Same with just pressure vars
    figures_qualitycheck.append(
        plot_scatter_from_variable_list_by_index(
            df, "Pressure variables", pressure_quicklook
        )
    )

    """ Generate plots for qualitychecks based on their variable name in the
        merged dataframe. The two parameter Tuple[List[str], str] represents
        the list of variables, and the title given to the plot in the second
        parameter.
    """
    for variables, instrument in [
        (
            [
                f"{instruments.flight_computer.name}_vBat",
                f"{instruments.flight_computer.name}_TEMPbox",
            ],
            "Flight Computer",
        ),
        (
            (
                [
                    f"{instruments.flight_computer.name}_TEMP1",
                    f"{instruments.flight_computer.name}_TEMP2",
                    f"{instruments.smart_tether.name}_T (deg C)",
                ],
                "Smart Tether",
            )
            if instruments.smart_tether in all_instruments
            else (
                [
                    f"{instruments.flight_computer.name}_TEMP1",
                    f"{instruments.flight_computer.name}_TEMP2",
                ],
                "Flight Computer",
            )
        ),  # Don't plot smart tether temp if not present
        (
            ([f"{instruments.pops.name}_POPS_Flow"], "POPS")
            if instruments.pops in all_instruments
            else (None, None)
        ),
        (
            (
                [
                    f"{instruments.msems_readings.name}_msems_errs",
                    f"{instruments.msems_readings.name}_mcpc_errs",
                ],
                "MSEMS Readings",
            )
            if instruments.msems_readings in all_instruments
            else (None, None)
        ),
        (
            (
                [
                    f"{instruments.pico.name}_win1Fit7",
                    f"{instruments.pico.name}_win1Fit8",
                ],
                "Pico",
            )
            if instruments.pico in all_instruments
            else (None, None)
        ),
        (
            (
                [
                    f"{instruments.ozone_monitor.name}_cell_temp",
                    f"{instruments.ozone_monitor.name}_cell_pressure",
                    f"{instruments.ozone_monitor.name}_flow_rate",
                ],
                "Ozone",
            )
            if instruments.ozone_monitor in all_instruments
            else (None, None)
        ),
        (
            (
                [
                    f"{instruments.filter.name}_cur_pos",
                    f"{instruments.filter.name}_smp_flw",
                    f"{instruments.filter.name}_pumpctl",
                ],
                "Filter",
            )
            if instruments.filter in all_instruments
            else (None, None)
        ),
        (
            (
                [f"{instruments.stap_raw.name}_smp_flw"],
                "STAP Raw",
            )
            if instruments.stap_raw in all_instruments
            else (None, None)
        ),
    ]:
        if variables is not None:
            figures_qualitycheck.append(
                plot_scatter_from_variable_list_by_index(
                    df,
                    instrument,
                    variables,
                )
            )

    # Generate MSEMS related plots (heatmaps and average bin concentration)
    if instruments.msems_scan in all_instruments:
        # Create a list of tuples of the form (title, time_start, time_end)
        # for each plot to be used to generate bins in msems altitude plot
        msems_bins = [
            (x, y["time_start"], y["time_end"])
            for x, y in plot_props["msems_readings_averaged"].items()
        ]
        figures_quicklook.append(
            generate_altitude_concentration_plot(
                df,
                msems_bins,
                at_ground_level=plot_props["altitude_ground_level"],
                altitude_col=altitude_col,
            )
        )

        heatmaps = generate_particle_heatmap(
            df,
            plot_props["heatmap"]["msems_inverted"],
            plot_props["heatmap"]["msems_scan"],
        )

        # Heatmaps are returned as a list of figures, so add them to the
        # figures list
        for figure in heatmaps:
            figures_quicklook.append(figure)

        # Generate average bin concentration plots
        for title, props in plot_props["msems_readings_averaged"].items():
            # If either of the times are None, skip this plot
            if props["time_start"] is None or props["time_end"] is None:
                logger.warning(
                    "No time set for MSEMS readings averaged plot "
                    f"titled: '{title}'. Skipping"
                )
                continue

            # Generate the plot using the parameters from the config file
            fig = generate_average_bin_concentration_plot(
                df=df,
                title=title,
                timestamp_start=props["time_start"],
                timestamp_end=props["time_end"],
                y_logscale=props["log_y"],
            )
            figures_quicklook.append(fig)

    # Save quicklook and qualitycheck plots to HTML files
    quicklook_filename = os.path.join(
        output_path_with_time, constants.QUICKLOOK_PLOT_FILENAME
    )
    qualitycheck_filename = os.path.join(
        output_path_with_time, constants.QUALITYCHECK_PLOT_FILENAME
    )

    write_plots_to_html(figures_quicklook, quicklook_filename)
    write_plots_to_html(figures_qualitycheck, qualitycheck_filename)

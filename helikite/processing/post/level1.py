import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pydantic import BaseModel

from helikite.classes.output_schemas import OutputSchema
from helikite.instruments import msems_inverted, pops
from helikite.instruments.base import filter_columns_by_instrument
from helikite.instruments.mcda_instrument import mcda, MCDA_MIDPOINT_DIAMETER_LIST


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


def rename_columns(df: pd.DataFrame, output_schema: OutputSchema):
    """
    Renames columns of the input DataFrame according to predefined rules and instrument-specific rules.
    See `Instrument.rename_dict`

    Parameters:
        df (pd.DataFrame): The DataFrame with columns to be renamed.
        output_schema (OutputSchema): The OutputSchema object containing the instruments.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    rename_dict = {
        'DateTime': 'datetime',
        'latitude_dd': 'Lat',
        'longitude_dd': 'Long',
        'flight_computer_pressure': 'P',
        'Average_Temperature': 'TEMP',
        'Average_RH': 'RH',
    }
    for instrument in output_schema.instruments:
        rename_dict = rename_dict | instrument.rename_dict

    df_renamed = df.rename(columns=rename_dict)

    return df_renamed


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


def flight_profiles_1(df, xlims, xticks, fig_title):
    
    # Find the index of the maximum altitude
    max_altitude_index = df['Altitude'].idxmax()
    max_altitude_pos = df.index.get_loc(max_altitude_index)

    # Split DataFrame into ascent and descent
    df_up = df.iloc[:max_altitude_pos + 1]
    df_down = df.iloc[max_altitude_pos + 1:]

    plt.close('all')
    fig, ax = plt.subplots(1, 5, figsize=(16,6), gridspec_kw={'width_ratios': [1,1,1,1,0.3]})
    plt.subplots_adjust(wspace=0.3)
    num_subplots = 4

    ax1 = ax[0]
    ax2 = ax1.twiny()
    ax3 = ax[1]
    ax4 = ax3.twiny()
    ax5 = ax[2]
    ax6 = ax5.twiny()
    ax7 = ax[3]
    ax8 = ax7.twiny()

    # Position second x-axis ticks below the axis for all twinned axes
    for twin_ax in [ax2, ax4, ax6, ax8]:
        twin_ax.xaxis.set_ticks_position('bottom')
        twin_ax.xaxis.set_label_position('bottom')
        twin_ax.spines['bottom'].set_position(('outward', 40))

    # Plot ascent data
    ax1.plot(df_up["Average_Temperature"], df_up["Altitude"], color="brown", linewidth=3.0)
    ax2.plot(df_up["Average_RH"], df_up["Altitude"], color="orange", linewidth=3.0)
    ax3.plot(df_up["msems_inverted_dN_totalconc_stp"], df_up["Altitude"], color="indigo", marker='.')
    ax4.plot(df_up["cpc_totalconc_stp"], df_up["Altitude"], color="orchid", linewidth=3.0)
    ax5.plot(df_up["pops_total_conc_stp"], df_up["Altitude"], color="teal", linewidth=3.0)
    ax6.plot(df_up["mcda_dN_totalconc_stp"], df_up["Altitude"], color="salmon", linewidth=3.0)
    ax7.scatter(df_up["smart_tether_Wind (m/s)"], df_up["Altitude"], color='palevioletred', marker='.')
    ax8.scatter(df_up["smart_tether_Wind (degrees)"], df_up["Altitude"], color='olivedrab', marker='.')

    # Plot descent data with transparency
    ax1.plot(df_down["Average_Temperature"], df_down["Altitude"], color="brown", alpha=0.5, linewidth=3.0)
    ax2.plot(df_down["Average_RH"], df_down["Altitude"], color="orange", alpha=0.5, linewidth=3.0)
    ax3.plot(df_down["msems_inverted_dN_totalconc_stp"], df_down["Altitude"], color="indigo", alpha=0.3, marker='.')
    ax4.plot(df_down["cpc_totalconc_stp"], df_down["Altitude"], color="orchid", alpha=0.5, linewidth=3.0)
    ax5.plot(df_down["pops_total_conc_stp"], df_down["Altitude"], color="teal", alpha=0.5, linewidth=3.0)
    ax6.plot(df_down["mcda_dN_totalconc_stp"], df_down["Altitude"], color="salmon", alpha=0.5, linewidth=3.0)
    ax7.scatter(df_down["smart_tether_Wind (m/s)"], df_down["Altitude"], color='palevioletred', alpha=0.2, marker='.')
    ax8.scatter(df_down["smart_tether_Wind (degrees)"], df_down["Altitude"], color='olivedrab', alpha=0.3, marker='.')

    temp_divider = 2
    ws_divider = 2
    (temp_min_bound, temp_max_bound), temp_divider = _get_series_bounds(df["Average_Temperature"], temp_divider)
    (ws_min_bound, ws_max_bound), ws_divider = _get_series_bounds(df["smart_tether_Wind (m/s)"], ws_divider)

    # Default axis limits and ticks if none provided
    default_xlim = {
        'ax1': (temp_min_bound, temp_max_bound),
        'ax2': (60, 100),
        'ax3': (0, 1200),
        'ax4': (0, 1200),
        'ax5': (0, 60),
        'ax6': (0, 60),
        'ax7': (ws_min_bound, ws_max_bound),
        'ax8': (0, 360)
    }

    default_xticks = {
        'ax1': np.arange(temp_min_bound, temp_max_bound + 1, temp_divider),
        'ax2': np.arange(60, 101, 10),
        'ax3': np.arange(0, 1201, 200),
        'ax4': np.arange(0, 1201, 200),
        'ax5': np.arange(0, 61, 10),
        'ax6': np.arange(0, 61, 10),
        'ax7': np.arange(ws_min_bound, ws_max_bound + 1, ws_divider),
        'ax8': np.arange(0, 361, 90)
    }

    xlims = xlims or default_xlim
    xticks = xticks or default_xticks

    # Apply limits and ticks
    ax1.set_xlim(*xlims.get('ax1', default_xlim['ax1']))
    ax1.set_xticks(xticks.get('ax1', default_xticks['ax1']))

    ax2.set_xlim(*xlims.get('ax2', default_xlim['ax2']))
    ax2.set_xticks(xticks.get('ax2', default_xticks['ax2']))

    ax3.set_xlim(*xlims.get('ax3', default_xlim['ax3']))
    ax3.set_xticks(xticks.get('ax3', default_xticks['ax3']))

    ax4.set_xlim(*xlims.get('ax4', default_xlim['ax4']))
    ax4.set_xticks(xticks.get('ax4', default_xticks['ax4']))

    ax5.set_xlim(*xlims.get('ax5', default_xlim['ax5']))
    ax5.set_xticks(xticks.get('ax5', default_xticks['ax5']))

    ax6.set_xlim(*xlims.get('ax6', default_xlim['ax6']))
    ax6.set_xticks(xticks.get('ax6', default_xticks['ax6']))

    ax7.set_xlim(*xlims.get('ax7', default_xlim['ax7']))
    ax7.set_xticks(xticks.get('ax7', default_xticks['ax7']))

    ax8.set_xlim(*xlims.get('ax8', default_xlim['ax8']))
    ax8.set_xticks(xticks.get('ax8', default_xticks['ax8']))

    # Y axis minor ticks, grid and limits on main axes
    for j in range(num_subplots):
        ax[j].yaxis.set_minor_locator(ticker.MultipleLocator(base=50))
        ax[j].set_axisbelow(True)
        ax[j].grid(which='major', linestyle='--', linewidth=0.5, color='gray')
        ax[j].grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax[j].set_ylim(-10, df['Altitude'].max() + 10)

    # Label settings
    ax[0].set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
    for i in range(1, num_subplots):
        ax[i].set_yticklabels('')

    ax1.set_xlabel('Temp (°C)', color="brown", fontweight='bold')
    ax1.tick_params(axis='x', labelcolor='brown')

    ax2.set_xlabel('RH (%)', color="orange", fontweight='bold')
    ax2.tick_params(axis='x', labelcolor='orange')

    ax3.set_xlabel('mSEMS conc. (cm$^{-3}$) [8-250 nm]', color="indigo", fontweight='bold')
    ax3.tick_params(axis='x', labelcolor='indigo')

    ax4.set_xlabel('CPC conc. (cm$^{-3}$) [7–2000 nm]', color="orchid", fontweight='bold')
    ax4.tick_params(axis='x', labelcolor='orchid')

    ax5.set_xlabel('POPS conc. (cm$^{-3}$) [186-3370 nm]', color="teal", fontweight='bold')
    ax5.tick_params(axis='x', labelcolor='teal')

    ax6.set_xlabel('mCDA conc. (cm$^{-3}$) [0.66–33 um]', color="salmon", fontweight='bold')
    ax6.tick_params(axis='x', labelcolor='salmon')

    ax7.set_xlabel('WS (m/s)', color="palevioletred", fontweight='bold')
    ax7.tick_params(axis='x', labelcolor='palevioletred')

    ax8.set_xlabel('WD (deg)', color="olivedrab", fontweight='bold')
    ax8.tick_params(axis='x', labelcolor='olivedrab')

    # Hide last subplot axis and add legend
    ax[4].axis('off')
    legend_lines = [
        Line2D([0], [0], color='darkgrey', lw=4, label='Ascent'),
        Line2D([0], [0], color='lightgrey', lw=4, label='Descent')
    ]
    ax[4].legend(
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


def flight_profiles_2(df, metadata, xlims=None, xticks=None, fig_title=None):
    
    # Find the index of the maximum altitude
    max_altitude_index = df['Altitude'].idxmax()
    max_altitude_pos = df.index.get_loc(max_altitude_index)

    # Split DataFrame into ascent and descent
    df_up = df.iloc[:max_altitude_pos + 1]
    df_down = df.iloc[max_altitude_pos + 1:]

    plt.close('all')
    fig, ax = plt.subplots(1, 5, figsize=(16,6), gridspec_kw={'width_ratios': [1,1,1,1,0.3]})
    plt.subplots_adjust(wspace=0.3)
    num_subplots = 4

    ax1 = ax[0]
    ax2 = ax1.twiny()
    ax3 = ax[1]
    ax4 = ax3.twiny()
    ax5 = ax[2]
    ax6 = ax5.twiny()
    ax7 = ax[3]
    ax8 = ax7.twiny()

    # Position second x-axis ticks below the axis for all twinned axes
    for twin_ax in [ax2, ax4, ax6, ax8]:
        twin_ax.xaxis.set_ticks_position('bottom')
        twin_ax.xaxis.set_label_position('bottom')
        twin_ax.spines['bottom'].set_position(('outward', 40))

    # Shade clouds separately for ascent and descent
    if 'flag_cloud' in df.columns:
        cloud_points = df[df['flag_cloud'] == 1]
        if not cloud_points.empty:
            label_added = False
            delta_alt = 5  # meters to extend single-point shading

            # Find the position of max altitude
            max_alt_pos = df.index.get_loc(df['Altitude'].idxmax())

            # Split ascent and descent using iloc (integer positions)
            df_up = df.iloc[:max_alt_pos + 1]
            df_down = df.iloc[max_alt_pos + 1:]

            def shade_cloud_segment(df_segment, ax_list):
                cloud_times_seg = df_segment[df_segment['flag_cloud'] == 1].index
                if not cloud_times_seg.empty:
                    start_idx = cloud_times_seg[0]
                    label_added = False  # define inside the function
                    for i in range(1, len(cloud_times_seg)):
                        if (cloud_times_seg[i] - cloud_times_seg[i-1]) > pd.Timedelta(seconds=20):
                            ymin = df_segment.loc[start_idx, 'Altitude']
                            ymax = df_segment.loc[cloud_times_seg[i-1], 'Altitude']
                            delta_alt = 5  # for single-point shading
                            if ymin == ymax:
                                ymin -= delta_alt
                                ymax += delta_alt
                            for ax in ax_list:
                                ax.axhspan(ymin, ymax, color='lightblue', alpha=0.3,
                                           label='Cloud' if not label_added else None)
                            label_added = True
                            start_idx = cloud_times_seg[i]

                    # Final segment
                    ymin = df_segment.loc[start_idx, 'Altitude']
                    ymax = df_segment.loc[cloud_times_seg[-1], 'Altitude']
                    delta_alt = 5
                    if ymin == ymax:
                        ymin -= delta_alt
                        ymax += delta_alt
                    for ax in ax_list:
                        ax.axhspan(ymin, ymax, color='lightblue', alpha=0.3,
                                   label='Cloud' if not label_added else None)

            # Apply separately for ascent and descent
            shade_cloud_segment(df_up, [ax3, ax5])
            shade_cloud_segment(df_down, [ax3, ax5])


    # Shade areas for pollution on ax3 (altitude-based)
    if 'flag_pollution' in df.columns:
        pollution_times = df[df['flag_pollution'] == 1].index
        if not pollution_times.empty:
            start_idx = pollution_times[0]
            label_added = False
            for i in range(1, len(pollution_times)):
                if (pollution_times[i] - pollution_times[i-1]) > pd.Timedelta(seconds=1):
                    ymin = df.loc[start_idx, 'Altitude']
                    ymax = df.loc[pollution_times[i-1], 'Altitude']
                    ax3.axhspan(ymin, ymax, color='lightcoral', alpha=0.3,
                                label='Pollution' if not label_added else None)
                    label_added = True
                    start_idx = pollution_times[i]
            ymin = df.loc[start_idx, 'Altitude']
            ymax = df.loc[pollution_times[-1], 'Altitude']
            ax3.axhspan(ymin, ymax, color='lightcoral', alpha=0.3,
                        label='Pollution' if not label_added else None)

    # Plot ascent data
    ax1.plot(df_up["TEMP"], df_up["Altitude"], color="brown", linewidth=3.0)
    ax2.plot(df_up["RH"], df_up["Altitude"], color="orange", linewidth=3.0)
    ax3.plot(df_up["mSEMS_total_N"], df_up["Altitude"], color="indigo", marker='.')
    ax4.plot(df_up["CPC_total_N"], df_up["Altitude"], color="orchid", linewidth=3.0)
    ax5.plot(df_up["POPS_total_N"], df_up["Altitude"], color="teal", linewidth=3.0)
    ax6.plot(df_up["mCDA_total_N"], df_up["Altitude"], color="salmon", linewidth=3.0)
    ax7.scatter(df_up["WindSpeed"], df_up["Altitude"], color='palevioletred', marker='.')
    ax8.scatter(df_up["WindDir"], df_up["Altitude"], color='olivedrab', marker='.')

    # Plot descent data with transparency
    ax1.plot(df_down["TEMP"], df_down["Altitude"], color="brown", alpha=0.5, linewidth=3.0)
    ax2.plot(df_down["RH"], df_down["Altitude"], color="orange", alpha=0.5, linewidth=3.0)
    ax3.plot(df_down["mSEMS_total_N"], df_down["Altitude"], color="indigo", alpha=0.3, marker='.')
    ax4.plot(df_down["CPC_total_N"], df_down["Altitude"], color="orchid", alpha=0.5, linewidth=3.0)
    ax5.plot(df_down["POPS_total_N"], df_down["Altitude"], color="teal", alpha=0.5, linewidth=3.0)
    ax6.plot(df_down["mCDA_total_N"], df_down["Altitude"], color="salmon", alpha=0.5, linewidth=3.0)
    ax7.scatter(df_down["WindSpeed"], df_down["Altitude"], color='palevioletred', alpha=0.2, marker='.')
    ax8.scatter(df_down["WindDir"], df_down["Altitude"], color='olivedrab', alpha=0.3, marker='.')

    temp_divider = 2
    ws_divider = 2
    (temp_min_bound, temp_max_bound), temp_divider = _get_series_bounds(df["TEMP"], temp_divider)
    (ws_min_bound, ws_max_bound), ws_divider = _get_series_bounds(df["WindSpeed"], ws_divider)

    # Default axis limits and ticks if none provided
    default_xlim = {
        'ax1': (temp_min_bound, temp_max_bound),
        'ax2': (60, 100),
        'ax3': (0, 1200),
        'ax4': (0, 1200),
        'ax5': (0, 60),
        'ax6': (0, 60),
        'ax7': (ws_min_bound, ws_max_bound),
        'ax8': (0, 360)
    }

    default_xticks = {
        'ax1': np.arange(temp_min_bound, temp_max_bound + 1, 2),
        'ax2': np.arange(60, 101, 10),
        'ax3': np.arange(0, 1201, 200),
        'ax4': np.arange(0, 1201, 200),
        'ax5': np.arange(0, 61, 10),
        'ax6': np.arange(0, 61, 10),
        'ax7': np.arange(ws_min_bound, ws_max_bound + 1, 2),
        'ax8': np.arange(0, 361, 90)
    }

    xlims = xlims or default_xlim
    xticks = xticks or default_xticks

    # Apply limits and ticks
    ax1.set_xlim(*xlims.get('ax1', default_xlim['ax1']))
    ax1.set_xticks(xticks.get('ax1', default_xticks['ax1']))

    ax2.set_xlim(*xlims.get('ax2', default_xlim['ax2']))
    ax2.set_xticks(xticks.get('ax2', default_xticks['ax2']))

    ax3.set_xlim(*xlims.get('ax3', default_xlim['ax3']))
    ax3.set_xticks(xticks.get('ax3', default_xticks['ax3']))

    ax4.set_xlim(*xlims.get('ax4', default_xlim['ax4']))
    ax4.set_xticks(xticks.get('ax4', default_xticks['ax4']))

    ax5.set_xlim(*xlims.get('ax5', default_xlim['ax5']))
    ax5.set_xticks(xticks.get('ax5', default_xticks['ax5']))

    ax6.set_xlim(*xlims.get('ax6', default_xlim['ax6']))
    ax6.set_xticks(xticks.get('ax6', default_xticks['ax6']))

    ax7.set_xlim(*xlims.get('ax7', default_xlim['ax7']))
    ax7.set_xticks(xticks.get('ax7', default_xticks['ax7']))

    ax8.set_xlim(*xlims.get('ax8', default_xlim['ax8']))
    ax8.set_xticks(xticks.get('ax8', default_xticks['ax8']))

    # Y axis minor ticks, grid and limits on main axes
    for j in range(num_subplots):
        ax[j].yaxis.set_minor_locator(ticker.MultipleLocator(base=50))
        ax[j].set_axisbelow(True)
        ax[j].grid(which='major', linestyle='--', linewidth=0.5, color='gray')
        ax[j].grid(which='minor', linestyle='--', linewidth=0.5, color='lightgray')
        ax[j].set_ylim(-10, df['Altitude'].max() + 10)

    # Label settings
    ax[0].set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
    for i in range(1, num_subplots):
        ax[i].set_yticklabels('')

    ax1.set_xlabel('Temp (°C)', color="brown", fontweight='bold')
    ax1.tick_params(axis='x', labelcolor='brown')

    ax2.set_xlabel('RH (%)', color="orange", fontweight='bold')
    ax2.tick_params(axis='x', labelcolor='orange')

    ax3.set_xlabel('mSEMS conc. (cm$^{-3}$) [8-250 nm]', color="indigo", fontweight='bold')
    ax3.tick_params(axis='x', labelcolor='indigo')

    ax4.set_xlabel('CPC conc. (cm$^{-3}$) [7–2000 nm]', color="orchid", fontweight='bold')
    ax4.tick_params(axis='x', labelcolor='orchid')

    ax5.set_xlabel('POPS conc. (cm$^{-3}$) [186-3370 nm]', color="teal", fontweight='bold')
    ax5.tick_params(axis='x', labelcolor='teal')

    ax6.set_xlabel('mCDA conc. (cm$^{-3}$) [0.66–33 um]', color="salmon", fontweight='bold')
    ax6.tick_params(axis='x', labelcolor='salmon')

    ax7.set_xlabel('WS (m/s)', color="palevioletred", fontweight='bold')
    ax7.tick_params(axis='x', labelcolor='palevioletred')

    ax8.set_xlabel('WD (deg)', color="olivedrab", fontweight='bold')
    ax8.tick_params(axis='x', labelcolor='olivedrab')

    # Hide last subplot axis and add legend
    ax[4].axis('off')
    legend_lines = [
        Line2D([0], [0], color='darkgrey', lw=4, label='Ascent'),
        Line2D([0], [0], color='lightgrey', lw=4, label='Descent'),
        Patch(facecolor='lightblue', alpha=0.3, label='Cloud'),
        Patch(facecolor='lightcoral', alpha=0.3, label='Pollution')
    ]

    ax[4].legend(
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


def _get_series_bounds(x: pd.Series, default_divider: int) -> tuple[tuple[int, int], int]:
    min_bound = np.floor(x.quantile(0.01)).astype(int)
    max_bound = np.ceil(x.quantile(0.99)).astype(int)

    if max_bound - min_bound > default_divider * 7:
        divider = ((max_bound - min_bound) // 7)
        divider = (divider + default_divider - divider % default_divider) if divider % default_divider else divider
    else:
        divider = default_divider

    min_bound = min_bound - min_bound % divider
    max_bound = max_bound + max_bound % divider

    return (min_bound, max_bound), divider


def plot_size_distributions(df: pd.DataFrame, output_schema: OutputSchema, fig_title: str,
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

    # Shade areas for flag_pollution == 1
    if "flag_pollution" in df.columns:
        pollution_times = df[df['flag_pollution'] == 1].index
        if not pollution_times.empty:
            start = pollution_times[0]
            for i in range(1, len(pollution_times)):
                if (pollution_times[i] - pollution_times[i - 1]) > timedelta:
                    ax1.axvspan(start, pollution_times[i - 1], color='lightcoral', alpha=0.8, label='Pollution')
                    start = pollution_times[i]
            ax1.axvspan(start, pollution_times[-1], color='lightcoral', alpha=0.8, label='Pollution')

    # Shade areas for flag_hovering == 1
    if "flag_hovering" in df.columns:
        hovering_times = df[df['flag_hovering'] == 1].index
        if not hovering_times.empty:
            start = hovering_times[0]
            for i in range(1, len(hovering_times)):
                if (hovering_times[i] - hovering_times[i - 1]) > timedelta:
                    ax1.axvspan(start, hovering_times[i - 1], color='beige', alpha=1, label='Hovering')
                    start = hovering_times[i]
            ax1.axvspan(start, hovering_times[-1], color='beige', alpha=1, label='Hovering')

    # Shade areas for flag_cloud == 1
    if "flag_cloud" in df.columns:
        cloud_times = df[df['flag_cloud'] == 1].index
        if not cloud_times.empty:
            start = cloud_times[0]
            for i in range(1, len(cloud_times)):
                if (cloud_times[i] - cloud_times[i - 1]) > timedelta:
                    ax1.axvspan(start, cloud_times[i - 1], color='lightblue', alpha=0.5, label='Cloud')
                    start = cloud_times[i]
            ax1.axvspan(start, cloud_times[-1], color='lightblue', alpha=0.5, label='Cloud')

    # Shade areas for Filter_position !== 1.0
    if "Filter_position" in df.columns:
        filter_mask = df['Filter_position'] != 1.0
        filter_times = df[filter_mask].index

        if not filter_times.empty:
            start = filter_times[0]
            for i in range(1, len(filter_times)):
                if (filter_times[i] - filter_times[i - 1]) > timedelta:
                    ax1.axvspan(start, filter_times[i - 1], facecolor='none', edgecolor='gray', hatch='////', alpha=0.8,
                                label='Filter')
                    start = filter_times[i]
            ax1.axvspan(start, filter_times[-1], facecolor='none', edgecolor='gray', hatch='////', alpha=0.8,
                        label='Filter')

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


def filter_data(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Filter_position
    ax1.plot(df.index, df['flight_computer_F_cur_pos'], color='tab:blue', label='Filter Position')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Filter Position', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a second y-axis for Filter_flow
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['flight_computer_F_pump_pw'], color='tab:red', label='Pump Power')
    ax2.set_ylabel('Pump Power', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Title and legend
    #fig.suptitle('Filter Position and Flow vs Time')
    fig.tight_layout()
    plt.show()

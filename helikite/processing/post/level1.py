import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helikite.processing.post import level1
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D


def create_level1_dataframe(df):
    """
    Create a level 1 DataFrame by selecting specific columns and returning a copy.

    Parameters:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: Filtered copy of the DataFrame with selected columns.
    """
    # Manually specified columns
    base_columns = [
        'DateTime', 'Altitude', 'latitude_dd', 'longitude_dd',
        'flight_computer_pressure', 'Average_Temperature', 'Average_RH',
        'smart_tether_Wind (m/s)', 'smart_tether_Wind (degrees)',
        'pops_total_conc_stp', 'msems_inverted_dN_totalconc_stp',
        'mcda_dN_totalconc_stp', 'cpc_totalconc_stp',
        'flight_computer_F_cur_pos', 'flight_computer_F_smp_flw'
    ]

    # Dynamically generated column ranges
    pops_range = [f'pops_b{i}_dlogDp_stp' for i in range(3, 16)]
    msems_range = [f'msems_inverted_Bin_Conc{i}_stp' for i in range(1, 61)]
    mcda_range = [f'mcda_dataB {i}_dN_dlogDp_stp' for i in range(1, 257)]

    # TAPIR columns
    tapir_columns = [
        'tapir_GL', 'tapir_Lat', 'tapir_Le', 'tapir_Lon', 'tapir_Lm', 'tapir_speed',
        'tapir_route', 'tapir_TP', 'tapir_Tproc1', 'tapir_Tproc2', 'tapir_Tproc3', 'tapir_Tproc4',
        'tapir_TH', 'tapir_Thead1', 'tapir_Thead2', 'tapir_Thead3', 'tapir_Thead4',
        'tapir_TB', 'tapir_Tbox'
    ]

    # Combine all columns
    selected_columns = base_columns + pops_range + msems_range + mcda_range + tapir_columns

    # Return the filtered DataFrame
    return df[selected_columns].copy()


def rename_columns(df):
    """
    Renames columns of the input DataFrame according to predefined rules.

    Parameters:
        df (pd.DataFrame): The DataFrame with columns to be renamed.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    # Define all the renaming rules
    manual_rename = {
        'DateTime': 'datetime',
        'latitude_dd': 'Lat',
        'longitude_dd': 'Long',
        'flight_computer_pressure': 'P',
        'Average_Temperature': 'TEMP',
        'Average_RH': 'RH',
        'smart_tether_Wind (m/s)': 'WindSpeed',
        'smart_tether_Wind (degrees)': 'WindDir',
        'pops_total_conc_stp': 'POPS_total_N',
        'msems_inverted_dN_totalconc_stp': 'mSEMS_total_N',
        'mcda_dN_totalconc_stp': 'mCDA_total_N',
        'cpc_totalconc_stp': 'CPC_total_N',
        'flight_computer_F_cur_pos': 'Filter_position',
        'flight_computer_F_smp_flw': 'Filter_flow'
    }

    # Ranges to rename
    pops_range = [f'pops_b{i}_dlogDp_stp' for i in range(3, 16)]
    msems_range = [f'msems_inverted_Bin_Conc{i}_stp' for i in range(1, 61)]
    mcda_range = [f'mcda_dataB {i}_dN_dlogDp_stp' for i in range(1, 257)]

    # Automatically generated rename mappings
    pops_rename = {col: f'POPS_b{i}' for i, col in zip(range(3, 16), pops_range)}
    msems_rename = {col: f'mSEMS_Bin_Conc{i}' for i, col in zip(range(1, 61), msems_range)}
    mcda_rename = {col: f'mCDA_dataB{i}' for i, col in zip(range(1, 257), mcda_range)}

    # Combine all rename mappings
    rename_dict = {**manual_rename, **pops_rename, **msems_rename, **mcda_rename}

    # Apply renaming
    df_renamed = df.rename(columns=rename_dict)

    return df_renamed

def round_flightnbr_campaign(df, metadata, decimals=2):
    """
    Round numeric columns of the DataFrame with special handling for 'Lat' and 'Long',
    and add columns for flight number and campaign.

    Parameters:
        df (pd.DataFrame): The DataFrame to be rounded and modified.
        metadata (object): Metadata object containing the 'flight' attribute.
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
    df['flight_nr'] = metadata.flight
    df['campaign'] = 'ORACLES'

    return df



def fill_msems_takeoff_landing(df, metadata, time_window_seconds=90):
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


def flight_profiles_1(df, metadata, xlims=None, xticks=None, fig_title=None):
    
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

    # Default axis limits and ticks if none provided
    default_xlim = {
        'ax1': (-8, 0),
        'ax2': (60, 100),
        'ax3': (0, 1200),
        'ax4': (0, 1200),
        'ax5': (0, 60),
        'ax6': (0, 60),
        'ax7': (0, 8),
        'ax8': (0, 360)
    }

    default_xticks = {
        'ax1': np.arange(-8, 1, 2),
        'ax2': np.arange(60, 101, 10),
        'ax3': np.arange(0, 1201, 200),
        'ax4': np.arange(0, 1201, 200),
        'ax5': np.arange(0, 61, 10),
        'ax6': np.arange(0, 61, 10),
        'ax7': np.arange(0, 9, 2),
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

    # Default axis limits and ticks if none provided
    default_xlim = {
        'ax1': (-8, 0),
        'ax2': (60, 100),
        'ax3': (0, 1200),
        'ax4': (0, 1200),
        'ax5': (0, 60),
        'ax6': (0, 60),
        'ax7': (0, 8),
        'ax8': (0, 360)
    }

    default_xticks = {
        'ax1': np.arange(-8, 1, 2),
        'ax2': np.arange(60, 101, 10),
        'ax3': np.arange(0, 1201, 200),
        'ax4': np.arange(0, 1201, 200),
        'ax5': np.arange(0, 61, 10),
        'ax6': np.arange(0, 61, 10),
        'ax7': np.arange(0, 9, 2),
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

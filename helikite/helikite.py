import sys
from processing import preprocess, sorting
from constants import constants
import instruments
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import datetime
import plots
from functools import reduce
import logging


# Define a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(constants.LOGLEVEL_CONSOLE)
console_handler.setFormatter(constants.LOGFORMAT_CONSOLE)
logging.getLogger().addHandler(console_handler)

# Define logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


def main():
    # Get the yaml config
    yaml_config = preprocess.read_yaml_config(
        os.path.join(constants.INPUTS_FOLDER,
                     constants.CONFIG_FILE))

    # List to add plots to that will end up being exported
    figures = []

    # Create a folder with the current UTC time in outputs
    output_path_with_time = os.path.join(
        constants.OUTPUTS_FOLDER,
        datetime.datetime.utcnow().isoformat())
    output_path_instrument_subfolder = os.path.join(
        output_path_with_time, constants.OUTPUTS_INSTRUMENT_SUBFOLDER)
    os.makedirs(output_path_with_time)
    os.makedirs(output_path_instrument_subfolder)

    # Add a file logger
    logfile_handler = logging.FileHandler(os.path.join(output_path_with_time,
                                                       constants.LOGFILE_NAME))
    logfile_handler.setLevel(constants.LOGLEVEL_FILE)
    logfile_handler.setFormatter(constants.LOGFORMAT_FILE)
    logging.getLogger().addHandler(logfile_handler)

    # Append each df for merging
    all_export_dfs = []

    time_trim_start = pd.to_datetime(yaml_config['global']['time_trim']['start'])
    time_trim_end = pd.to_datetime(yaml_config['global']['time_trim']['end'])

    start_altitude = yaml_config['global']['altitude']

    # Go through each instrument and perform the operations on each instrument
    for instrument, props in yaml_config['instruments'].items():

        instrument_obj = getattr(instruments, props['config'])
        instrument_obj.add_yaml_config(props)

        if instrument_obj.filename is None:
            logger.warning(f"Skipping {instrument}: No file assigned!")
            continue
        else:
            logger.info(f"Processing {instrument}: {instrument_obj.filename}")

        df = instrument_obj.read_data()

        # Modify the DateTime index based off the configuration offsets
        df = instrument_obj.set_time_as_index(df)

        # Using the time corrections from configuration, correct time index
        df = instrument_obj.correct_time_from_config(
            df, time_trim_start, time_trim_end
        )

        # Apply any corrections on the data
        df = instrument_obj.data_corrections(df, start_altitude=start_altitude)

        # Create housekeeping pressure variable to help align pressure visually
        df = instrument_obj.set_housekeeping_pressure_offset_variable(
            df, column_name=constants.HOUSEKEEPING_VAR_PRESSURE
        )

        # Generate the plots and add them to the list
        figures = figures + instrument_obj.create_plots(df)

        # Save dataframe to outputs folder
        df.to_csv(
            f"{os.path.join(output_path_instrument_subfolder, instrument)}.csv"
        )

        # Prepare df for combining with other instruments
        df = instrument_obj.add_device_name_to_columns(df)

        # Add tuple of df and export order to df merge list
        all_export_dfs.append((df, instrument_obj.export_order,
                               instrument_obj.housekeeping_columns,
                               instrument_obj.export_columns,))

    preprocess.export_yaml_config(
        yaml_config,
        os.path.join(output_path_with_time,
                        constants.CONFIG_FILE)
    )

    # Sort the export columns in their numerical hierarchy order
    all_export_dfs.sort(key=sorting.df_column_sort_key)

    master_export_cols = []
    master_housekeeping_cols = []

    # Merge all the dataframes together, first df is the master
    master_df, sort_id, hk_cols, export_cols = all_export_dfs[0]

    master_export_cols += export_cols    # Combine list of data export columns
    master_housekeeping_cols += hk_cols  # Combine list of housekeeping columns
    for df, sort_id, hk_cols, export_cols in all_export_dfs[1:]:  # Merge rest
        master_df = master_df.merge(
            df, how="outer", left_index=True, right_index=True)
        master_export_cols += export_cols
        master_housekeeping_cols += hk_cols

    # Sort rows by the date index
    master_df.index = pd.to_datetime(master_df.index)
    master_df.sort_index(inplace=True)

    # Export data and housekeeping CSV files
    master_df[master_export_cols].to_csv(
        os.path.join(output_path_with_time,
                     constants.MASTER_CSV_FILENAME))

    master_df[master_housekeeping_cols].to_csv(
        os.path.join(output_path_with_time,
                     constants.HOUSEKEEPING_CSV_FILENAME))


    # Plots
    color_scale = px.colors.sequential.Rainbow
    normalized_index = (master_df.index - master_df.index.min()) / (master_df.index.max() - master_df.index.min())
    colors = [color_scale[int(x * (len(color_scale)-1))] for x in normalized_index]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=master_df.index,
            y=master_df["flight_computer_Altitude"],
            name="smart_tether_Wind",
            mode="markers",
            marker=dict(
                color=colors,
                size=3,
                showscale=False),
        )
    )

    # Update background to white and add black border
    fig.update_layout(
        title="Altitude",
        xaxis=dict(
            title="Time",
            mirror=True,
            showline=True,
            linecolor='black',
            linewidth=2
        ),
        yaxis=(dict(
            title="Altitude (m)",
            mirror=True,
            showline=True,
            linecolor='black',
            linewidth=2
        )),
        height=600,
        template="plotly_white",)
    # fig.update_xaxes()
    fig.update_yaxes(mirror=True)
    figures.append(fig)

    fig = make_subplots(rows=2, cols=4, shared_yaxes=False)

    fig.add_trace(go.Scatter(
        x=master_df["flight_computer_TEMP1"],
        y=master_df["flight_computer_Altitude"],
        name="Temperature1 (Flight Computer)",
        mode="markers",
        marker=dict(
            color=colors,
            size=5,
            showscale=False,
            symbol="circle"
        )),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=master_df["flight_computer_TEMP2"],
        y=master_df["flight_computer_Altitude"],
        name="Temperature2 (Flight Computer)",
        mode="markers",
        marker=dict(
            color=colors,
            size=5,
            showscale=False,
            symbol="cross"
        )),
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=master_df["smart_tether_T (deg C)"],
        y=master_df["flight_computer_Altitude"],
        name="Temperature (Smart Tether)",
        mode="markers",
        marker=dict(
            color=colors,
            size=5,
            showscale=False,
            symbol="diamond"
        )),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=master_df["flight_computer_RH1"],
        y=master_df["flight_computer_Altitude"],
        name="flight_computer_RH1",
        mode="markers",
        marker=dict(
            color=colors,
            size=3,
            showscale=False
        )),
        row=1, col=2).update_layout(xaxis_title="Relative Humidity (%)")

    fig.add_trace(go.Scatter(
        x=master_df["smart_tether_Wind (m/s)"],
        y=master_df["flight_computer_Altitude"],
        name="smart_tether_Wind",
        mode="markers",
        marker=dict(
            color=colors,
            size=3,
            showscale=False
        )),
        row=1, col=3)

    fig.add_trace(go.Scatter(
        x=master_df["smart_tether_Wind (degrees)"],
        y=master_df["flight_computer_Altitude"],
        name="smart_tether_Wind",
        mode="markers",
        marker=dict(
            color=colors,
            size=3,
            showscale=False
        )),

        row=1, col=4)
    fig.add_trace(go.Scatter(
        x=master_df["smart_tether_Wind (degrees)"],
        y=master_df["flight_computer_Altitude"],
        name="smart_tether_Wind",
        mode="markers",
        marker=dict(
            color=colors,
            size=3,
            showscale=False
        )),
        row=1, col=4)

    fig.add_trace(go.Scatter(
        x=master_df.index,
        y=master_df["flight_computer_Altitude"],
        name="smart_tether_Wind",
        mode="markers",
        marker=dict(
            color=colors,
            size=3,
            showscale=False
        )),
        row=2, col=1)

    # Give each plot a border and white background
    for row in [1, 2]:
        for col in [1, 2, 3, 4]:
            fig.update_yaxes(
                row=row, col=col,
                mirror=True, showline=True, linecolor='black', linewidth=2)
            fig.update_xaxes(
                row=row, col=col,
                mirror=True, showline=True, linecolor='black', linewidth=2)

    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Relative Humidity (%)", row=1, col=2)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=3)
    fig.update_xaxes(title_text="Wind Direction (degrees)", row=1, col=4)
    fig.update_layout(coloraxis=dict(colorbar=dict(orientation='h', y=-0.15)),
                      template="plotly_white",)

    figures.append(fig)



    # Housekeeping pressure vars
    figures.append(
        plots.plot_scatter_from_variable_list_by_index(
            master_df, "Housekeeping pressure variables",
            [
                "flight_computer_housekeeping_pressure",
                "msems_readings_housekeeping_pressure",
                "msems_scan_housekeeping_pressure",
                "stap_housekeeping_pressure",
                "smart_tether_housekeeping_pressure",
                "pops_housekeeping_pressure"
            ],
        )
    )

    # Pressure vars
    figures.append(
        plots.plot_scatter_from_variable_list_by_index(
            master_df, "Pressure variables",
            [
                "flight_computer_P_baro",
                "msems_readings_pressure",
                "msems_scan_press_avg",
                "stap_sample_press_mbar",
                "smart_tether_P (mbar)",
                "pops_P"
            ],
        )
    )
    import numpy as np
    z = master_df[[f"msems_inverted_Bin_Conc{x}" for x in range(1, 60)]].dropna()
    y = master_df[[f"msems_inverted_Bin_Lim{x}" for x in range(1, 60)]].dropna()
    x = master_df[['msems_inverted_StartTime']].dropna()
    fig = go.Figure(
        data=go.Heatmap(
        z=z.T,
        x=x.index.values,
        y=y.mean(), colorscale = 'Viridis',
        #zmin=-200, zmid=1400, zmax=1400
        ))
    fig.update_yaxes(type='log',
                    range=(np.log10(y.mean()[0]), np.log10(y.mean()[-1])),
                    tickformat="f",
                    nticks=4
                    )
    # fig.show()
    figures.append(fig)
    html_filename = os.path.join(output_path_with_time,
                                 constants.HTML_OUTPUT_FILENAME)
    plots.write_plots_to_html(figures, html_filename)

if __name__ == '__main__':
    # If docker arg given, don't run main
    if len(sys.argv) > 1:
        if sys.argv[1] == 'preprocess':
            preprocess.generate_config(overwrite=False)  # Write conf file
            preprocess.preprocess()
        elif sys.argv[1] == 'generate_config':
            logger.info("Generating YAML configuration in input folder")
            preprocess.generate_config(overwrite=True)
        else:
            logger.error("Unknown argument. Options are: preprocess, generate_config")
    else:
        # If no args, run the main application
        main()

import plotly.graph_objects as go
from ipywidgets import Output, VBox, Dropdown, ToggleButton
import pandas as pd


def choose_outliers(df, y, outlier_file="outliers.csv"):
    """Creates a plot to interactively select outliers in the data.

    A plot is generated where two variables are plotted, and the user can
    click on points to select or deselect them as outliers, or use Plotly's
    selection tools to select multiple points at once.

    Args:
        df (pandas.DataFrame): The dataframe containing the data
        y (str): The column name of the y-axis variable
        outlier_file (str): The path to the CSV file to store the outliers
    """
    # Create a figure widget for interactive plotting
    fig = go.FigureWidget()
    out = Output()
    out.append_stdout(
        "Click on a point to toggle its outlier status in zoom mode.\n\n"
        "Otherwise, use either the lasso or box selection tool to select "
        "multiple points, and toggle their addition or deletion with the "
        "Add/Remove Mode toggle button. \n\n"
        "Double click to clear the selection, "
        "zoom out by double clicking in the zoom function, and use the "
        "dropdown to change the x-axis variable."
    )
    df = df.copy()

    # Initialize x with the first numerical column other than y
    df = df.fillna("")
    # Add the index as a column to allow it to be used on the x-axis and place
    # it as the first column
    index_column_name = (
        f"Index<{df.index.name}>" if df.index.name else "Index<>"
    )
    df[index_column_name] = df.index
    df = df[
        [index_column_name]
        + [col for col in df.columns if col != index_column_name]
    ]

    variable_options = [var for var in df.columns if var != y]
    x = variable_options[1]  # Set first non-index var to the first in the list

    # Load or create the outliers DataFrame
    try:
        pd.set_option("future.no_silent_downcasting", False)
        outliers = pd.read_csv(
            outlier_file, index_col=0, parse_dates=True
        ).fillna(False)
    except FileNotFoundError:
        print(f"Outlier file not found. Creating new one at {outlier_file}")
        outliers = pd.DataFrame(columns=df.columns).fillna(False)

    # Ensure the indices are aligned and of the same type
    outliers.index = pd.to_datetime(outliers.index)
    df.index = pd.to_datetime(df.index)

    # Create the variable selection dropdown
    variable_dropdown = Dropdown(
        options=variable_options, value=x, description="Variable:"
    )

    # Create the mode toggle button (now always visible)
    add_remove_toggle = ToggleButton(
        value=True,
        description="Add Mode",
        disabled=False,
        button_style="",
        tooltip="Click to toggle between add and remove modes",
        icon="plus",  # You can use 'minus' for remove mode
    )

    def on_toggle_change(change):
        if change["new"]:
            add_remove_toggle.description = "Add Mode"
            add_remove_toggle.icon = "plus"
        else:
            add_remove_toggle.description = "Remove Mode"
            add_remove_toggle.icon = "minus"

    add_remove_toggle.observe(on_toggle_change, names="value")

    def update_plot(*args):
        # Get the current variable from the dropdown
        current_x = variable_dropdown.value

        # Update the main trace
        with fig.batch_update():
            fig.data[0].x = df[current_x]
            fig.data[0].y = df[y]
            fig.data[0].name = current_x
            fig.data[0].text = [
                f"Time: {time} X: {x_val} Y: {y_val}"
                for time, x_val, y_val in zip(df.index, df[current_x], df[y])
            ]
            fig.layout.xaxis.title = current_x
            fig.layout.title = f"{y} vs {current_x}"

            # Update the outlier trace
            if current_x in outliers.columns:
                outlier_mask = outliers[current_x].fillna(False)
                outlier_indices = outliers[outlier_mask].index
                outlier_points = df.loc[outlier_indices]
            else:
                outlier_points = pd.DataFrame(columns=df.columns)

            fig.data[1].x = outlier_points[current_x]
            fig.data[1].y = outlier_points[y]

    @out.capture(clear_output=True)
    def click_point_callback(trace, points, selector):
        # Callback function for click events to toggle outlier status
        nonlocal outliers
        if points.point_inds:
            point_index = points.point_inds[0]
            selected_index = df.iloc[point_index]

            # Get the current x variable from the dropdown
            current_x = variable_dropdown.value

            # Check if the point is already an outlier
            if (
                selected_index.name in outliers.index
                and current_x in outliers.columns
                and outliers.loc[selected_index.name, current_x]
            ):
                # Remove the outlier
                outliers.loc[selected_index.name, current_x] = False
                # Remove the row if all entries are False
                if not outliers.loc[selected_index.name].any():
                    outliers = outliers.drop(selected_index.name)
                print("Removed 1 outlier")
            else:
                # Add the outlier
                if selected_index.name not in outliers.index:
                    # Initialize a new row with False values
                    outliers.loc[selected_index.name] = False
                outliers.loc[selected_index.name, current_x] = True
                print("Added 1 outlier")

            outliers_without_index_column = outliers.drop(
                columns=[index_column_name]
            )

            outliers_without_index_column.to_csv(
                outlier_file, date_format="%Y-%m-%d %H:%M:%S"
            )

            # Update the plot
            update_plot()

    @out.capture(clear_output=True)
    def select_points_callback(trace, points, selector):
        # Callback function to add/remove selected points as outliers
        nonlocal outliers
        if points.point_inds:
            selected_indices = df.iloc[points.point_inds].index

            # Get the current x variable from the dropdown
            current_x = variable_dropdown.value

            mode = "add" if add_remove_toggle.value else "remove"
            count = 0
            if mode == "add":
                # Add selected indices to outliers
                for index in selected_indices:
                    if index not in outliers.index:
                        # Initialize a new row with False values
                        outliers.loc[index] = False
                    if not outliers.loc[index, current_x]:
                        outliers.loc[index, current_x] = True
                        count += 1
                print(f"Added {count} outliers")
            elif mode == "remove":
                # Remove selected indices from outliers
                for index in selected_indices:
                    if (
                        index in outliers.index
                        and current_x in outliers.columns
                        and outliers.loc[index, current_x]
                    ):
                        outliers.loc[index, current_x] = False
                        # Remove the row if all entries are False
                        if not outliers.loc[index].any():
                            outliers = outliers.drop(index)
                        count += 1
                print(f"Removed {count} outliers")

            outliers_without_index_column = outliers.drop(
                columns=[index_column_name]
            )

            outliers_without_index_column.to_csv(
                outlier_file, date_format="%Y-%m-%d %H:%M:%S"
            )

            # Update the plot
            update_plot()

    # Initial plot
    fig.add_trace(
        go.Scattergl(
            x=df[x],
            y=df[y],
            name=x,
            opacity=1,
            mode="markers",
            marker=dict(
                color=df.index.to_series().astype(int),
                colorscale="Viridis",
                colorbar=dict(
                    tickvals=[df.index.min().value, df.index.max().value],
                    ticktext=[
                        df.index.min().strftime("%Y-%m-%d %H:%M:%S"),
                        df.index.max().strftime("%Y-%m-%d %H:%M:%S"),
                    ],
                ),
            ),
            hoverinfo="text",
            text=[
                f"Time: {time} X: {x_val} Y: {y_val}"
                for time, x_val, y_val in zip(df.index, df[x], df[y])
            ],
        )
    )

    # Attach the callbacks to the main trace
    fig.data[0].on_click(click_point_callback)
    fig.data[0].on_selection(select_points_callback)

    # Add the outlier points to the plot
    if x in outliers.columns:
        outlier_mask = outliers[x].fillna(False)
        outlier_indices = outliers[outlier_mask].index
        outlier_points = df.loc[outlier_indices]
    else:
        outlier_points = pd.DataFrame(columns=df.columns)

    fig.add_trace(
        go.Scattergl(
            x=outlier_points[x],
            y=outlier_points[y],
            name="Outliers",
            mode="markers",
            marker=dict(
                color="red",
                symbol="x",
                size=10,
            ),
            showlegend=True,
        )
    )

    fig.update_layout(
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title=f"{y} vs {x}",
        xaxis_title=x,
        yaxis_title=y,
        hovermode="closest",
        showlegend=True,
        height=600,
        width=1000,
    )

    # Observe variable selection changes
    variable_dropdown.observe(update_plot, names="value")

    # Show plot with interactive selection functionality
    return VBox([variable_dropdown, add_remove_toggle, fig, out])


def choose_flags(df, y, flag_file="flags.csv", key="flag", value="selected"):
    """Creates a plot to interactively select and assign flags to data points.

    A plot is generated where two variables are plotted, and the user can
    click on points to assign or remove a flag value to a specified key (column).
    The flags are stored in a CSV file, which can be used to tag the data points
    later.


    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    y : str
        The column name of the y-axis variable.
    flag_file : str, optional
        The path to the CSV file to store the flags (default is "flags.csv").
    key : str, optional
        The column name to assign flags (default is "flag").
    value : str, optional
        The value to assign to the key when points are selected (default is "selected").

    Returns
    -------
    VBox
        A VBox widget containing the variable dropdown, add/remove toggle button,
        the interactive plot, and the output widget.
    """

    # Create a figure widget for interactive plotting
    fig = go.FigureWidget()
    out = Output()
    out.append_stdout(
        "Click on a point to toggle its flag status in zoom mode.\n\n"
        "Otherwise, use either the lasso or box selection tool to select "
        "multiple points, and toggle their addition or deletion with the "
        "Add/Remove Mode toggle button. \n\n"
        "Double click to clear the selection, "
        "zoom out by double clicking in the zoom function, and use the "
        "dropdown to change the x-axis variable."
    )
    df = df.copy()

    # Initialize x with the first numerical column other than y
    df = df.fillna("")

    # Add the index as a column to allow it to be used on the x-axis and place
    # it as the first column
    index_column_name = (
        f"Index<{df.index.name}>" if df.index.name else "Index<>"
    )
    df[index_column_name] = df.index
    df = df[
        [index_column_name]
        + [col for col in df.columns if col != index_column_name]
    ]

    variable_options = [var for var in df.columns if var != y]
    x = variable_options[1]  # Set first non-index var to the first in the list

    # Load or create the flags DataFrame
    try:
        flags = pd.read_csv(flag_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Flag file not found. Creating new one at {flag_file}")
        flags = pd.DataFrame(columns=[key])

    # Ensure the indices are aligned and of the same type
    flags.index = pd.to_datetime(flags.index)
    df.index = pd.to_datetime(df.index)

    # Check if there are values in the specified key, and print summary
    if key in flags.columns and not flags[key].isnull().all():
        counts = flags[key].value_counts()
        print(f'In "{key}", these values:')
        for val, count in counts.items():
            print(f"{val}: {count}")

    # Create the variable selection dropdown
    variable_dropdown = Dropdown(
        options=variable_options, value=x, description="Variable:"
    )

    # Create the mode toggle button
    add_remove_toggle = ToggleButton(
        value=True,
        description="Add Mode",
        disabled=False,
        button_style="",
        tooltip="Click to toggle between add and remove modes",
        icon="plus",
    )

    def on_toggle_change(change):
        if change["new"]:
            add_remove_toggle.description = "Add Mode"
            add_remove_toggle.icon = "plus"
        else:
            add_remove_toggle.description = "Remove Mode"
            add_remove_toggle.icon = "minus"

    add_remove_toggle.observe(on_toggle_change, names="value")

    def update_plot(*args):
        # Get the current variable from the dropdown
        current_x = variable_dropdown.value

        with fig.batch_update():
            # Update the main trace
            fig.data[0].x = df[current_x]
            fig.data[0].y = df[y]
            fig.data[0].name = current_x
            fig.data[0].text = [
                f"Time: {time} X: {x_val} Y: {y_val}"
                for time, x_val, y_val in zip(df.index, df[current_x], df[y])
            ]
            fig.layout.xaxis.title = current_x
            fig.layout.title = f"{y} vs {current_x}"

            # Remove existing flagged points traces
            while len(fig.data) > 1:
                fig.data = fig.data[:-1]

            # Update the flagged points traces
            if key in flags.columns:
                unique_values = flags[key].dropna().unique()
                colors_list = [
                    "red",
                    "green",
                    "blue",
                    "orange",
                    "purple",
                    "brown",
                    "pink",
                    "gray",
                    "olive",
                    "cyan",
                ]
                color_mapping = {
                    val: colors_list[i % len(colors_list)]
                    for i, val in enumerate(unique_values)
                }
                for val in unique_values:
                    flagged_indices = flags[flags[key] == val].index
                    flagged_points = df.loc[flagged_indices]
                    fig.add_trace(
                        go.Scattergl(
                            x=flagged_points[current_x],
                            y=flagged_points[y],
                            name=f"Flagged: {val}",
                            mode="markers",
                            marker=dict(
                                color=color_mapping[val],
                                symbol="x",
                                size=10,
                            ),
                            showlegend=True,
                        )
                    )

    @out.capture(clear_output=True)
    def click_point_callback(trace, points, selector):
        # Callback function for click events to toggle flag status
        nonlocal flags
        if points.point_inds:
            point_index = points.point_inds[0]
            selected_index = df.iloc[point_index]

            # Check if the point is already flagged with the value
            current_value = (
                flags.loc[selected_index.name, key]
                if selected_index.name in flags.index and key in flags.columns
                else None
            )
            if current_value == value:
                # Remove the flag
                flags.loc[selected_index.name, key] = pd.NA
                # Remove the row if all entries are NA
                if flags.loc[selected_index.name].isnull().all():
                    flags = flags.drop(selected_index.name)
                print("Removed 1 flag")
            else:
                # Add the flag
                flags.loc[selected_index.name, key] = value
                print("Added 1 flag")

            flags.to_csv(flag_file, date_format="%Y-%m-%d %H:%M:%S")

            # Update the plot
            update_plot()

    @out.capture(clear_output=True)
    def select_points_callback(trace, points, selector):
        # Callback function to add/remove selected points as flags
        nonlocal flags
        if points.point_inds:
            selected_indices = df.iloc[points.point_inds].index

            mode = "add" if add_remove_toggle.value else "remove"
            count = 0
            if mode == "add":
                # Add selected indices to flags
                for index in selected_indices:
                    current_value = (
                        flags.loc[index, key]
                        if index in flags.index and key in flags.columns
                        else None
                    )
                    if current_value != value:
                        flags.loc[index, key] = value
                        count += 1
                print(f"Added {count} flags")
            elif mode == "remove":
                # Remove selected indices from flags
                for index in selected_indices:
                    current_value = (
                        flags.loc[index, key]
                        if index in flags.index and key in flags.columns
                        else None
                    )
                    if current_value == value:
                        flags.loc[index, key] = pd.NA
                        # Remove the row if all entries are NA
                        if flags.loc[index].isnull().all():
                            flags = flags.drop(index)
                        count += 1
                print(f"Removed {count} flags")

            flags.to_csv(flag_file, date_format="%Y-%m-%d %H:%M:%S")

            # Update the plot
            update_plot()

    # Initial plot
    fig.add_trace(
        go.Scattergl(
            x=df[x],
            y=df[y],
            name=x,
            opacity=1,
            mode="markers",
            marker=dict(
                color=df.index.to_series().astype(int),
                colorscale="Viridis",
                colorbar=dict(
                    tickvals=[df.index.min().value, df.index.max().value],
                    ticktext=[
                        df.index.min().strftime("%Y-%m-%d %H:%M:%S"),
                        df.index.max().strftime("%Y-%m-%d %H:%M:%S"),
                    ],
                ),
            ),
            hoverinfo="text",
            text=[
                f"Time: {time} X: {x_val} Y: {y_val}"
                for time, x_val, y_val in zip(df.index, df[x], df[y])
            ],
        )
    )

    # Attach the callbacks to the main trace
    fig.data[0].on_click(click_point_callback)
    fig.data[0].on_selection(select_points_callback)

    # Add the flagged points traces
    if key in flags.columns:
        unique_values = flags[key].dropna().unique()
        colors_list = [
            "red",
            "green",
            "blue",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        color_mapping = {
            val: colors_list[i % len(colors_list)]
            for i, val in enumerate(unique_values)
        }
        for val in unique_values:
            flagged_indices = flags[flags[key] == val].index
            flagged_points = df.loc[flagged_indices]
            fig.add_trace(
                go.Scattergl(
                    x=flagged_points[x],
                    y=flagged_points[y],
                    name=f"Flagged: {val}",
                    mode="markers",
                    marker=dict(
                        color=color_mapping[val],
                        symbol="x",
                        size=10,
                    ),
                    showlegend=True,
                )
            )

    fig.update_layout(
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title=f"{y} vs {x}",
        xaxis_title=x,
        yaxis_title=y,
        hovermode="closest",
        showlegend=True,
        height=600,
        width=1000,
    )

    # Observe variable selection changes
    variable_dropdown.observe(update_plot, names="value")

    # Show plot with interactive selection functionality
    return VBox([variable_dropdown, add_remove_toggle, fig, out])

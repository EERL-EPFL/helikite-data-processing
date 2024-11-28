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
    variable_options = [var for var in df.columns if var != y]
    x = variable_options[0]  # Set the first x plot to the first in the list

    # Load or create the outliers DataFrame
    try:
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

            outliers.to_csv(outlier_file, date_format="%Y-%m-%d %H:%M:%S")

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

            outliers.to_csv(outlier_file, date_format="%Y-%m-%d %H:%M:%S")

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

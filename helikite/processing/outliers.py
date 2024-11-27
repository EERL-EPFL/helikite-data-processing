import plotly.graph_objects as go
from ipywidgets import Output, VBox, Dropdown, ToggleButton
import pandas as pd


def choose_outliers(df, y, outlier_file="outliers.csv"):
    """Creates a plot to interactively select outliers in the data.

    A plot is generated where two variables are plotted, and the user can
    use the lasso or box selection tools to select or deselect multiple points as outliers.
    The selected points are stored in a list and saved to a CSV, which can be accessed later.

    Args:
        df (pandas.DataFrame): The dataframe containing the data
        y (str): The column name of the y-axis variable
        outlier_file (str): The path to the CSV file to store the outliers
    """
    # Create a figure widget for interactive plotting
    fig = go.FigureWidget()
    out = Output()
    out.append_stdout("Use the lasso or box select tool to select points.\n")
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

    # Create the mode toggle button
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
    def select_point_callback(trace, points, selector):
        # Callback function for selection events to add/remove selected points as outliers
        nonlocal outliers
        if points.point_inds:
            selected_indices = df.iloc[points.point_inds].index

            # Get the current x variable from the dropdown
            current_x = variable_dropdown.value

            mode = "add" if add_remove_toggle.value else "remove"

            if mode == "add":
                # Add selected indices to outliers
                for index in selected_indices:
                    if index not in outliers.index:
                        # Initialize a new row with False values
                        outliers.loc[index] = False
                    outliers.loc[index, current_x] = True
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

    # Attach the callback to the main trace
    fig.data[0].on_selection(select_point_callback)

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

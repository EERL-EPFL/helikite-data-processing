import plotly.graph_objects as go
from ipywidgets import Output, VBox
import pandas as pd


def choose_outliers(df, y, outlier_file="outliers.csv"):
    """Creates a plot to interactively select outliers in the data

    A plot is generated, where two variables are plotted, and the user can
    click on points to select them as outliers. The selected points are
    stored in a list, which can be accessed later.

    Args:
        df (pandas.DataFrame): The dataframe containing the data
        x (str): The column name of the x-axis variable
        y (str): The column name of the y-axis variable

    """
    # Create a figure widget for interactive plotting
    fig = go.FigureWidget()
    out = Output()
    out.append_stdout("Click on a point to set as outlier.\n")
    df = df.copy()
    selected_points = []

    # Create a list of buttons for the user to select which variable to plot
    variable_list = []
    df = df.fillna("")
    x = None
    for variable in df.columns:
        if variable == y or df[variable].dtype == "object":
            # Skip the x and y variables for the list
            continue
        variable_list.append(
            dict(
                args=[
                    {"x": [df[variable]], "name": variable},
                    {
                        "title": f"{y} vs {variable}",
                        "xaxis": {"title": variable},
                    },
                ],
                label=variable,
                method="update",
            )
        )

        # Set x to the first variable in the list, this will become the
        # first variable to be plotted
        if x is None:
            x = variable

    # If outlier file exists, load it. Otherwise, create a new one with the
    # same columns as the dateframe, to allow for appending new outliers based
    # on their index
    try:
        outliers = pd.read_csv(outlier_file)
        outliers.set_index(outliers.columns[0], drop=True, inplace=True)

    except FileNotFoundError:
        print(f"Outlier file not found. Creating new one at {outlier_file}")
        outliers = pd.DataFrame(columns=df.columns)

    @out.capture(clear_output=True)
    def select_point_callback(trace, points, selector):
        # Callback function for click events to select points
        nonlocal outliers
        if points.point_inds:
            point_index = points.point_inds[0]
            print(f"Point index1: {points.point_inds}")
            print(f"Point index2: {point_index}")
            # print("Trace", trace)
            selected_x = trace.x[point_index]
            selected_y = trace.y[point_index]

            # Get the index of the selected point from the dataframe
            selected_index = df[(df[x] == selected_x) & (df[y] == selected_y)]
            selected_index = df.iloc[point_index]
            # print(f"Selected DF index: {selected_index}")
            print("X Variable: ", x)
            # print("Trace X", trace.x)
            print("Points X", points)
            print("Trace name: ", points.trace_name)
            print(f"Selected x/y: {selected_x}/{selected_y}")
            print("Point at index: ", df.index[point_index])
            print(
                "Variable x and y at index: ",
                df[x].iloc[point_index],
                df[y].iloc[point_index],
            )
            selected_points.append((selected_x, selected_y))

            # Append the selected point to the outliers dataframe by adding or
            # updating a row with the selected point's index, where the
            # corresponding outlier columns are set to true and the rest false.
            # If a row already exists at that index, update it, otherwise add
            # a new row. We can assume the columns of the outliers dataframe
            # are the same as the columns of the dataframe
            if selected_index.name in outliers.index:
                outliers.loc[selected_index.name, [points.trace_name, y]] = (
                    True
                )
            else:

                new_row = pd.DataFrame(
                    [[False] * len(outliers.columns)],
                    columns=outliers.columns,
                    index=[selected_index.name],
                )
                new_row.loc[selected_index.name, [points.trace_name, y]] = True
                outliers = pd.concat([outliers, new_row])
            outliers.to_csv(outlier_file)

            print(outliers)

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

    # Attach the callback to all traces
    for trace in fig.data:
        # Only allow the reference instrument to be clickable
        # if trace.name == self.reference_instrument.name:
        trace.on_click(select_point_callback)
        print(f"Callback attached to trace: {trace.name}")

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=variable_list,
                direction="down",
                pad={"r": 5, "t": 5},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
    )

    fig.update_layout(
        annotations=[
            dict(
                text="Variable:",
                showarrow=False,
                x=0,
                y=1.085,
                yref="paper",
                align="left",
            )
        ],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # fig.update_layout(
    #     updatemenus=[
    #         dict(
    #             buttons=list(
    #                 [
    #                     dict(
    #                         args=["type", "surface"],
    #                         label="3D Surface",
    #                         method="restyle",
    #                     ),
    #                     dict(
    #                         args=["type", "heatmap"],
    #                         label="Heatmap",
    #                         method="restyle",
    #                     ),
    #                 ]
    #             ),
    #             direction="down",
    #             pad={"r": 10, "t": 10},
    #             showactive=True,
    #             x=0.1,
    #             xanchor="left",
    #             y=1.1,
    #             yanchor="top",
    #         ),
    #     ]
    # )
    # Customize plot layout
    fig.update_layout(
        title=f"{y} vs {x}",
        xaxis_title=x,
        yaxis_title=y,
        hovermode="closest",
        showlegend=True,
        height=600,
        width=800,
    )

    # Show plot with interactive click functionality
    return VBox([fig, out])  # Use VBox to stack the plot and output

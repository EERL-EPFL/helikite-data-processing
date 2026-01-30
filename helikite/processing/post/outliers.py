import logging

import matplotlib.pyplot as plt
import pandas as pd
import folium
from branca.colormap import linear
import matplotlib.dates as mdates

from helikite.constants import constants
from helikite.instruments.flight_computer import FlightComputer

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)

def plot_outliers_check(df, flight_computer: FlightComputer):
    """
    Plots various flight parameters against flight_computer_pressure.
    
    Args:
        df (pd.DataFrame): The DataFrame containing flight data.
        flight_computer (FlightComputer):
            Flight computer instance, required to obtain correct column names, since they differ between versions.

    """
    plt.close('all')

    gps_available = flight_computer.lat_column is not None and flight_computer.long_column is not None
    ncols = 4 if gps_available else 3
    fig, axs = plt.subplots(2, ncols, figsize=(ncols * 4, 8), sharey=True, constrained_layout=True, squeeze=False)

    # Plot Wind Speed vs Pressure
    axs[0, 0].scatter(df['smart_tether_Wind (m/s)'], df['flight_computer_pressure'],
                      color='palevioletred', alpha=0.7, s=10)
    axs[0, 0].set_xlabel('Wind Speed (m/s)', fontsize=10)
    axs[0, 0].set_ylabel('Pressure (hPa)', fontsize=10)
    axs[0, 0].set_title('WS', fontsize=10, fontweight='bold')

    # Plot Wind Direction vs Pressure
    axs[1, 0].scatter(df['smart_tether_Wind (degrees)'], df['flight_computer_pressure'],
                      color='olivedrab', alpha=0.7, s=10)
    axs[1, 0].set_xlabel('Wind Direction (°)', fontsize=10)
    axs[1, 0].set_ylabel('Pressure (hPa)', fontsize=10)
    axs[1, 0].set_xticks([0, 90, 180, 270, 360])
    axs[1, 0].set_title('WD', fontsize=10, fontweight='bold')

    t1_column = f"{flight_computer.name}_{flight_computer.T1_column}"
    t2_column = f"{flight_computer.name}_{flight_computer.T2_column}"
    h1_column = f"{flight_computer.name}_{flight_computer.H1_column}"
    h2_column = f"{flight_computer.name}_{flight_computer.H2_column}"
    lat_column = f"{flight_computer.name}_{flight_computer.lat_column}"
    long_column = f"{flight_computer.name}_{flight_computer.long_column}"

    # Plot Out1_T vs Pressure
    axs[0, 1].scatter(df[t1_column], df['flight_computer_pressure'],
                      color='brown', alpha=0.7, s=10)
    axs[0, 1].set_xlabel('Temperature (°C)', fontsize=10)
    axs[0, 1].set_title('Out1_T', fontsize=10, fontweight='bold')

    # Plot Out1_H vs Pressure
    axs[1, 1].scatter(df[h1_column], df['flight_computer_pressure'],
                      color='orange', alpha=0.7, s=10)
    axs[1, 1].set_xlabel('RH (%)', fontsize=10)
    axs[1, 1].set_title('Out1_H', fontsize=10, fontweight='bold')

    # Plot Out2_T vs Pressure
    axs[0, 2].scatter(df[t2_column], df['flight_computer_pressure'],
                      color='sienna', alpha=0.7, s=10)
    axs[0, 2].set_xlabel('Temperature (°C)', fontsize=10)
    axs[0, 2].set_title('Out2_T', fontsize=10, fontweight='bold')

    # Plot Out2_H vs Pressure
    axs[1, 2].scatter(df[h2_column], df['flight_computer_pressure'],
                      color='darkorange', alpha=0.7, s=10)
    axs[1, 2].set_xlabel('RH (%)', fontsize=10)
    axs[1, 2].set_title('Out2_H', fontsize=10, fontweight='bold')

    if gps_available:
        # Plot Latitude vs Pressure
        axs[0, 3].scatter(df[lat_column], df['flight_computer_pressure'],
                          color='teal', alpha=0.7, s=10)
        axs[0, 3].set_xlabel('Latitude', fontsize=10)
        axs[0, 3].set_title(flight_computer.lat_column, fontsize=10, fontweight='bold')

        # Plot Longitude vs Pressure
        axs[1, 3].scatter(df[long_column], df['flight_computer_pressure'],
                          color='salmon', alpha=0.7, s=10)
        axs[1, 3].set_xlabel('Longitude', fontsize=10)
        axs[1, 3].set_title(flight_computer.lat_column, fontsize=10, fontweight='bold')

    for ax in axs.flat:
        ax.grid(True)
        ax.invert_yaxis()

    plt.show()


def convert_gps_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str, lat_dir: str, lon_dir: str) -> pd.DataFrame:
    lat_dd_col = 'latitude_dd'
    lon_dd_col = 'longitude_dd'

    if lat_col not in df.columns or lon_col not in df.columns:
        lat_col, lon_col = _guess_lat_lon_columns(df.columns, lat_col, lon_col)
        if lat_col is None or lon_col is None:
            logger.warning(f"Setting `{lat_dd_col}` and `{lon_dd_col}` to NaN.")
            df[lat_dd_col] = pd.Series(pd.NA, index=df.index, dtype="Float64")
            df[lon_dd_col] = pd.Series(pd.NA, index=df.index, dtype="Float64")

            return df

    def convert_dm_to_dd(dm_values: pd.Series, direction: str):
        degrees = (dm_values / 100).round()
        minutes = dm_values - degrees * 100
        dd = degrees + minutes / 60
        if direction in ['S', 'W']:
            dd *= -1
        return dd

    # Convert latitude and longitude
    df[lat_dd_col] = convert_dm_to_dd(df[lat_col], lat_dir)
    df[lon_dd_col] = convert_dm_to_dd(df[lon_col], lon_dir)

    return df


def plot_gps_on_map(df, center_coords, zoom_start) -> folium.Map | None:
    # Drop rows with NaNs in converted coordinates
    df_clean = df.dropna(subset=['latitude_dd', 'longitude_dd'])

    if len(df_clean) == 0:
        logger.warning("All GPS points are NaN. Skipping plotting.")
        return None

    # Convert time to numeric
    time_numeric = mdates.date2num(df_clean.index.to_pydatetime())

    # Setup colormap
    colormap = linear.Reds_06.scale(time_numeric.min(), time_numeric.max())
    colormap.caption = 'Time progression'

    # Center coordinates
    lat_center, lon_center = center_coords
    m = folium.Map(location=[lat_center, lon_center], zoom_start=zoom_start)

    # Add GPS points
    for lat, lon, t in zip(df_clean['latitude_dd'], df_clean['longitude_dd'], time_numeric):
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=colormap(t),
            fill=True,
            fill_opacity=0.8
        ).add_to(m)

    # Add center marker
    folium.CircleMarker(
        location=[lat_center, lon_center],
        radius=6,
        color='black',
        fill=True,
        fill_color='black',
        fill_opacity=1,
        popup='NM'
    ).add_to(m)

    # Add colormap legend
    colormap.add_to(m)

    return m


def _guess_lat_lon_columns(columns: list[str], lat_col: str, lon_col: str) -> tuple[str | None, str | None]:
    def _guess_column(default_col: str, arg_name: str, substring: str) -> str | None:
        if default_col not in columns:
            logger.warning(f"Column '{default_col}' not found in DataFrame.")
            col_candidates = [col for col in columns if substring.lower() in col.lower()]
            if len(col_candidates) == 0:
                logger.warning(f"Provide a valid '{arg_name}' argument.")
                return None
            else:
                col = col_candidates[0]
                logger.warning(f"Using '{col}' as '{arg_name}'.")
                return col
        return default_col

    lat_col = _guess_column(lat_col,  "lat_col", "latitude")
    lon_col = _guess_column(lon_col, "lon_col", "longitude")

    return lat_col, lon_col
import matplotlib.pyplot as plt
import pandas as pd
import folium
from branca.colormap import linear
import matplotlib.dates as mdates


def plot_outliers_check(df):
    """
    Plots various flight parameters against flight_computer_pressure.
    
    Args:
        df (pd.DataFrame): The DataFrame containing flight data.
    """
    plt.close('all')

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True, constrained_layout=True)

    # Plot Wind Speed vs Pressure
    axs[0, 0].scatter(df['smart_tether_Wind (m/s)'], df['flight_computer_pressure'],
                      color='palevioletred', alpha=0.7, s=10)
    axs[0, 0].set_xlabel('Wind Speed (m/s)', fontsize=10)
    axs[0, 0].set_ylabel('Pressure (hPa)', fontsize=10)
    axs[0, 0].set_title('WS', fontsize=10, fontweight='bold')
    axs[0, 0].grid(True)
    axs[0, 0].invert_yaxis()

    # Plot Wind Direction vs Pressure
    axs[1, 0].scatter(df['smart_tether_Wind (degrees)'], df['flight_computer_pressure'],
                      color='olivedrab', alpha=0.7, s=10)
    axs[1, 0].set_xlabel('Wind Direction (°)', fontsize=10)
    axs[1, 0].set_ylabel('Pressure (hPa)', fontsize=10)
    axs[1, 0].set_xticks([0, 90, 180, 270, 360])
    axs[1, 0].set_title('WD', fontsize=10, fontweight='bold')
    axs[1, 0].grid(True)
    axs[1, 0].invert_yaxis()

    # Plot Out1_T vs Pressure
    axs[0, 1].scatter(df['flight_computer_Out1_T'], df['flight_computer_pressure'],
                      color='brown', alpha=0.7, s=10)
    axs[0, 1].set_xlabel('Temperature (°C)', fontsize=10)
    axs[0, 1].set_title('Out1_T', fontsize=10, fontweight='bold')
    axs[0, 1].grid(True)
    axs[0, 1].invert_yaxis()

    # Plot Out1_H vs Pressure
    axs[1, 1].scatter(df['flight_computer_Out1_H'], df['flight_computer_pressure'],
                      color='orange', alpha=0.7, s=10)
    axs[1, 1].set_xlabel('RH (%)', fontsize=10)
    axs[1, 1].set_title('Out1_H', fontsize=10, fontweight='bold')
    axs[1, 1].grid(True)
    axs[1, 1].invert_yaxis()

    # Plot Out2_T vs Pressure
    axs[0, 2].scatter(df['flight_computer_Out2_T'], df['flight_computer_pressure'],
                      color='sienna', alpha=0.7, s=10)
    axs[0, 2].set_xlabel('Temperature (°C)', fontsize=10)
    axs[0, 2].set_title('Out2_T', fontsize=10, fontweight='bold')
    axs[0, 2].grid(True)
    axs[0, 2].invert_yaxis()

    # Plot Out2_H vs Pressure
    axs[1, 2].scatter(df['flight_computer_Out2_H'], df['flight_computer_pressure'],
                      color='darkorange', alpha=0.7, s=10)
    axs[1, 2].set_xlabel('RH (%)', fontsize=10)
    axs[1, 2].set_title('Out2_H', fontsize=10, fontweight='bold')
    axs[1, 2].grid(True)
    axs[1, 2].invert_yaxis()

    plt.show()


def plot_gps_on_map(df, lat_col='flight_computer_Lat', lon_col='flight_computer_Long', 
                    lat_dir='S', lon_dir='W', center_coords=(-70.6587, -8.2850), zoom_start=13):
    
    def convert_dm_to_dd(dm_value, direction):
        if pd.isna(dm_value):
            return None
        degrees = int(dm_value / 100)
        minutes = dm_value - degrees * 100
        dd = degrees + minutes / 60
        if direction in ['S', 'W']:
            dd *= -1
        return dd

    # Convert latitude and longitude
    df['latitude_dd'] = df[lat_col].apply(lambda x: convert_dm_to_dd(x, lat_dir))
    df['longitude_dd'] = df[lon_col].apply(lambda x: convert_dm_to_dd(x, lon_dir))

    # Drop rows with NaNs in converted coordinates
    df_clean = df.dropna(subset=['latitude_dd', 'longitude_dd'])

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

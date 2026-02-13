import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from helikite.instruments import Instrument
from helikite.instruments.flight_computer import FlightComputer


def T_RH_averaging(df, columns_t: list[str], columns_rh: list[str], nan_threshold: int):
    """
    Averages flight computer temperature and humidity data from two sensors,
    based on the number of NaNs, and plots temperature and RH versus pressure.

    Parameters:
        df (pd.DataFrame): DataFrame containing flight computer data.
        columns_t: List of column names containing temperature data.
        columns_rh: List of column names containing humidity data.
        nan_threshold (int): Number of NaNs to tolerate before discarding a sensor.

    Returns:
        pd.DataFrame: Updated DataFrame with 'Average_Temperature' and 'Average_RH' columns.
    """

    # Count number of NaNs
    columns_filtered_t = _filter_columns_by_nan_count(columns_t, df, nan_threshold)
    columns_filtered_rh = _filter_columns_by_nan_count(columns_rh, df, nan_threshold)

    if len(columns_filtered_t) > 0:
        df["Average_Temperature"] = df[columns_filtered_t].mean(axis=1).ffill().bfill()
    else:
        df["Average_Temperature"] = pd.Series(pd.NA, index=df.index, dtype="Float64")

    if len(columns_filtered_rh) > 0:
        df["Average_RH"] = df[columns_filtered_rh].mean(axis=1).ffill().bfill()
    else:
        df["Average_RH"] = pd.Series(pd.NA, index=df.index, dtype="Float64")

    return df


def _filter_columns_by_nan_count(columns: list[str], df: pd.DataFrame, nan_threshold: int) -> list[str]:
    print("Number of NaNs -", end="")
    columns_filtered = []
    for col in columns:
        nan_count = df[col].isna().sum()
        print(f" {col}: {nan_count}", end="")
        if nan_count <= nan_threshold:
            columns_filtered.append(col)
    print()

    if len(columns_filtered) == 0:
        columns_filtered = columns

    return columns_filtered


def plot_T_RH(df,
              reference_instrument: Instrument,
              flight_computer: FlightComputer,
              save_path: str | pathlib.Path | None):
    t1_column = f"{flight_computer.name}_{flight_computer.T1_column}"
    t2_column = f"{flight_computer.name}_{flight_computer.T2_column}"
    h1_column = f"{flight_computer.name}_{flight_computer.H1_column}"
    h2_column = f"{flight_computer.name}_{flight_computer.H2_column}"

    pressure_column = f"{reference_instrument.name}_pressure"

    # PLOT
    plt.close('all')
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Temperature plot
    ax[0].plot(df[t1_column], df[pressure_column], label="Out1_T", color='blue')
    ax[0].plot(df[t2_column], df[pressure_column], label="Out2_T", color='orange')
    ax[0].plot(df["Average_Temperature"], df[pressure_column], label="Average_T", color='red')
    ax[0].plot(df["smart_tether_T (deg C)"], df[pressure_column], label="ST_T", color='green', linestyle='--')
    ax[0].set_xlabel("Temperature (°C)")
    ax[0].set_ylabel("Pressure (hPa)")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].invert_yaxis()

    # Humidity plot
    ax[1].plot(df[h1_column], df[pressure_column], label="Out1_RH", color='blue')
    ax[1].plot(df[h2_column], df[pressure_column], label="Out2_RH", color='orange')
    ax[1].plot(df["Average_RH"], df[pressure_column], label="Average_RH", color='red')
    ax[1].plot(df["smart_tether_%RH"], df[pressure_column], label="ST_RH", color='green', linestyle='--')
    ax[1].set_xlabel("Relative Humidity (%)")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()

    if save_path is not None:
        print("Saving figure to:", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
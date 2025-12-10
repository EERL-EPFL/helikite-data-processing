import matplotlib.pyplot as plt
import pandas as pd

from helikite.instruments.flight_computer import FlightComputer


def T_RH_averaging(df, flight_computer: FlightComputer, nan_threshold=400):
    """
    Averages flight computer temperature and humidity data from two sensors,
    based on the number of NaNs, and plots temperature and RH versus pressure.

    Parameters:
        df (pd.DataFrame): DataFrame containing flight computer data.
        nan_threshold (int): Number of NaNs to tolerate before discarding a sensor.
        flight_computer (FlightComputer):
            Flight computer instance, required to obtain correct column names, since they differ between versions.

    Returns:
        pd.DataFrame: Updated DataFrame with 'Average_Temperature' and 'Average_RH' columns.
    """

    t1_column = f"{flight_computer.name}_{flight_computer.T1_column}"
    t2_column = f"{flight_computer.name}_{flight_computer.T2_column}"
    h1_column = f"{flight_computer.name}_{flight_computer.H1_column}"
    h2_column = f"{flight_computer.name}_{flight_computer.H2_column}"

    # Count number of NaNs
    Out1T_nan = df[t1_column].isna().sum()
    Out2T_nan = df[t2_column].isna().sum()
    Out1H_nan = df[h1_column].isna().sum()
    Out2H_nan = df[h2_column].isna().sum()

    print("Number of NaNs - Out1_T:", Out1T_nan, "Out2_T:", Out2T_nan)
    print("Number of NaNs - Out1_H:", Out1H_nan, "Out2_H:", Out2H_nan)

    # Temperature averaging
    if Out1T_nan > nan_threshold:
        df["Average_Temperature"] = df[t2_column].copy()
    elif Out2T_nan > nan_threshold:
        df["Average_Temperature"] = df[t1_column].copy()
    else:
        df["Average_Temperature"] = df[[t1_column, t2_column]].mean(axis=1)

    # Humidity averaging
    if Out1H_nan > nan_threshold:
        df["Average_RH"] = df[h2_column].copy()
    elif Out2H_nan > nan_threshold:
        df["Average_RH"] = df[h1_column].copy()
    else:
        df["Average_RH"] = df[[h1_column, h2_column]].mean(axis=1)

    # PLOT
    plt.close('all')
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Temperature plot
    ax[0].plot(df[t1_column], df["flight_computer_pressure"], label="Out1_T", color='blue')
    ax[0].plot(df[t2_column], df["flight_computer_pressure"], label="Out2_T", color='orange')
    ax[0].plot(df["Average_Temperature"], df["flight_computer_pressure"], label="Average_T", color='red')
    ax[0].plot(df["smart_tether_T (deg C)"], df["flight_computer_pressure"], label="ST_T", color='green', linestyle='--')
    ax[0].set_xlabel("Temperature (°C)")
    ax[0].set_ylabel("Pressure (hPa)")
    ax[0].legend()
    ax[0].grid(True)
    ax[0].invert_yaxis()

    # Humidity plot
    ax[1].plot(df[h1_column], df["flight_computer_pressure"], label="Out1_RH", color='blue')
    ax[1].plot(df[h2_column], df["flight_computer_pressure"], label="Out2_RH", color='orange')
    ax[1].plot(df["Average_RH"], df["flight_computer_pressure"], label="Average_RH", color='red')
    ax[1].plot(df["smart_tether_%RH"], df["flight_computer_pressure"], label="ST_RH", color='green', linestyle='--')
    ax[1].set_xlabel("Relative Humidity (%)")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    return df
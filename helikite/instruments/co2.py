import logging
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from helikite.constants import constants
from helikite.instruments import Instrument
from helikite.processing.post.co2 import stp_moist_test

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)

class CO2(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "CO2"

    def file_identifier(self, first_lines_of_csv: list[str]) -> bool:
        columns = first_lines_of_csv[self.header].split(",")
        return "CO2" in columns

    def data_corrections(self, df, *args, **kwargs) -> pd.DataFrame:
        return df

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if "DateTime" in df.columns:
            # Flight computer V1 uses seconds since 1970-01-01
            df["DateTime"] = pd.to_datetime(df["DateTime"], unit="s", errors="coerce")
        if "Time" in df.columns:
            df["DateTime"] = pd.to_datetime(df["Time"], format="%y%m%d-%H%M%S")
            df.drop(columns=["Time"], inplace=True)

        df.set_index("DateTime", inplace=True)
        df.index = df.index.astype("datetime64[s]")

        return df

    def read_data(self) -> pd.DataFrame:
        time_columns = ["Time", "DateTime"]

        for i, time_column_to_use in enumerate(time_columns):
            columns = [col for col in self.dtype.keys() if col not in time_columns or col == time_column_to_use]
            try:
                df = pd.read_csv(
                    self.filename,
                    on_bad_lines="warn",
                    low_memory=False,
                    sep=",",
                    usecols=columns,
                )
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(self.dtype[col])
                df = df.dropna(subset=[time_column_to_use])
                return df

            except Exception as e:
                if i < len(time_columns) - 1:
                    logger.warning(f"Failed to read CO2 data from {self.filename} using columns {columns}:\n"
                                   f"{e}\n"
                                   + f"Trying again with different columns.")
                else:
                    logger.error(f"Failed to read CO2 data from {self.filename}: {e}")
                    raise

        raise ValueError(f"Failed to read CO2 data from {self.filename}")


co2 = CO2(
    name="co2",
    dtype={
        "DateTime": "Int64",
        "Time": "Int64",
        "CO2": "Float64",
    },
    na_values=[],
    header=0,
    comment="#",
)


def process_CO2_STP(df, min_threshold, max_threshold):
    """
    Process CO2 data to convert to STP moist and dry values, apply calibration,
    and filter out unrealistic values. Only processes if mean CO2 is above threshold.

    Parameters:

    lv0: DataFrame with raw data (expects specific column names)
        co2_threshold: Minimum mean CO2 value required to proceed


    Returns:


    lv0: Updated DataFrame with 'co2_CO2_moist' column (if processed)
      EDIT
      Using "T" and "P detrend " instead of T ref and FC_Pressure, and RH instead of RH ref"""
    co2_column_name = 'co2_CO2'
    co2_moist_column_name = 'co2_CO2_moist'

    mean = df[co2_column_name].mean()
    if mean is not pd.NA and mean < min_threshold:
        print(f"Skipping CO2 processing: "
              f"mean {co2_column_name} = {mean :.1f}, threshold = {min_threshold:.1f}")
        df[co2_moist_column_name] = pd.Series(pd.NA, index=df.index, dtype=df[co2_column_name].dtype)

        return df

    print("CO2 STP processing started")
    # Perform STP conversion
    print(_describe(df[co2_column_name].values))
    CO2_moist = stp_moist_test(df[co2_column_name],
                               df['Average_Temperature'], df['flight_computer_pressure'], df['Average_RH'])


    # Apply calibration (same for both, but only using moist in output)
    CO2_final_moist = 1.011742 * CO2_moist - 26.332125
    print(_describe(CO2_final_moist))

    # Assign to DataFrame and mask outliers
    df[co2_moist_column_name] = CO2_final_moist
    remove_outliers(df, co2_moist_column_name, min_threshold, max_threshold)

    print("CO2 STP processing complete.")

    # PLOT
    plt.figure(figsize=(8, 6))
    plt.plot(df[co2_column_name], df['Altitude'], label='Measured', color='blue', marker='.', linestyle='none')
    plt.plot(df[co2_moist_column_name], df['Altitude'], label='Corrected', color='red', marker='.', linestyle='none')
    plt.xlabel('CO2 concentration (ppm)', fontsize=12)
    plt.ylabel('Altitude (m)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df


def _describe(x: np.ndarray) -> dict[str, Number]:
    try:
        description = stats.describe(x, nan_policy='omit')
        description_concise = {
            "nobs": int(description.nobs.item()),
            "min": float(description.minmax[0].item()),
            "max": float(description.minmax[1].item()),
            "mean": float(description.mean.item()),
            "variance": float(description.variance.item()),
            "skewness": float(description.skewness.item()),
            "kurtosis": float(description.kurtosis.item()),
        }
        return description_concise
    except TypeError as e:
        logger.warning(f"Failed to describe array {x} with error: {e}. Check that not all values are NaN.")

    return {"No description available.": np.nan}



def remove_outliers(df: pd.DataFrame, column: str, min_threshold: Number, max_threshold: Number):
    mask = (df[column] <= min_threshold) | (df[column] >= max_threshold)
    print(f"{mask.sum()} outliers removed from {column} column based on thresholds {min_threshold} and {max_threshold}.")
    df[column] = df[column].where(~mask, np.nan)

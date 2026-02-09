"""
4) mSEMS ->
    mSEMS_103_220929_101343_INVERTED.txt (data to be plotted) (has pressure)
    mSEMS_103_220929_101343_READINGS.txt (high resolution raw data with
                                          houskeeping info, error messages)
                                          (has pressure)
    mSEMS_103_220929_101343_SCANS.txt (raw scan data with some houskeeping
                                       varaibles)

The mSEMS measures particles size distribution for particles from 8 nanometers
to roughly 300 nm. The number of bins can vary. Typically we use 60 log-spaced
bins but it can change. The time resolution depends on the amount of bins and
the scan time but it is typically between 1 and 2 minutes. I mormally do not
merge this data with the rest because of the courser time resolution and the
amount of variables.

The file provides some information on temperature, pressure etc. It then gives
the center diamater of each bin (i.e. Bin_Dia1, Bin_Dia2, ...) and then the
numbe rof particles per bin (i.e. Bin_Conc1, Bin_Conc2, ...).

-> because of the coarser time resolution, data is easier to be displayed as a
timeseries (with the addition of total particle concentration and altitude).

Houskeeping file: Look at READINGS (look at msems_err / cpc_err)

"""
import datetime
import logging
import pathlib

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from helikite.constants import constants
from helikite.instruments.base import Instrument, filter_columns_by_instrument

# Define logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


class MSEMSInverted(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "mSEMS_inv"

    @property
    def has_size_distribution(self) -> bool:
        return True

    def data_corrections(self, df, **kwargs):
        """Create new columns to plot bins"""

        # Form column names of all bins
        bin_diameter_columns = [col for col in df.columns if col.startswith("Bin_Dia")]

        # Calculate the mean of the bin diameters
        bin_diameter_average = df[bin_diameter_columns].mean()

        # Convert the bin diameters to log10
        bin_diameter_log = np.log10(bin_diameter_average)

        # Calculate first and last bin limits
        first_bin_radius = (
            bin_diameter_log[bin_diameter_columns[1]]
            - bin_diameter_log[bin_diameter_columns[0]]
        ) / 2
        first_bin_min = (
            bin_diameter_log[bin_diameter_columns[0]] - first_bin_radius
        )

        last_bin_radius = (
            bin_diameter_log[bin_diameter_columns[-1]]
            - bin_diameter_log[bin_diameter_columns[-2]]
        ) / 2

        last_bin_max = (
            bin_diameter_log[bin_diameter_columns[-1]] + last_bin_radius
        )

        # Create the bin limit columns and add the first limit column
        # (so there are total n_bins + 1)
        bin_limit_columns = [f"Bin_Lim{col.removeprefix('Bin_Dia')}" for col in bin_diameter_columns]
        bin_limit_columns.insert(0, "Bin_Lim0")

        # Add first and last bins
        df[bin_limit_columns[0]] = first_bin_min
        df[bin_limit_columns[-1]] = last_bin_max

        # Calculate the bin limits for all but the first and last
        for i, col in enumerate(bin_limit_columns[1:-1]):
            df[col] = (
                bin_diameter_log.iloc[i]
                + (bin_diameter_log.iloc[i + 1] - bin_diameter_log.iloc[i]) / 2
            )

        # Return values to the inverse of log10
        df[bin_limit_columns] = 10 ** df[bin_limit_columns]

        # Set the EndTime to the StartTime of the next row
        df["StartTime"] = df.index
        df["EndTime"] = pd.Series(df.index).shift(-1).values

        # Set the string time values to Date objects
        df["StartTime"] = pd.to_datetime(df["StartTime"])
        df["EndTime"] = pd.to_datetime(df["EndTime"])

        # The last row needs an end date, just add 1 minute
        if not df.empty:
            df.loc[df.index[-1], "EndTime"] = df.loc[
                df.index[-1], "StartTime"
            ] + pd.DateOffset(minutes=1)

        # Set the EndTime as one second less so that the bin end date does not
        # equal the start of the next bin start
        df["EndTime"] += pd.DateOffset(seconds=-1)

        # Reference msems_inverted dataframe before filling 3 minutes intervals
        self.msems_inverted_ref = df.copy()

        # Set scan direction from MSEMS scan dataframe
        msems_scan_df = getattr(msems_scan, "df", None)
        if msems_scan_df is not None:
            df['scan_direction'] = msems_scan_df['scan_direction']  # To have 0 / 1 values when changing scans

        df = df.resample("1s").ffill()

        # Repeat last timestamp for addditional 3 minutes in 1-second intervals
        if not df.empty:
            last_time = df.index[-1]
            extended_index = pd.date_range(
                start=last_time + pd.Timedelta(seconds=1),
                end=last_time + pd.Timedelta(minutes=3),
                freq="1s",
            )
            last_row = df.iloc[[-1]].copy()
            extension = pd.DataFrame(
                [last_row.values[0]] * len(extended_index),
                index=extended_index,
                columns=df.columns,
            )
            df = pd.concat([df, extension])

        return df

    def header_lines(self, file_path: str | pathlib.Path) -> list[str]:
        with open(file_path, encoding="utf-8", errors="replace") as in_file:
            # the header line of MSEMS Inverted is either the 0th or 55th line
            line = next(in_file)
            if self.expected_header_value in line:
                return [line]
            else:
                return super().header_lines(file_path)

    def file_identifier(self, first_lines_of_csv) -> bool:
        # the header line of MSEMS Inverted is either the 0th or 55th line
        # To match "...INVERTED.txt" file
        if self.expected_header_value in first_lines_of_csv[0]:
            return True

        if self.expected_header_value in first_lines_of_csv[self.header]:
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        Using values in the time_offset variable, correct DateTime index
        """

        df["DateTime"] = pd.to_datetime(
            df["#Date"] + " " + df["Time"], format="%y/%m/%d %H:%M:%S"
        )
        df.drop(columns=["#Date", "Time"], inplace=True)

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)
        df.index = df.index.floor("s").astype("datetime64[s]")

        return df

    def read_data(self) -> pd.DataFrame:
        # the header line of MSEMS Inverted is either the 0th or 55th line
        if len(self.header_lines(self.filename)) == 1:
            header = 0
        else:
            header = self.header

        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            na_values=self.na_values,
            skiprows=header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df

    def calculate_derived(self, df: pd.DataFrame, verbose: bool, *args, **kwargs) -> pd.DataFrame:
        """
        Calculate the total concentration from mSEMS measurements
        and add it to the dataframe.

        Parameters:
        df (pd.DataFrame): DataFrame containing mSEMS inverted data and Altitude.

        Returns:
        df (pd.DataFrame): Updated DataFrame with mSEMS dN columns inserted.
        """
        # Select only the mSEMS inverted columns
        filter_msems = filter_columns_by_instrument(df.columns, msems_inverted)
        msems_data = df[filter_msems]

        # Calculate dN for each bin and total concentration
        msems_dN = calcN(
            msems_data,
            start_column="msems_inverted_Bin_Dia1",
            end_column="msems_inverted_Bin_Dia60",
            start_conc="msems_inverted_Bin_Conc1",
            end_conc="msems_inverted_Bin_Conc60",
        )

        # Rename the columns corresponding to dN
        msems_dN = msems_dN.rename(
            columns=lambda col: col.replace(
                "msems_inverted_", "msems_inverted_dN_"
            )
        )

        # Insert msems_dN into df at the right position
        if filter_msems:
            last_msems_index = (
                df.columns.get_loc(filter_msems[-1]) + 1
            )  # Insert after this column
        else:
            last_msems_index = len(df.columns)

        df = pd.concat(
            [
                df.iloc[:, :last_msems_index],
                msems_dN,
                df.iloc[:, last_msems_index:],
            ],
            axis=1,
        )
        df = df.loc[
            :, ~df.columns.duplicated()
        ]  # remove potential duplicate columns

        return df

    def normalize(self, df: pd.DataFrame, verbose: bool, *args, **kwargs) -> pd.DataFrame:
        """
        Normalize mSEMS concentrations to STP conditions and insert the results
        right after the existing mSEMS columns.

        Parameters:
        df (pd.DataFrame): DataFrame containing mSEMS measurements and metadata.

        Returns:
        df (pd.DataFrame): Updated DataFrame with STP-normalized columns inserted.
        """
        # Constants for STP
        P_STP = 1013.25  # hPa
        T_STP = 273.15  # Kelvin

        # Measured conditions
        P_measured = df["flight_computer_pressure"]
        T_measured = df["Average_Temperature"] + 273.15  # Convert °C to Kelvin

        # Calculate the STP correction factor
        correction_factor = (P_measured / P_STP) * (T_STP / T_measured)

        # List of columns to correct
        columns_to_normalize = [
                                   col for col in df.columns if col.startswith("msems_inverted_Bin_Conc")
                               ] + ["msems_inverted_dN_totalconc"]

        # Create dictionary for normalized columns
        normalized_columns = {}

        for col in columns_to_normalize:
            if col in df.columns:
                normalized_columns[col + "_stp"] = df[col] * correction_factor

        # Find where to insert (after the last mSEMS-related column)
        msems_columns = [col for col in df.columns if col.startswith("msems_")]
        if msems_columns:
            last_msems_index = df.columns.get_loc(msems_columns[-1]) + 1
        else:
            last_msems_index = len(df.columns)

        # Insert normalized columns
        df = pd.concat(
            [
                df.iloc[:, :last_msems_index],
                pd.DataFrame(normalized_columns, index=df.index),
                df.iloc[:, last_msems_index:],
            ],
            axis=1,
        )

        return df

    def plot_raw_and_normalized(self, df: pd.DataFrame, verbose: bool, *args, **kwargs):
        """Plots mSEMS total concentration vs Altitude."""
        plt.figure(figsize=(8, 6))
        plt.plot(
            df["msems_inverted_dN_totalconc"],
            df["Altitude"],
            label="Measured",
            color="blue",
            marker=".",
            linestyle="none",
        )
        if "msems_inverted_dN_totalconc_stp" in df.columns:
            plt.plot(
                df["msems_inverted_dN_totalconc_stp"],
                df["Altitude"],
                label="STP-normalized",
                color="red",
                marker=".",
                linestyle="none",
            )
        plt.xlabel("mSEMS total concentration (cm$^{-3}$)", fontsize=12)
        plt.ylabel("Altitude (m)", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_distribution(self, df: pd.DataFrame, verbose: bool,
                          time_start: datetime.datetime, time_end: datetime.datetime,
                          subplot: tuple[plt.Figure, plt.Axes] | None = None):
        """
        Plots mSEMS size distribution and total concentration from a given DataFrame.

        Parameters:
        - df: pandas DataFrame with required mSEMS columns
        - time_start, time_end: optional pandas.Timestamp or str for limiting the time range
        """
        if subplot is not None:
            fig, ax = subplot
            is_custom_subplot = True
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            is_custom_subplot = False

        if time_start is not None:
            df = df[df.index >= pd.to_datetime(time_start)]
        if time_end is not None:
            df = df[df.index <= pd.to_datetime(time_end)]

        # Define diameter and concentration column ranges
        start_dia = 'msems_inverted_Bin_Dia1'
        end_dia = 'msems_inverted_Bin_Dia60'
        start_conc = self.column_name(df, 'msems_inverted_Bin_Conc1_stp')
        end_conc = self.column_name(df, 'msems_inverted_Bin_Conc60_stp')

        # Get diameter bin averages
        if start_dia and end_dia in df.columns:
            bin_diameter_averages = df.loc[:, start_dia:end_dia].mean()
            if bin_diameter_averages.isna().any():
                logger.warning("NaNs are present in `bin_diameter_averages`. "
                               "Setting `bin_diameter_averages` to the default `MSEMS_BIN_DIAMETER_AVERAGES`")
                bin_diameter_averages = MSEMS_BIN_DIAMETER_AVERAGES
        else:
            bin_diameter_averages = MSEMS_BIN_DIAMETER_AVERAGES

        # Prepare the concentration data
        counts = df.loc[:, start_conc:end_conc]
        counts.index = df.index
        counts = counts.astype(float)
        counts = counts.dropna(how='any') if not counts.isna().all().all() else counts
        counts = counts.clip(lower=1)

        vmax_value = np.nanmax(counts.values) if not counts.isna().all().all() else np.nan
        print(f"max value ({self.name}): {vmax_value}")

        # Create the 2D meshgrid
        xx, yy = np.meshgrid(counts.index.values, bin_diameter_averages)
        Z = counts.values.T

        # Plot the pcolormesh
        norm = mcolors.LogNorm(vmin=1, vmax=1000)
        mesh = ax.pcolormesh(xx, yy, Z, cmap='viridis', norm=norm, shading="gouraud")

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = inset_axes(
            ax,
            width="1.5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.08, -0.025, 1, 1),
            bbox_transform=ax.transAxes
        )
        cb = fig.colorbar(mesh, cax=cax, orientation="vertical")
        cb.set_label("dN/dlogD$_p$ (cm$^{-3}$)", fontsize=13, fontweight="bold")
        cb.ax.tick_params(labelsize=11)

        if not is_custom_subplot:
            # Custom x-axis formatter
            class CustomDateFormatter(mdates.DateFormatter):
                def __init__(
                    self, fmt="%H:%M", date_fmt="%Y-%m-%d %H:%M", *args, **kwargs
                ):
                    super().__init__(fmt, *args, **kwargs)
                    self.date_fmt = date_fmt
                    self.prev_date = None

                def __call__(self, x, pos=None):
                    date = mdates.num2date(x)
                    current_date = date.date()
                    if self.prev_date != current_date:
                        self.prev_date = current_date
                        return date.strftime(self.date_fmt)
                    else:
                        return date.strftime(self.fmt)

            ax.xaxis.set_major_formatter(
                CustomDateFormatter(fmt="%H:%M", date_fmt="%Y-%m-%d %H:%M")
            )
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            ax.tick_params(axis="x", rotation=90, labelsize=11)

        # Axis labels and formatting
        ax.set_yscale("log")
        ax.set_ylabel("Part. Diameter (nm)", fontsize=12, fontweight="bold")
        ax.set_ylim(8, 250)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.6, axis="x")

        if not is_custom_subplot:
            ax.set_xlabel("Time", fontsize=12, fontweight="bold", labelpad=10)
            ax.set_title(
                "mSEMS size distribution and total concentration",
                fontsize=13,
                fontweight="bold",
            )

        # Secondary y-axis for total concentration
        total_conc = df[self.column_name(df, "msems_inverted_dN_totalconc_stp")]
        total_conc_max = total_conc.max() if not total_conc.isna().all() else 2000
        ax2 = ax.twinx()
        ax2.scatter(df.index, total_conc, color="red", marker='.')
        ax2.set_ylabel("mSEMS conc (cm$^{-3}$)", fontsize=12, fontweight="bold", color="red", labelpad=8)
        ax2.tick_params(axis="y", labelsize=11, colors="red")
        ax2.set_ylim(0, total_conc_max * 1.1)

        if not is_custom_subplot:
            plt.subplots_adjust(bottom=0.25, right=0.85)
            plt.show()


class MSEMSReadings(Instrument):
    # To match a "...READINGS.txt" file
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "mSEMS"

    def file_identifier(self, first_lines_of_csv) -> bool:
        if (
            "#mSEMS" in first_lines_of_csv[0]
            and "#YY/MM/DD" in first_lines_of_csv[self.header]
            and "mono_dia" in first_lines_of_csv[self.header]
        ):
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        Using values in the time_offset variable, correct DateTime index
        """

        df["DateTime"] = pd.to_datetime(
            df["#YY/MM/DD"] + " " + df["HR:MN:SC"], format="%y/%m/%d %H:%M:%S"
        )
        df.drop(columns=["#YY/MM/DD", "HR:MN:SC"], inplace=True)

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)
        df.index = df.index.floor("s").astype("datetime64[s]")

        return df

    def data_corrections(self, df, *args, **kwargs):
        return df

    def read_data(self) -> pd.DataFrame:

        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            na_values=self.na_values,
            skiprows=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df


class MSEMSScan(Instrument):
    # To match a "...SCAN.txt" file
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "mSEMS_scan"

    def file_identifier(self, first_lines_of_csv) -> bool:
        if (
            "#mSEMS" in first_lines_of_csv[0]
            and "#scan_conf" in first_lines_of_csv[31]
            and "scan_direction" in first_lines_of_csv[self.header]
        ):
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        Using values in the time_offset variable, correct DateTime index
        """

        df["DateTime"] = pd.to_datetime(
            df["#YY/MM/DD"] + " " + df["HR:MN:SC"], format="%y/%m/%d %H:%M:%S"
        )
        df.drop(columns=["#YY/MM/DD", "HR:MN:SC"], inplace=True)

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)
        df.index = df.index.floor("s").astype("datetime64[s]")

        return df

    def data_corrections(self, df, *args, **kwargs):
        df.index = pd.to_datetime(df.index)
        self.msems_scan_ref = (
            df.copy()
        )  # Reference msems_inverted dataframe (before filling empty 3 minutes intervals)
        df = df.resample(
            "1s"
        ).ffill()  # Fill in data for every second based on 3 min measurements (for cross-correlation)

        return df

    def read_data(self) -> pd.DataFrame:

        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            na_values=self.na_values,
            skiprows=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df


msems_scan = MSEMSScan(
    name="msems_scan",
    header=55,
    delimiter="\t",
    dtype={
        "#YY/MM/DD": "str",
        "HR:MN:SC": "str",
        "scan_direction": "Int64",
        "actual_max_dia": "Int64",
        "scan_max_volts": "Float64",
        "scan_min_volts": "Float64",
        "sheath_flw_avg": "Float64",
        "sheath_flw_stdev": "Float64",
        "mcpc_smpf_avg": "Float64",
        "mcpc_smpf_stdev": "Float64",
        "press_avg": "Float64",
        "press_stdev": "Float64",
        "temp_avg": "Float64",
        "temp_stdev": "Float64",
        "sheath_rh_avg": "Float64",
        "sheath_rh_stdev": "Float64",
        "bin1": "Int64",
        "bin2": "Int64",
        "bin3": "Int64",
        "bin4": "Int64",
        "bin5": "Int64",
        "bin6": "Int64",
        "bin7": "Int64",
        "bin8": "Int64",
        "bin9": "Int64",
        "bin10": "Int64",
        "bin11": "Int64",
        "bin12": "Int64",
        "bin13": "Int64",
        "bin14": "Int64",
        "bin15": "Int64",
        "bin16": "Int64",
        "bin17": "Int64",
        "bin18": "Int64",
        "bin19": "Int64",
        "bin20": "Int64",
        "bin21": "Int64",
        "bin22": "Int64",
        "bin23": "Int64",
        "bin24": "Int64",
        "bin25": "Int64",
        "bin26": "Int64",
        "bin27": "Int64",
        "bin28": "Int64",
        "bin29": "Int64",
        "bin30": "Int64",
        "bin31": "Int64",
        "bin32": "Int64",
        "bin33": "Int64",
        "bin34": "Int64",
        "bin35": "Int64",
        "bin36": "Int64",
        "bin37": "Int64",
        "bin38": "Int64",
        "bin39": "Int64",
        "bin40": "Int64",
        "bin41": "Int64",
        "bin42": "Int64",
        "bin43": "Int64",
        "bin44": "Int64",
        "bin45": "Int64",
        "bin46": "Int64",
        "bin47": "Int64",
        "bin48": "Int64",
        "bin49": "Int64",
        "bin50": "Int64",
        "bin51": "Int64",
        "bin52": "Int64",
        "bin53": "Int64",
        "bin54": "Int64",
        "bin55": "Int64",
        "bin56": "Int64",
        "bin57": "Int64",
        "bin58": "Int64",
        "bin59": "Int64",
        "bin60": "Int64",
        "msems_errs": "Int64",
        "mcpc_smpf": "Float64",
        "mcpc_satf": "Float64",
        "mcpc_cndt": "Float64",
        "mcpc_satt": "Float64",
        "mcpc_errs": "Int64",
    },
    export_order=710,
    pressure_variable="press_avg",
    cols_export=[],
    cols_housekeeping=[],
    cols_final=[],
)

# To match a "...READINGS.txt" file
msems_readings = MSEMSReadings(
    name="msems_readings",
    header=31,
    delimiter="\t",
    dtype={
        "#YY/MM/DD": "str",
        "HR:MN:SC": "str",
        "msems_mode": "Int64",
        "mono_dia": "Int64",
        "sheath_sp": "Float64",
        "sheath_rh": "Int64",
        "sheath_temp": "Float64",
        "pressure": "Int64",
        "lfe_temp": "Float64",
        "sheath_flow": "Float64",
        "sheath_pwr": "Int64",
        "impct_prs": "Float64",
        "hv_volts": "Float64",
        "hv_dac": "Int64",
        "sd_install": "Int64",
        "ext_volts": "Float64",
        "msems_errs": "Float64",
        "mcpc_hrtb": "Float64",
        "mcpc_smpf": "Float64",
        "mcpc_satf": "Float64",
        "mcpc_cndt": "Float64",
        "mcpc_satt": "Float64",
        "mcpcpwr": "Int64",
        "mcpcpmp": "Int64",
        "sd_save": "Int64",
        "mcpc_errs": "Int64",
        "mcpc_a_conc": "Float64",
        "mcpc_a_cnt": "Int64",
    },
    export_order=700,
    pressure_variable="pressure",
    cols_export=[],
    cols_housekeeping=[],
    cols_final=[],
)

# To match a "...INVERTED.txt" file
msems_inverted = MSEMSInverted(
    name="msems_inverted",
    header=55,
    delimiter="\t",
    dtype={
        "#Date": "str",
        "Time": "str",
        "Temp(C)": "Float64",
        "Press(hPa)": "Float64",
        "NumBins": "Int64",
        "Bin_Dia1": "Float64",
        "Bin_Dia2": "Float64",
        "Bin_Dia3": "Float64",
        "Bin_Dia4": "Float64",
        "Bin_Dia5": "Float64",
        "Bin_Dia6": "Float64",
        "Bin_Dia7": "Float64",
        "Bin_Dia8": "Float64",
        "Bin_Dia9": "Float64",
        "Bin_Dia10": "Float64",
        "Bin_Dia11": "Float64",
        "Bin_Dia12": "Float64",
        "Bin_Dia13": "Float64",
        "Bin_Dia14": "Float64",
        "Bin_Dia15": "Float64",
        "Bin_Dia16": "Float64",
        "Bin_Dia17": "Float64",
        "Bin_Dia18": "Float64",
        "Bin_Dia19": "Float64",
        "Bin_Dia20": "Float64",
        "Bin_Dia21": "Float64",
        "Bin_Dia22": "Float64",
        "Bin_Dia23": "Float64",
        "Bin_Dia24": "Float64",
        "Bin_Dia25": "Float64",
        "Bin_Dia26": "Float64",
        "Bin_Dia27": "Float64",
        "Bin_Dia28": "Float64",
        "Bin_Dia29": "Float64",
        "Bin_Dia30": "Float64",
        "Bin_Dia31": "Float64",
        "Bin_Dia32": "Float64",
        "Bin_Dia33": "Float64",
        "Bin_Dia34": "Float64",
        "Bin_Dia35": "Float64",
        "Bin_Dia36": "Float64",
        "Bin_Dia37": "Float64",
        "Bin_Dia38": "Float64",
        "Bin_Dia39": "Float64",
        "Bin_Dia40": "Float64",
        "Bin_Dia41": "Float64",
        "Bin_Dia42": "Float64",
        "Bin_Dia43": "Float64",
        "Bin_Dia44": "Float64",
        "Bin_Dia45": "Float64",
        "Bin_Dia46": "Float64",
        "Bin_Dia47": "Float64",
        "Bin_Dia48": "Float64",
        "Bin_Dia49": "Float64",
        "Bin_Dia50": "Float64",
        "Bin_Dia51": "Float64",
        "Bin_Dia52": "Float64",
        "Bin_Dia53": "Float64",
        "Bin_Dia54": "Float64",
        "Bin_Dia55": "Float64",
        "Bin_Dia56": "Float64",
        "Bin_Dia57": "Float64",
        "Bin_Dia58": "Float64",
        "Bin_Dia59": "Float64",
        "Bin_Dia60": "Float64",
        "Bin_Conc1": "Float64",
        "Bin_Conc2": "Float64",
        "Bin_Conc3": "Float64",
        "Bin_Conc4": "Float64",
        "Bin_Conc5": "Float64",
        "Bin_Conc6": "Float64",
        "Bin_Conc7": "Float64",
        "Bin_Conc8": "Float64",
        "Bin_Conc9": "Float64",
        "Bin_Conc10": "Float64",
        "Bin_Conc11": "Float64",
        "Bin_Conc12": "Float64",
        "Bin_Conc13": "Float64",
        "Bin_Conc14": "Float64",
        "Bin_Conc15": "Float64",
        "Bin_Conc16": "Float64",
        "Bin_Conc17": "Float64",
        "Bin_Conc18": "Float64",
        "Bin_Conc19": "Float64",
        "Bin_Conc20": "Float64",
        "Bin_Conc21": "Float64",
        "Bin_Conc22": "Float64",
        "Bin_Conc23": "Float64",
        "Bin_Conc24": "Float64",
        "Bin_Conc25": "Float64",
        "Bin_Conc26": "Float64",
        "Bin_Conc27": "Float64",
        "Bin_Conc28": "Float64",
        "Bin_Conc29": "Float64",
        "Bin_Conc30": "Float64",
        "Bin_Conc31": "Float64",
        "Bin_Conc32": "Float64",
        "Bin_Conc33": "Float64",
        "Bin_Conc34": "Float64",
        "Bin_Conc35": "Float64",
        "Bin_Conc36": "Float64",
        "Bin_Conc37": "Float64",
        "Bin_Conc38": "Float64",
        "Bin_Conc39": "Float64",
        "Bin_Conc40": "Float64",
        "Bin_Conc41": "Float64",
        "Bin_Conc42": "Float64",
        "Bin_Conc43": "Float64",
        "Bin_Conc44": "Float64",
        "Bin_Conc45": "Float64",
        "Bin_Conc46": "Float64",
        "Bin_Conc47": "Float64",
        "Bin_Conc48": "Float64",
        "Bin_Conc49": "Float64",
        "Bin_Conc50": "Float64",
        "Bin_Conc51": "Float64",
        "Bin_Conc52": "Float64",
        "Bin_Conc53": "Float64",
        "Bin_Conc54": "Float64",
        "Bin_Conc55": "Float64",
        "Bin_Conc56": "Float64",
        "Bin_Conc57": "Float64",
        "Bin_Conc58": "Float64",
        "Bin_Conc59": "Float64",
        "Bin_Conc60": "Float64",
    },
    expected_header_value="#Date\tTime\tTemp(C)\tPress(hPa)\tNumBins\tBin_Dia1\tBin_Dia2\tBin_Dia3",
    export_order=720,
    pressure_variable="Press(hPa)",
    cols_export=[],
    cols_housekeeping=[],
    cols_final=[f"Bin_Conc{i}_stp" for i in range(1, 61)] + ["dN_totalconc_stp"],
    coupled_columns=[
        [f"msems_inverted_Bin_Conc{i}" for i in range(1, 61)] +
        [f"msems_inverted_dN_Bin_Conc{i}" for i in range(1, 61)] +
        ["msems_inverted_dN_totalconc"],
    ],
    rename_dict={f'msems_inverted_Bin_Conc{i}_stp': f'mSEMS_Bin_Conc{i}' for i in range(1, 61)} |
                {'msems_inverted_dN_totalconc_stp': 'mSEMS_total_N'},
)


MSEMS_BIN_DIAMETER_AVERAGES = np.array([
    8.209426, 8.652978, 9.120773, 9.614176, 10.134633, 10.683676, 11.262924,
    11.874097, 12.519017, 13.199618, 13.917952, 14.676198, 15.476672,
    16.319869, 17.212476, 18.155273, 19.151085, 20.203080, 21.314636,
    22.489353, 23.731074, 25.043907, 26.432241, 27.900779, 29.454554,
    31.098968, 32.839819, 34.683337, 36.636229, 38.705716, 40.899590,
    43.226267, 45.694846, 48.315185, 51.097974, 54.054825, 57.198371,
    60.542379, 64.101873, 67.893280, 71.934588, 76.245533, 80.847801,
    85.765265, 91.024253, 96.653846, 102.686223, 109.157047, 116.105907,
    123.576809, 131.618741, 140.286300, 149.640401, 159.749066, 170.688305,
    182.543091, 195.408432, 209.392962, 224.611976, 241.211823
])


def calcN(df, start_column, end_column, start_conc, end_conc):
    """
    Calculate dN (differential number concentration) using logarithmic bin diameters.

    Parameters:
        df (pd.DataFrame): DataFrame containing bin diameters and concentrations.
        start_column (str): First bin diameter column name.
        end_column (str): Last bin diameter column name.
        start_conc (str): First bin concentration column name.
        end_conc (str): Last bin concentration column name.

    Returns:
        pd.DataFrame: DataFrame containing dN values and total concentration.
    """

    bin_diams = df.loc[:, start_column:end_column].astype(np.float64)  # iloc[:,4:63]#[:,4:44]
    # print(bin_diams.columns)
    bin_concs = df.loc[:, start_conc:end_conc].astype(np.float64)  # iloc[:,64:123]#[:,44:84]
    # print(bin_concs.columns)
    CMD = df.loc[:, start_column:end_column].astype(np.float64)  # .iloc[:,3:63]

    diff_df = CMD.diff(axis=1)
    delta = diff_df / 2
    upper_boundary = CMD + delta
    lower_boundary = CMD - delta

    log_lower = np.log10(lower_boundary)
    log_upper = np.log10(upper_boundary)
    dlogDp = log_upper - log_lower

    BD1_upper = lower_boundary["msems_inverted_Bin_Dia2"]
    BD1_lower = df["msems_inverted_Bin_Dia1"] - delta["msems_inverted_Bin_Dia2"]
    BD1 = np.log10(BD1_upper) - np.log10(BD1_lower)
    dlogDp["msems_inverted_Bin_Dia1"] = BD1

    # Compute dN
    dN = bin_concs.mul(dlogDp.values).astype("Float64")

    # Sum without NaN issues
    dN["msems_inverted_totalconc"] = dN.sum(axis=1, skipna=True, min_count=1)

    return dN

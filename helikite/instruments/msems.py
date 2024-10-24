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

from helikite.instruments.base import Instrument
import pandas as pd
import numpy as np


class MSEMSInverted(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "msems_inverted"

    def data_corrections(self, df, **kwargs):
        """Create new columns to plot bins"""
        bins = df.groupby("NumBins").all().index.to_list()
        if len(bins) != 1:
            # Check that there is only one single value.
            raise ValueError(
                "There are multiple bins in this dataset. " "Cannot proceed"
            )
        else:
            # Unpack the single value list to a single integer
            bins = bins[0]
        # Form column names of all bins
        bin_diameter_columns = [f"Bin_Dia{i}" for i in range(1, bins + 1)]

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
        bin_limit_columns = [f"Bin_Lim{i}" for i in range(1, bins + 1)]
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
        df.loc[df.index[-1], "EndTime"] = df.loc[
            df.index[-1], "StartTime"
        ] + pd.DateOffset(minutes=1)

        # Set the EndTime as one second less so that the bin end date does not
        # equal the start of the next bin start
        df["EndTime"] += pd.DateOffset(seconds=-1)

        return df

    def file_identifier(self, first_lines_of_csv) -> bool:
        # To match "...INVERTED.txt" file
        if (
            "#Date\tTime\tTemp(C)\tPress(hPa)\tNumBins\tBin_Dia1\t"
            "Bin_Dia2\tBin_Dia3"
        ) in first_lines_of_csv[0]:
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

        return df

    def read_data(self) -> pd.DataFrame:

        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            na_values=self.na_values,
            header=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df


class MSEMSReadings(Instrument):
    # To match a "...READINGS.txt" file
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "msems_readings"

    def file_identifier(self, first_lines_of_csv) -> bool:
        if (
            "#mSEMS" in first_lines_of_csv[0]
            and "#YY/MM/DD" in first_lines_of_csv[31]
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

        return df

    def data_corrections(self, df, *args, **kwargs):
        return df

    def read_data(self) -> pd.DataFrame:

        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            na_values=self.na_values,
            header=self.header,
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
        self.name = "msems_scan"

    def file_identifier(self, first_lines_of_csv) -> bool:
        if (
            "#mSEMS" in first_lines_of_csv[0]
            and "#scan_conf" in first_lines_of_csv[31]
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

        return df

    def data_corrections(self, df, *args, **kwargs):
        return df

    def read_data(self) -> pd.DataFrame:

        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            na_values=self.na_values,
            header=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )

        return df


msems_scan = MSEMSScan(
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
)

# To match a "...READINGS.txt" file
msems_readings = MSEMSReadings(
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
)

# To match a "...READINGS.txt" file
msems_inverted = MSEMSInverted(
    # header=31,
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
    export_order=720,
    cols_export=[],
    cols_housekeeping=[],
)

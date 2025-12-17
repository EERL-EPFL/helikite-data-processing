""" Single Channel Tricolor Absorption Photometer (STAP)
The STAP measures the light absorption of particles deposited on a filter.

Resolution: 1 sec

Variables to keep: Everything

Time is is seconds since 1904-01-01 (weird starting date for Igor software)
"""

from helikite.instruments.base import Instrument
import pandas as pd


class STAP(Instrument):
    """This class is a processed version of the STAP data

    For outputs directly from the instrument, use the STAPRaw class
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "STAP"

    def data_corrections(self, df, *args, **kwargs):
        return df

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        Using values in the time_offset variable, correct DateTime index
        """
        # Column 'datetimes' represents seconds since 1904-01-01
        df["DateTime"] = pd.to_datetime(
            pd.Timestamp("1904-01-01")
            + pd.to_timedelta(df["datetimes"], unit="s")
        )
        df.drop(columns=["datetimes"], inplace=True)

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)
        df.index = df.index.astype("datetime64[s]")

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


class STAPRaw(Instrument):
    """This instrument class is for the raw STAP data"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "STAP_raw"

    def data_corrections(self, df, *args, **kwargs):
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        return df

    def file_identifier(self, first_lines_of_csv) -> bool:
        if (
            "#YY/MM/DD\tHR:MN:SC\tinvmm_r\tinvmm_g\tinvmm_b\tred_smp\t"
        ) in first_lines_of_csv[self.header]:
            return True

        return False

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the DateTime as index of the dataframe and correct if needed

        Instrument contains date and time separately and appears to
        include an extra whitespace in the field of each of those two columns
        """

        # Combine both date and time columns into one, strip extra whitespace
        df["DateTime"] = pd.to_datetime(
            df["#YY/MM/DD"].str.strip() + " " + df["HR:MN:SC"].str.strip(),
            format="%y/%m/%d %H:%M:%S",
        )
        df.drop(columns=["#YY/MM/DD", "HR:MN:SC"], inplace=True)

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)

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


stap = STAP(
    name="stap",
    dtype={
        "datetimes": "Int64",
        "sample_press_mbar": "Float64",
        "sample_temp_C": "Float64",
        "sigmab": "Float64",
        "sigmag": "Float64",
        "sigmar": "Float64",
        "sigmab_smth": "Float64",
        "sigmag_smth": "Float64",
        "sigmar_smth": "Float64",
    },
    expected_header_value=(
        "datetimes,sample_press_mbar,sample_temp_C,sigmab,sigmag,sigmar,"
        "sigmab_smth,sigmag_smth,sigmar_smth\n"
    ),
    na_values=["NAN"],
    export_order=500,
    cols_export=[
        "sample_press_mbar",
        "sample_temp_C",
        "sigmab",
        "sigmag",
        "sigmar",
        "sigmab_smth",
        "sigmag_smth",
        "sigmar_smth",
    ],
    cols_housekeeping=[
        "sample_press_mbar",
        "sample_temp_C",
        "sigmab",
        "sigmag",
        "sigmar",
        "sigmab_smth",
        "sigmag_smth",
        "sigmar_smth",
    ],
    pressure_variable="sample_press_mbar",
)


stap_raw = STAPRaw(
    name="stap_raw",
    header=29,
    delimiter="\t",
    dtype={
        "#YY/MM/DD": "str",
        "HR:MN:SC": "str",
        "invmm_r": "Float64",
        "invmm_g": "Float64",
        "invmm_b": "Float64",
        "red_smp": "Int64",
        "red_ref": "Int64",
        "grn_smp": "Int64",
        "grn_ref": "Int64",
        "blu_smp": "Int64",
        "blu_ref": "Int64",
        "blk_smp": "Int64",
        "blk_ref": "Int64",
        "smp_flw": "Float64",
        "smp_tmp": "Float64",
        "smp_prs": "Int64",
        "pump_pw": "Int64",
        "psvolts": "Float64",
        "err_rpt": "Int64",
        "cntdown": "Int64",
        "sd_stat": "Float64",
        "fltstat": "Int64",
        "flow_sp": "Int64",
        "intervl": "Int64",
        "stapctl": "Int64",
    },
    na_values=["-0.00*", "0.00* "],  # Values with * represent sensor warming
    export_order=520,
    cols_export=["invmm_r", "invmm_g", "invmm_b"],
    cols_housekeeping=[
        "invmm_r",
        "invmm_g",
        "invmm_b",
        "red_smp",
        "red_ref",
        "grn_smp",
        "grn_ref",
        "blu_smp",
        "blu_ref",
        "blk_smp",
        "blk_ref",
        "smp_flw",
        "smp_tmp",
        "smp_prs",
        "pump_pw",
        "psvolts",
        "err_rpt",
        "cntdown",
        "sd_stat",
        "fltstat",
        "flow_sp",
        "intervl",
        "stapctl",
    ],
    pressure_variable="smp_prs",
)

def STAP_STP_normalization(df):
    """
    Normalize STAP measurements to STP conditions and plot the results.

    Parameters:
    df (pd.DataFrame): DataFrame containing STAP measurements and necessary metadata
                       like 'flight_computer_pressure' and 'Average_Temperature'.

    Returns:
    df (pd.DataFrame): Updated DataFrame with new STP-normalized columns added.
    """
    import matplotlib.pyplot as plt
    plt.close('all')

    # Constants for STP
    P_STP = 1013.25  # hPa
    T_STP = 273.15  # Kelvin

    # Measured conditions
    P_measured = df["flight_computer_pressure"]
    T_measured = df["Average_Temperature"] + 273.15  # Convert °C to Kelvin

    # Calculate the STP correction factor
    correction_factor = (P_measured / P_STP) * (T_STP / T_measured)

    # Columns to normalize
    sigmab_column = 'stap_sigmab_smth'
    sigmag_column = 'stap_sigmag_smth'
    sigmar_column = 'stap_sigmar_smth'
    columns_to_normalize = [sigmab_column, sigmag_column, sigmar_column]

    # Dictionary to hold new columns
    normalized_columns = {}

    for col in columns_to_normalize:
        if col in df.columns:
            normalized_columns[col + '_stp'] = df[col] * correction_factor

    # Insert the new columns
    df = pd.concat(
        [df, pd.DataFrame(normalized_columns, index=df.index)],
        axis=1
    )

    # Define colors for each variable
    colors = {
        sigmab_column: 'blue',
        sigmag_column: 'green',
        sigmar_column: 'red'
    }

    # Plot example (optional)
    plt.figure(figsize=(8, 6))
    for col in columns_to_normalize:
        if col + '_stp' in df.columns:
            plt.plot(df[col], df['Altitude'], label=f'{col} measured', color=colors.get(col, 'black'), alpha=0.5)
            plt.plot(df[col + "_stp"], df['Altitude'], label=f'{col} STP-normalized', linestyle='--',
                     color=colors.get(col, 'black'))

    plt.xlabel('Signal')
    plt.ylabel('Altitude [m]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df
import pandas as pd

from helikite.instruments import Instrument


class MAeth(Instrument):
    def __repr__(self):
        return "mAeth"

    def data_corrections(self, df, *args, **kwargs) -> pd.DataFrame:
        df["Pressure"] = df["Internal pressure (Pa)"].copy() / 100
        df["Pressure"].replace(0, pd.NA, inplace=True)

        return df

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df["DateTime"] = pd.to_datetime(
            df["Date local (yyyy/MM/dd)"].astype(str) + " " + df["Time local (hh:mm:ss)"].astype(str),
            format="%Y/%m/%d %H:%M:%S",
        ) - pd.to_timedelta(df["Timezone offset (mins)"], unit="min")

        df.drop(columns=["Date local (yyyy/MM/dd)", "Time local (hh:mm:ss)"], inplace=True)

        # Define the datetime column as the index
        df.set_index("DateTime", inplace=True)
        df.index = df.index.floor('s').astype("datetime64[s]")

        return df

    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.filename,
            encoding='Latin-1',
            header=self.header,
            sep=',',
            dtype=self.dtype,
        )

        return df

    def normalize(self, df: pd.DataFrame, reference_instrument: Instrument,
                  verbose: bool, *args, **kwargs) -> pd.DataFrame:
        pressure = df[f"{reference_instrument.name}_pressure"]
        temperature = df["Average_Temperature"]

        df["maeth_UV BC1_STP"] = MAeth._stp_convert_dry(df["maeth_UV BC1"], temperature, pressure)
        df["maeth_Blue BC1_STP"] = MAeth._stp_convert_dry(df["maeth_Blue BC1"], temperature, pressure)
        df["maeth_Green BC1_STP"] = MAeth._stp_convert_dry(df["maeth_Green BC1"], temperature, pressure)
        df["maeth_Red BC1_STP"] = MAeth._stp_convert_dry(df["maeth_Red BC1"], temperature, pressure)
        df["maeth_IR BC1_STP"] = MAeth._stp_convert_dry(df["maeth_IR BC1"], temperature, pressure)

        df["maeth_eBC_375"] = MAeth._convert_babs_stp_to_ebc(df["maeth_UV BC1_STP"], 375)
        df["maeth_eBC_470"] = MAeth._convert_babs_stp_to_ebc(df["maeth_Blue BC1_STP"], 470)
        df["maeth_eBC_528"] = MAeth._convert_babs_stp_to_ebc(df["maeth_Green BC1_STP"], 528)
        df["maeth_eBC_625"] = MAeth._convert_babs_stp_to_ebc(df["maeth_Red BC1_STP"], 625)
        df["maeth_eBC_880"] = MAeth._convert_babs_stp_to_ebc(df["maeth_IR BC1_STP"], 880)

        return df

    @staticmethod
    def _stp_convert_dry(x, t, p1):
        p = p1 * 100
        t = t + 273.15
        v_stp = (273.15 / t) * (p / 101315)

        return x / v_stp

    @staticmethod
    def _convert_babs_stp_to_ebc(babs_stp, wavelength):
        """
        Convert STP-corrected absorption (Mm^-1) to eBC (Âµg/m3).
        """
        MAC = {
            375: 18.52,
            470: 14.67,
            528: 13.10,
            625: 10.84,
            880: 7.78
        }

        return babs_stp / MAC[wavelength]


maeth = MAeth(
    name="maeth",
    expected_header_value="""
    "UV BC1","UV BC2","UV BCc","Blue BC1","Blue BC2","Blue BCc","Green BC1","Green BC2","Green BCc","Red BC1","Red BC2","Red BCc","IR BC1","IR BC2","IR BCc"
    """,
    dtype={
        "Serial number": "str",
        "Datum ID": "Int64",
        "Session ID": "Int64",
        "Data format version": "str",
        "Firmware version": "str",
        "App version": "str",
        "Date / time local": "str",
        "Timezone offset (mins)": "Int64",
        "Date local (yyyy/MM/dd)": "str",
        "Time local (hh:mm:ss)": "str",
        "GPS lat (ddmm.mmmmm)": "Float64",
        "GPS long (dddmm.mmmmm)": "Float64",
        "GPS speed (km/h)": "Float64",
        "GPS sat count": "Int64",
        "Timebase (s)": "Int64",
        "Status": "Int64",
        "Battery remaining (%)": "Int64",
        "Accel X": "Int64",
        "Accel Y": "Int64",
        "Accel Z": "Int64",
        "Tape position": "Int64",
        "Flow setpoint (mL/min)": "Int64",
        "Flow total (mL/min)": "Float64",
        "Flow1 (mL/min)": "Float64",
        "Flow2 (mL/min)": "Float64",
        "Sample temp (C)": "Float64",
        "Sample RH (%)": "Float64",
        "Sample dewpoint (C)": "Float64",
        "Internal pressure (Pa)": "Int64",
        "Internal temp (C)": "Float64",
        "Optical config": "str",
        "UV Sen1": "Int64",
        "UV Sen2": "Int64",
        "UV Ref": "Int64",
        "UV ATN1": "Float64",
        "UV ATN2": "Float64",
        "UV K": "Float64",
        "Blue Sen1": "Int64",
        "Blue Sen2": "Int64",
        "Blue Ref": "Int64",
        "Blue ATN1": "Float64",
        "Blue ATN2": "Float64",
        "Blue K": "Float64",
        "Green Sen1": "Int64",
        "Green Sen2": "Int64",
        "Green Ref": "Int64",
        "Green ATN1": "Float64",
        "Green ATN2": "Float64",
        "Green K": "Float64",
        "Red Sen1": "Int64",
        "Red Sen2": "Int64",
        "Red Ref": "Int64",
        "Red ATN1": "Float64",
        "Red ATN2": "Float64",
        "Red K": "Float64",
        "IR Sen1": "Int64",
        "IR Sen2": "Int64",
        "IR Ref": "Int64",
        "IR ATN1": "Float64",
        "IR ATN2": "Float64",
        "IR K": "Float64",
        "UV BC1": "Float64",
        "UV BC2": "Float64",
        "UV BCc": "Float64",
        "Blue BC1": "Float64",
        "Blue BC2": "Float64",
        "Blue BCc": "Float64",
        "Green BC1": "Float64",
        "Green BC2": "Float64",
        "Green BCc": "Float64",
        "Red BC1": "Float64",
        "Red BC2": "Float64",
        "Red BCc": "Float64",
        "IR BC1": "Float64",
        "IR BC2": "Float64",
        "IR BCc": "Float64",
        "Readable status": "str",
    },
    header=0,
    pressure_variable="Pressure",
    rename_dict=(
        {
            "maeth_UV BC1_STP": "MA_Abs_Coeff_375",
            "maeth_Blue BC1_STP": "MA_Abs_Coeff_470",
            "maeth_Green BC1_STP": "MA_Abs_Coeff_528",
            "maeth_Red BC1_STP": "MA_Abs_Coeff_625",
            "maeth_IR BC1_STP": "MA_Abs_Coeff_880",
        } |
        {
            f"maeth_eBC_{wavelength}": f"MA_eBC_{wavelength}"
            for wavelength in [375, 470, 528, 625, 880]
        }
    )
)

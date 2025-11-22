import pandas as pd

from helikite.instruments import Instrument


class CO2(Instrument):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def file_identifier(self, first_lines_of_csv: list[str]) -> bool:
        columns = first_lines_of_csv[self.header].split(",")
        return "CO2" in columns

    def data_corrections(self, df, *args, **kwargs) -> pd.DataFrame:
        return df

    def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df["DateTime"] = pd.to_datetime(df["Time"], format="%y%m%d-%H%M%S")
        df.drop(columns=["Time"], inplace=True)

        df.set_index("DateTime", inplace=True)
        df.index = df.index.astype("datetime64[s]")

        return df

    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.filename,
            on_bad_lines="skip",
            low_memory=False,
            sep=",",
            usecols=self.dtype.keys(),
        )
        df = df.dropna(subset=["Time"])

        return df


co2 = CO2(
    name="co2",
    dtype={
        "Time": "str",
        "CO2": "Float64",
    },
    na_values=[],
    header=0,
    comment="#",
)

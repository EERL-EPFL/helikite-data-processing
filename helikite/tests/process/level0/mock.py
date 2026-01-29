import pandas as pd

from helikite.classes.output_schemas import OutputSchema
from helikite.instruments.base import Instrument


# Mock Instrument class for testing purposes
class MockInstrument(Instrument):
    def __init__(self, name, data=None, **kwargs):
        super().__init__(name, **kwargs)
        self.df_raw = pd.DataFrame(
            data if data else {"time": [], "pressure": []}
        )
        self.df = self.df_raw.copy()

    def __repr__(self):
        return "Mock"

    def read_from_folder(self, input_folder: str, quiet: bool = False, interactive: bool = True) -> pd.DataFrame:
        # Return df_raw directly for testing purposes
        return self.df_raw

    def file_identifier(self, first_lines_of_csv):
        # Assume mock instrument always identifies its file for simplicity
        return True

    def read_data(self):
        # Return the raw DataFrame for testing
        return self.df_raw

    def data_corrections(self, df, **kwargs):
        # Apply a mock correction, for instance adding a constant to pressure
        df["pressure"] += 10
        return df

    def set_time_as_index(self, df):
        # Set the "time" column as the index
        df.index = pd.to_datetime(df["time"])
        return df


def get_mock_output_schema(reference_instrument: Instrument, instruments: list[Instrument] | None = None):
    if instruments is None:
        instruments = [reference_instrument]

    return OutputSchema(
        campaign=None,
        instruments=instruments,
        reference_instrument_candidates=[reference_instrument],
        colors={reference_instrument.name: "C0"},
    )

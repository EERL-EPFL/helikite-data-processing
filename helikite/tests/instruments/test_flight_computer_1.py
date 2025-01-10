from helikite.instruments import flight_computer_v2
import pandas as pd
from helikite.constants import constants


def test_read_csv(campaign_file_paths_and_instruments_2024: str):
    """Test the reading of a CSV file"""

    # Test the reading of a CSV file
    df = flight_computer_v2.read_data()

    assert df is not None, "The DataFrame is None"
    assert (
        len(df) == 758 - 15 - 1
    ), "The number of rows in the DataFrame is not correct. "
    "(Total lines of CSV) - (occurrences of errors) - (header line)"


def test_set_time(campaign_file_paths_and_instruments_2024: str):
    """Test the setting of the time"""

    # Test the setting of the time
    df = flight_computer_v2.read_data()
    df = flight_computer_v2.set_time_as_index(df)

    assert df is not None, "The DataFrame is None"
    assert df.index.name == "DateTime", "The index name is not correct"

    # Assert first value 240926-141333 is 2024-09-26 14:13:33
    assert df.index[0] == pd.Timestamp(
        "2024-09-26 14:13:33"
    ), "The first value is not correct"

    # Assert last value 240926-163903 is 2024-09-26 16:39:03
    assert df.index[-1] == pd.Timestamp(
        "2024-09-26 16:39:03"
    ), "The last value is not correct"


def test_set_pressure_column(campaign_file_paths_and_instruments_2024: str):
    """Test the setting of the pressure column"""

    # Test the setting of the pressure column
    df = flight_computer_v2.read_data()
    df = flight_computer_v2.set_time_as_index(df)
    df = flight_computer_v2.set_housekeeping_pressure_offset_variable(df)
    assert df is not None, "The DataFrame is None"
    assert (
        constants.HOUSEKEEPING_VAR_PRESSURE in df.columns
    ), "The column 'Pressure' is not in the DataFrame"

    # First value 947.0
    assert (
        df[constants.HOUSEKEEPING_VAR_PRESSURE].iloc[0] == 947.0
    ), "The first value is not correct"

    # Last value 948.0
    assert (
        df[constants.HOUSEKEEPING_VAR_PRESSURE].iloc[-1] == 948.0
    ), "The last value is not correct"

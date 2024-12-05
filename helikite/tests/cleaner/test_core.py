from helikite.classes.cleaning import Cleaner
from helikite.tests.cleaner.mock import MockInstrument
import pandas as pd
import datetime


def test_set_pressure_column(campaign_data):
    instrument1 = MockInstrument(
        name="inst1",
        data={
            "time": pd.date_range("2023-01-01", periods=5, freq="min"),
            "pressure": [100, 101, 102, 103, 104],
        },
        dtype={"time": "datetime64[ns]", "pressure": "float64"},
        cols_export=["time", "pressure"],
        cols_housekeeping=["pressure"],
        pressure_variable="pressure",
    )

    # Use the temp directory as input_folder for Cleaner
    cleaner = Cleaner(
        instruments=[instrument1],
        reference_instrument=instrument1,
        input_folder=str(campaign_data),
        flight_date=datetime.date(2023, 1, 1),
    )

    cleaner.set_pressure_column(column_name_override="pressure")
    assert "pressure" in instrument1.df.columns


def test_set_time_as_index(campaign_data):
    instrument1 = MockInstrument(
        name="inst1",
        data={
            "time": pd.date_range("2023-01-01", periods=5, freq="min"),
            "pressure": [100, 101, 102, 103, 104],
        },
        dtype={"time": "datetime64[ns]", "pressure": "float64"},
        cols_export=["time", "pressure"],
        cols_housekeeping=["pressure"],
        pressure_variable="pressure",
    )

    cleaner = Cleaner(
        instruments=[instrument1],
        reference_instrument=instrument1,
        input_folder=str(campaign_data),
        flight_date=datetime.date(2023, 1, 1),
    )

    cleaner.set_time_as_index()
    assert instrument1.df.index.name == "time"


def test_data_corrections(campaign_data):
    instrument1 = MockInstrument(
        name="inst1",
        data={
            "time": pd.date_range("2023-01-01", periods=5, freq="min"),
            "pressure": [100, 101, 102, 103, 104],
        },
        dtype={"time": "datetime64[ns]", "pressure": "float64"},
        cols_export=["time", "pressure"],
        cols_housekeeping=["pressure"],
        pressure_variable="pressure",
    )

    cleaner = Cleaner(
        instruments=[instrument1],
        reference_instrument=instrument1,
        input_folder=str(campaign_data),
        flight_date=datetime.date(2023, 1, 1),
    )
    cleaner.set_time_as_index()
    cleaner.data_corrections()
    assert all(
        instrument1.df["pressure"] == [110, 111, 112, 113, 114]
    )  # Pressure increased by 10

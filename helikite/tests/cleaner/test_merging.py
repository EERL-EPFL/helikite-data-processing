import pytest
from helikite.classes.cleaning import Cleaner
from helikite.tests.cleaner.mock import MockInstrument
import pandas as pd
import datetime


def test_merge_instruments(campaign_data):
    instrument1 = MockInstrument(
        "inst1",
        data={
            "time": pd.date_range("2023-01-01", periods=3, freq="min"),
            "pressure": [100, 101, 102],
        },
        export_order=1,  # Ensure export_order is set
    )
    instrument2 = MockInstrument(
        "inst2",
        data={
            "time": pd.date_range("2023-01-01", periods=3, freq="min"),
            "pressure": [200, 201, 202],
        },
        export_order=2,  # Ensure export_order is set
    )

    cleaner = Cleaner(
        instruments=[instrument1, instrument2],
        reference_instrument=instrument1,
        input_folder=str(campaign_data),
        flight_date=datetime.date(2023, 1, 1),
    )

    cleaner.set_time_as_index()
    cleaner.correct_time()
    cleaner.data_corrections()
    cleaner.merge_instruments()

    # Assert that the merged DataFrame is correct
    assert len(cleaner.master_df) == 3

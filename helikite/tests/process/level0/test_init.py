import datetime

import pandas as pd

from helikite.classes.cleaning import Cleaner
from helikite.tests.process.level0.mock import MockInstrument, get_mock_output_schema


def test_cleaner_initialization(campaign_data):
    # Create mock instruments
    instrument1 = MockInstrument(
        "inst1",
        data={
            "time": pd.date_range("2023-01-01", periods=5, freq="min"),
            "pressure": [100, 101, 102, 103, 104],
        },
    )
    instrument2 = MockInstrument(
        "inst2",
        data={
            "time": pd.date_range("2023-01-01", periods=5, freq="min"),
            "pressure": [200, 201, 202, 203, 204],
        },
    )

    cleaner = Cleaner(
        output_schema=get_mock_output_schema(instrument1, [instrument1, instrument2]),
        instruments=[instrument1, instrument2],
        reference_instrument=instrument1,
        input_folder=str(campaign_data),
        flight_date=datetime.date(2023, 1, 1),
    )

    # Check attributes are set correctly
    assert len(cleaner._instruments) == 2
    assert cleaner.reference_instrument == instrument1
    assert cleaner.input_folder == str(campaign_data)

import os
from helikite.classes.cleaning import Cleaner
from helikite import instruments
import datetime


def test_antarctica_2025_02_12(campaign_data):
    """Test the antarctic campaign of 2025-02-12

    Second iteration of the flight computer cleaner
    """
    cleaner = Cleaner(
        instruments=[
            instruments.flight_computer_v2,
            instruments.smart_tether,
            instruments.msems_readings,
            instruments.msems_inverted,
            instruments.msems_scan,
        ],
        reference_instrument=instruments.flight_computer_v2,
        input_folder=os.path.join(campaign_data, "20250212"),
        flight_date=datetime.date(2025, 2, 12),
    )
    cleaner.set_time_as_index()
    cleaner.data_corrections()
    cleaner.set_pressure_column("pressure")
    cleaner.correct_time_and_pressure(max_lag=180)
    cleaner.merge_instruments()

    # Assert that the merged DataFrame is correct
    assert len(cleaner.master_df) == 8163

    # Todo: Add more assertions to validate the merged DataFrame

import os
from helikite.classes.cleaning import Cleaner
from helikite import instruments
import datetime


def test_2024_04_02(campaign_data):
    """Test the campaign of 2024-04-02"""

    cleaner = Cleaner(
        instruments=[
            instruments.flight_computer_v1,
            instruments.smart_tether,
            instruments.pops,
            instruments.msems_readings,
            instruments.msems_inverted,
            instruments.msems_scan,
            instruments.stap,
        ],
        reference_instrument=instruments.flight_computer_v1,
        input_folder=os.path.join(campaign_data, "20240402"),
        flight_date=datetime.date(2024, 4, 2),
        time_takeoff=datetime.datetime(2024, 4, 2, 10, 0, 35),
        time_landing=datetime.datetime(2024, 4, 2, 13, 4, 4),
        time_offset=datetime.time(0),
    )

    cleaner.set_time_as_index()
    cleaner.data_corrections()
    cleaner.set_pressure_column()
    cleaner.correct_time_and_pressure(max_lag=180)
    cleaner.remove_duplicates()
    cleaner.merge_instruments()
    cleaner.export_data()

    # Assert that the merged DataFrame is correct
    assert len(cleaner.master_df) == 10792


def test_2025_02_12(campaign_data):
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

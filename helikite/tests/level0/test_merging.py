import os
from helikite.classes.cleaning import Cleaner
from helikite import instruments
import datetime


def test_merge_instruments(campaign_data):

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
        time_landing=datetime.datetime(2024, 4, 2, 13, 4, 2),
        time_offset=datetime.time(0),
    )
    cleaner.set_time_as_index()
    cleaner.data_corrections()
    cleaner.set_pressure_column("pressure")
    cleaner.correct_time_and_pressure(
        max_lag=10,
        reference_pressure_thresholds=(900, 1200),
        walk_time_seconds=60,
    )
    cleaner.merge_instruments()

    # Assert that the merged DataFrame is correct
    assert len(cleaner.master_df) == 10792

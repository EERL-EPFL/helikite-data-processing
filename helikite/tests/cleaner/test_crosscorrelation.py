import os
from helikite.classes.cleaning import Cleaner
from helikite import instruments
import datetime


def test_crosscorrelation_offsets(campaign_data):

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
        offsets=[
            (instruments.pops, -3600),
            (instruments.stap, 7200),
        ],  # Shift POPS by 1 hour
    )
    # Validate the time shift is approximately 1 hour, allowing for cross-correlation shifts
    pops_shifted_time_diff = (
        cleaner.pops.df.index[0] - cleaner.pops.df_before_timeshift.index[0]
    ).total_seconds()
    staps_shifted_time_diff = (
        cleaner.stap.df.index[0] - cleaner.stap.df_before_timeshift.index[0]
    ).total_seconds()
    # Ensure the shift is within the expected range (e.g., 3600 ± 60 seconds)
    assert (
        3000 <= pops_shifted_time_diff <= 4000
    ), f"Time shift is out of expected range: {pops_shifted_time_diff} seconds"
    assert (
        -8000 <= staps_shifted_time_diff <= -7000
    ), f"Time shift is out of expected range: {staps_shifted_time_diff} seconds"

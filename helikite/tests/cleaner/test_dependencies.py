from helikite.classes.cleaning import Cleaner
from helikite.tests.cleaner.mock import MockInstrument
import datetime


def test_function_dependencies(campaign_data):
    instrument1 = MockInstrument("inst1")
    cleaner = Cleaner(
        instruments=[instrument1],
        reference_instrument=instrument1,
        input_folder=str(campaign_data),
        flight_date=datetime.date(2023, 1, 1),
    )

    cleaner.set_time_as_index()
    assert "set_time_as_index" in cleaner._completed_operations

    # Try to run set_time_as_index again (should not run due to use_once=True)
    cleaner.set_time_as_index()
    assert cleaner._completed_operations.count("set_time_as_index") == 1

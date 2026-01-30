from helikite.classes.cleaning import Cleaner
from helikite.tests.process.level0.mock import MockInstrument, get_mock_output_schema
import datetime


def test_function_dependencies(campaign_data):
    instrument1 = MockInstrument("inst1")
    cleaner = Cleaner(
        output_schema=get_mock_output_schema(instrument1),
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

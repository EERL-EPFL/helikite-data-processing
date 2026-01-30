import datetime

from helikite.classes.cleaning import Cleaner
from helikite.tests.process.level0.mock import MockInstrument, get_mock_output_schema


def test_cleaner_state(capfd, campaign_data):
    instrument1 = MockInstrument("inst1")
    cleaner = Cleaner(
        output_schema=get_mock_output_schema(instrument1),
        instruments=[instrument1],
        reference_instrument=instrument1,
        input_folder=str(campaign_data),
        flight_date=datetime.date(2023, 1, 1),
    )

    cleaner.state()
    captured = capfd.readouterr()
    assert "Instrument" in captured.out
    assert "inst1" in captured.out


def test_cleaner_help(capfd, campaign_data):
    instrument1 = MockInstrument("inst1")
    cleaner = Cleaner(
        get_mock_output_schema(instrument1),
        instruments=[instrument1],
        reference_instrument=instrument1,
        input_folder=str(campaign_data),
        flight_date=datetime.date(2023, 1, 1),
    )

    cleaner.help()
    captured = capfd.readouterr()
    assert (
        "There are several methods available to process the data:"
        in captured.out
    )

import pytest
import os
import datetime
from helikite.instruments import (  # noqa
    msems_scan,
    msems_readings,
    msems_inverted,
    smart_tether,
    flight_computer_v1,
    flight_computer_v2,
    pops,
    stap,
)


@pytest.fixture
def campaign_data_location_2022():
    return os.path.join(
        os.path.dirname(__file__), "resources", "campaigns", "20220929"
    )


@pytest.fixture
def campaign_data_location_2024_new_msesms():
    return os.path.join(
        os.path.dirname(__file__), "resources", "campaigns", "20241105"
    )


@pytest.fixture
def campaign_data_location_2024():
    return os.path.join(
        os.path.dirname(__file__), "resources", "campaigns", "20240926"
    )


@pytest.fixture
def fc_data_2022(campaign_data_location_2022: str):
    # Import flight computer data from the campaign data folder

    flight_computer_v1.filename = os.path.join(
        campaign_data_location_2022, "LOG_20220929.txt"
    )
    df = flight_computer_v1.read_data()

    return df


@pytest.fixture
def st_data_fake_midnight():
    # Set date that would usually happen in the config file/preprocess step
    smart_tether.date = datetime.datetime.strptime("9/29/2022", "%m/%d/%Y")
    smart_tether.filename = os.path.join(
        os.path.dirname(__file__),
        "resources",
        "smart_tether_during_midnight.csv",
    )

    df = smart_tether.read_data()

    return df


@pytest.fixture
def campaign_file_paths_and_instruments_2022(campaign_data_location_2022):
    instruments = {}

    # Assign filenames to instrument objects
    flight_computer_v1.filename = os.path.join(
        campaign_data_location_2022, "LOG_20220929.txt"
    )
    msems_inverted.filename = os.path.join(
        campaign_data_location_2022, "mSEMS_103_220929_101343_INVERTED.txt"
    )
    msems_readings.filename = os.path.join(
        campaign_data_location_2022, "mSEMS_103_220929_101343_READINGS.txt"
    )
    msems_scan.filename = os.path.join(
        campaign_data_location_2022, "mSEMS_103_220929_101343_SCAN.txt"
    )
    smart_tether.filename = os.path.join(
        campaign_data_location_2022, "LOG_20220929_A.csv"
    )
    pops.filename = os.path.join(
        campaign_data_location_2022, "HK_20220929x001.csv"
    )
    stap.filename = os.path.join(
        campaign_data_location_2022, "STAP_220929A0_processed.txt"
    )

    # Add instruments to dictionary
    instruments["flight_computer"] = flight_computer_v1
    instruments["msems_inverted"] = msems_inverted
    instruments["msems_readings"] = msems_readings
    instruments["msems_scan"] = msems_scan
    instruments["smart_tether"] = smart_tether
    instruments["pops"] = pops
    instruments["stap"] = stap

    return instruments


@pytest.fixture
def campaign_file_paths_and_instruments_2024(campaign_data_location_2024):
    instruments = {}

    # Assign filenames to instrument objects
    flight_computer_v2.filename = os.path.join(
        campaign_data_location_2024, "HFC_240926_3.csv"
    )

    # Add instruments to dictionary
    instruments["flight_computer"] = flight_computer_v2

    return instruments

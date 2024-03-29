import pytest
import os
import sys
import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from instruments import (  # noqa
    msems_scan, msems_readings, msems_inverted, smart_tether, flight_computer,
    pops, stap
)


@pytest.fixture
def campaign_data_location():

    return os.path.join(
        os.path.dirname(__file__), "resources", "campaigns", "20220929")


@pytest.fixture
def fc_data(campaign_data_location: str):
    # Import flight computer data from the campaign data folder

    flight_computer.filename = os.path.join(campaign_data_location,
                                            "LOG_20220929.txt")
    df = flight_computer.read_data()

    return df


@pytest.fixture
def st_data_fake_midnight():
    # Set date that would usually happen in the config file/preprocess step
    smart_tether.date = datetime.datetime.strptime("9/29/2022", "%m/%d/%Y")
    smart_tether.filename = os.path.join(
        os.path.dirname(__file__), "resources",
        "smart_tether_during_midnight.csv")

    df = smart_tether.read_data()

    return df


@pytest.fixture
def campaign_file_paths_and_instruments(campaign_data_location):
    instruments = {}

    # Assign filenames to instrument objects
    flight_computer.filename = os.path.join(
        campaign_data_location, 'LOG_20220929.txt')
    msems_inverted.filename = os.path.join(
        campaign_data_location, 'mSEMS_103_220929_101343_INVERTED.txt')
    msems_readings.filename = os.path.join(
        campaign_data_location, 'mSEMS_103_220929_101343_READINGS.txt')
    msems_scan.filename = os.path.join(
        campaign_data_location, 'mSEMS_103_220929_101343_SCAN.txt')
    smart_tether.filename = os.path.join(
        campaign_data_location, 'LOG_20220929_A.csv')
    pops.filename = os.path.join(
        campaign_data_location, 'HK_20220929x001.csv')
    stap.filename = os.path.join(
        campaign_data_location, 'STAP_220929A0_processed.txt')

    # Add instruments to dictionary
    instruments["flight_computer"] = flight_computer
    instruments['msems_inverted'] = msems_inverted
    instruments['msems_readings'] = msems_readings
    instruments['msems_scan'] = msems_scan
    instruments['smart_tether'] = smart_tether
    instruments['pops'] = pops
    instruments['stap'] = stap

    return instruments

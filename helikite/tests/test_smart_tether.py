import pandas as pd
import pytest
import os
import sys
from io import StringIO
import datetime

# Append the root directory of your project to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from instruments.smart_tether import SmartTether, smart_tether


def test_set_time_as_index(st_data):
    # call the function under test
    df = smart_tether.set_time_as_index(st_data)

    # check that the datetime column is set as the index
    assert df.index.name == "DateTime"

    # check that the date after midnight is set correctly
    assert df.iloc[0].name == pd.to_datetime("2022-09-29 23:59:58")
    assert df.iloc[1].name == pd.to_datetime("2022-09-29 23:59:59")
    assert df.iloc[2].name == pd.to_datetime("2022-09-30 00:00:01")
    assert df.iloc[3].name == pd.to_datetime("2022-09-30 00:00:02")
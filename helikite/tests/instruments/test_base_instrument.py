import datetime
import importlib
import inspect
import pathlib
import pkgutil
import re

import pandas as pd

import helikite.instruments
from helikite.classes.data_processing import OutputSchemas
from helikite.instruments import Instrument, cpc, smart_tether
from helikite.instruments.base import filter_columns_by_instrument


def test_read_data(campaign_file_paths_and_instruments_2022):
    for instrument in campaign_file_paths_and_instruments_2022.values():
        # Read filename (already defined in initiated class)
        df = instrument.read_data()

        assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame"
        assert df.empty is False, "No data found in file"


def test_instrument_instances():
    registered_classes = {obj.__class__ for obj in Instrument.REGISTRY.values()}
    # specify classes which are not supposed to be instantiated
    excluded_classes = {helikite.instruments.Instrument}
    all_classes = set()

    for _, name, _ in pkgutil.iter_modules([helikite.instruments.__path__[0]]):
        module = importlib.import_module(f"{helikite.instruments.__name__}.{name}")
        for _, cls in inspect.getmembers(module, predicate=inspect.isclass):
            if issubclass(cls, Instrument) and cls not in excluded_classes:
                all_classes.add(cls)

    uninstantiated_classes = all_classes - registered_classes
    assert not uninstantiated_classes, (
        f"The following instrument classes are not instantiated: {uninstantiated_classes}. "
        f"Create an instance of each class in its module file (e.g., see mcpc.py where "
        f"'mcpc' is an instance of the MCPC class)."
    )

def test_columns_consistency():
    for name, obj in Instrument.REGISTRY.items():
        cols_all = set(obj.dtype.keys())

        if obj == helikite.instruments.pops:
            cols_all.add("PartCon_186")

        if obj == helikite.instruments.flight_computer_v1:
            cols_all = cols_all.union(["Altitude", "Altitude_agl"])

        for attr in ["cols_export", "cols_housekeeping"]:
            cols = getattr(obj, attr)
            unexpected_cols = set(cols) - cols_all

            assert not unexpected_cols, (f"Columns from `{attr}` are not present in `dtype` of `{name}`: "
                                         f"{unexpected_cols}")

        if obj.expected_header_value:
            cols = re.split("[,;]", obj.expected_header_value.strip("\n"))
            cols = [col.strip() for col in cols]
            cols = filter(lambda x: len(x) > 0, cols)
            unexpected_cols = set(cols) - cols_all

            assert not unexpected_cols, (f"Columns from `expected_header_value` are not present in `dtype` of "
                                         f"`{name}`: {unexpected_cols}")


def test_expected_columns_level0_oracles(
    campaign_data_location_2025: str
):
    df = pd.read_csv(pathlib.Path(campaign_data_location_2025) / "level0_2025-02-14T16-16_header.csv")
    df_cpc = pd.read_csv(pathlib.Path(campaign_data_location_2025) / "level0_2025-02-14T18-32_header.csv")

    # in the old version of processing "cpc_DateTime" was the last CPC column
    cpc_columns = filter_columns_by_instrument(df_cpc.columns, cpc)
    cpc_columns = ["cpc_DateTime"] + cpc_columns[:-1]

    columns = df.columns.to_list() + cpc_columns

    date = datetime.date(2025, 2, 14)
    cpc.date = date
    smart_tether.date = date

    for instrument in OutputSchemas.ORACLES.instruments:
        # TODO: remove this once inconsistency with flight computer is resolved
        if instrument.name == "flight_computer":
            continue

        # TODO: remove once filter is integrated in the pipeline
        if instrument.name == "filter":
            continue

        expected_columns = filter_columns_by_instrument(columns, instrument)
        actual_columns = instrument.get_expected_columns(level=0, is_reference=False)

        assert set(expected_columns) == set(actual_columns)

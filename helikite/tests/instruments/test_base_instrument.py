import importlib
import inspect
import pkgutil
import re

import pandas as pd

import helikite.instruments
from helikite.instruments import Instrument


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

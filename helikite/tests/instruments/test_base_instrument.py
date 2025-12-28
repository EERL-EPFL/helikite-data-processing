import importlib
import inspect
import pathlib
import pkgutil
import re

import pandas as pd

import helikite.instruments
from helikite.classes.base import OutputSchemas
from helikite.classes.data_processing_level1 import DataProcessorLevel1
from helikite.instruments import Instrument, cpc, filter, mcpc, pops, flight_computer_v2
from helikite.instruments.base import filter_columns_by_instrument
from helikite.instruments.flight_computer import FlightComputer


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

    for instrument in OutputSchemas.ORACLES.instruments:
        expected_columns = filter_columns_by_instrument(columns, instrument)
        actual_columns = instrument.get_expected_columns_level0(is_reference=isinstance(instrument, FlightComputer))

        # TODO: remove once filter is integrated in the pipeline
        if instrument.name == "filter":
            continue

        assert set(expected_columns) == set(actual_columns)


def test_expected_columns_level0_turtmann(
    campaign_data_location_turtmann: str
):
    df = pd.read_csv(pathlib.Path(campaign_data_location_turtmann) / "level0_2024-02-20_B_header.csv")
    df_filter_mcpc_pops = pd.read_csv(pathlib.Path(campaign_data_location_turtmann) / "level0_2024-02-26_B_header.csv")

    # in the old version of processing "cpc_DateTime" was the last CPC column
    filter_columns = filter_columns_by_instrument(df_filter_mcpc_pops.columns, filter)
    mcpc_columns = filter_columns_by_instrument(df_filter_mcpc_pops.columns, mcpc)
    pops_columns = filter_columns_by_instrument(df_filter_mcpc_pops.columns, pops)

    columns = df.columns.to_list() + filter_columns + mcpc_columns + pops_columns

    for instrument in OutputSchemas.TURTMANN.instruments:
        expected_columns = filter_columns_by_instrument(columns, instrument)
        actual_columns = instrument.get_expected_columns_level0(is_reference=isinstance(instrument, FlightComputer))

        assert set(expected_columns) == set(actual_columns)


def test_expected_columns_level1_oracles(
    campaign_data_location_2025: str
):
    df = pd.read_csv(pathlib.Path(campaign_data_location_2025) / "level1_2025-02-14_D_header.csv", index_col="DateTime")

    expected_columns = df.columns.to_list()
    actual_columns = DataProcessorLevel1.get_expected_columns(OutputSchemas.ORACLES,
                                                              reference_instrument=flight_computer_v2,
                                                              with_dtype=False)

    # TODO: remove once filter is integrated in the pipeline
    filter_columns = filter_columns_by_instrument(actual_columns, filter)
    actual_columns = set(col for col in actual_columns if col not in filter_columns)

    assert set(expected_columns) == set(actual_columns)


# TODO: enable testing of Turtmann data
# def test_expected_columns_level1_turtmann(
#     campaign_data_location_turtmann: str
# ):
#     df = pd.read_csv(pathlib.Path(campaign_data_location_turtmann) / "level1_2024-02-20_B_header.csv", index_col="DateTime")
#
#     expected_columns = df.columns.to_list()
#     actual_columns = DataProcessorLevel1.get_expected_columns(OutputSchemas.TURTMANN,
#                                                               reference_instrument=flight_computer_v1,
#                                                               with_dtype=False)
#
#     # TODO: remove once filter is integrated in the pipeline
#     filter_columns = filter_columns_by_instrument(actual_columns, filter)
#     actual_columns = set(col for col in actual_columns if col not in filter_columns)
#
#     assert set(expected_columns) == set(actual_columns)

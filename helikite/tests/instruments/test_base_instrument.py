import inspect

import pandas as pd

import helikite.instruments


def test_read_data(campaign_file_paths_and_instruments_2022):
    for instrument in campaign_file_paths_and_instruments_2022.values():
        # Read filename (already defined in initiated class)
        df = instrument.read_data()

        assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame"
        assert df.empty is False, "No data found in file"


def test_columns_consistency():
    for name, obj in inspect.getmembers(helikite.instruments):
        if isinstance(obj, helikite.instruments.Instrument):
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
                cols = obj.expected_header_value.strip("\n").split(",")
                cols = [col.strip() for col in cols]
                cols = filter(lambda x: len(x) > 0, cols)
                unexpected_cols = set(cols) - cols_all

                assert not unexpected_cols, (f"Columns from `expected_header_value` are not present in `dtype` of "
                                             f"`{name}`: {unexpected_cols}")

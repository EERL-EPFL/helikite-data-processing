import pandas as pd


def test_read_data(campaign_file_paths_and_instruments_2022):
    for instrument in campaign_file_paths_and_instruments_2022.values():
        # Read filename (already defined in initiated class)
        df = instrument.read_data()

        assert isinstance(df, pd.DataFrame), "Data is not a pandas DataFrame"
        assert df.empty is False, "No data found in file"

from helikite.metadata.models import Flight
import pandas as pd
import pyarrow.parquet as pq
import json


def load_parquet(filepath: str) -> tuple[pd.DataFrame, Flight]:
    """
    Load a Parquet file, extract pandas DataFrame and metadata.
    """

    # Read Parquet file
    table = pq.read_table(filepath)
    df = table.to_pandas()

    # Extract metadata and decode keys and values
    metadata = {
        k.decode("utf8"): v.decode("utf8")
        for k, v in table.schema.metadata.items()
    }

    print(metadata)
    # Create Flight object
    flight = Flight(
        flight=metadata.get("flight"),
        flight_date=(
            pd.Timestamp(metadata.get("flight_date")).date()
            if metadata.get("flight_date")
            else None
        ),
        takeoff_time=(
            pd.Timestamp(metadata.get("takeoff_time"))
            if metadata.get("takeoff_time")
            else None
        ),
        landing_time=(
            pd.Timestamp(metadata.get("landing_time"))
            if metadata.get("landing_time")
            else None
        ),
        reference_instrument=metadata.get("reference_instrument"),
        instruments=(
            json.loads(metadata.get("instruments"))
            if metadata.get("instruments")
            else []
        ),
    )

    return df, flight

from helikite.metadata.models import Level0
import pandas as pd
import pyarrow.parquet as pq
import orjson


def load_parquet(filepath: str) -> tuple[pd.DataFrame, Level0]:
    """
    Load a Parquet file, extract pandas DataFrame and metadata.
    """

    # Read Parquet file
    table = pq.read_table(filepath)
    df = table.to_pandas()

    # Extract level0 metadata and decode keys and values
    metadata = orjson.loads(
        table.schema.metadata.get(b"level0", b"{}").decode("utf8")
    )

    # Unpack into level0 metadata object
    level0_md = Level0(
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
            metadata.get("instruments") if metadata.get("instruments") else []
        ),
    )

    return df, level0_md

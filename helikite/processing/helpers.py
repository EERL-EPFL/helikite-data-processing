import pandas as pd
from typing import Any
import logging
from helikite.constants import constants


# Define logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


def reduce_column_to_single_unique_value(df: pd.DataFrame, col: str) -> Any:
    """Reduce a column to a single value, if possible.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to reduce.
    col : str
        Column to reduce.

    Returns
    -------
    Any
        Single value if possible, else the original column.

    Raises
    ------
    ValueError
        If the column cannot be reduced to a single value.
    """

    # Get number of bins
    values = df.groupby(col).all().index.to_list()
    if len(values) == 1:
        return values[0]
    else:
        raise ValueError(
            f"Unable to reduce column '{col}' to a single value. "
            f"All values: {values}"
        )


def remove_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Removes duplicate rows from a dataframe based on the DateTime column"""

    df_unique = df.copy()
    df_unique["duplicate_values"] = df_unique.index.duplicated()
    df_unique.drop_duplicates(subset=["DateTime"], inplace=True)
    logger.info(
        "Initial length:",
        len(df),
        "After unique length:",
        len(df_unique),
        "Removed:",
        len(df) - len(df_unique),
    )

    return df_unique

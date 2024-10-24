import pandas as pd
from typing import Tuple
from helikite.constants import constants
import logging
from helikite.instruments.base import Instrument

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


def df_column_sort_key(
    export_df: Tuple[pd.DataFrame, Instrument]
) -> pd.DataFrame:
    """Sort key for ordering of exported dataframes

    Uses the export_order in the Instrument object to define sort 'height'
    ie. (df, 100) is sorted higher than (df, 50)

    Parameters
    ----------
    export_df : Tuple[pd.DataFrame, Instrument]
        Tuple of dataframe and Instrument object

    Returns
    -------
    int
        Sort order

    """

    df, instrument = export_df

    if instrument.export_order is None:
        # If no order set, push it to the end
        logger.info(
            f"There is no sort key for columns {export_df[0].columns}. "
            "Placing them at the end."
        )
        return 999999

    return instrument.export_order

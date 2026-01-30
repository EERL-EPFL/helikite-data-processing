from collections import defaultdict

import pandas as pd
import numpy as np

from helikite.instruments.base import filter_columns_by_instrument


def crosscorr(datax, datay, lag=10):
    """Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """

    return datax.corr(datay.shift(lag))


def df_derived_by_shift(df_init: pd.DataFrame, lags: np.ndarray, NON_DER=[]):
    df = df_init.copy()

    cols = defaultdict(list)
    for x in list(df.columns):
        if x not in NON_DER:
            for lag in lags:
                cols[x].append("{}_{}".format(x, lag))
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        for c in columns:
            lag = int(c.removeprefix(k + "_"))
            dfn[c] = df[k].shift(periods=lag)
        df = pd.concat([df, dfn], axis=1)  # , join_axes=[df.index])
    return df


def df_findtimelag(df, lags, instrument):
    filter_inst = filter_columns_by_instrument(df.columns, instrument)
    assert len(filter_inst) == len(lags), (f"Number of detected instrument columns ({len(filter_inst)})"
                                           f" differ from number of lags {len(lags)}.")
    df_inst = df[filter_inst].iloc[0]
    df_inst = df_inst.set_axis(lags, copy=False)

    return df_inst


def df_lagshift(
    df_instrument, df_reference, shift_quantity, instrument_name=""
):
    """
    Shifts the instrument's dataframe by the given quantity.
    First, match the instrument with the index of the reference instrument.
    """
    print(f"\tShifting {instrument_name} by {shift_quantity} index")

    df_shifted = df_instrument.copy()
    df_shifted.index = df_shifted.index.shift(periods=shift_quantity, freq="1s")
    df_shifted = df_shifted.reindex(index=df_reference.index)

    return df_instrument, df_shifted


# correct the other instrument pressure with the reference pressure
def matchpress(dfpressure, refpresFC, takeofftimeFL, walktime):

    diffpress = (
        dfpressure.loc[takeofftimeFL - walktime : takeofftimeFL].mean()
        - refpresFC
    )
    if not diffpress or not isinstance(diffpress, float):
        raise ValueError("Error in match pressure: diffpress is not a float")
    dfprescorr = dfpressure.sub(np.float64(diffpress))  # .iloc[0]

    return dfprescorr


def presdetrend(dfpressure, takeofftimeFL, landingtimeFL):
    """detrend instrument pressure measurements"""

    # Check for NA values and handle them
    start_pressure = dfpressure.loc[takeofftimeFL]
    end_pressure = dfpressure.loc[landingtimeFL]

    # TODO: How to handle NA. Should there even be NA in the pressure data?
    if pd.isna(start_pressure) or pd.isna(end_pressure):
        print(
            f"\tNA values found in pressure data between take off time of "
            f"{takeofftimeFL} and landing time of {landingtimeFL}. \n"
            "\tDropping NA values to calculate linear fit."
        )
        # Use the first and last non-NA values as fallback
        start_pressure = dfpressure.dropna().iloc[0]
        end_pressure = dfpressure.dropna().iloc[-1]

    linearfit = np.linspace(
        start_pressure,
        end_pressure,
        len(dfpressure),
    )

    dfdetrend = dfpressure - linearfit + start_pressure

    return dfdetrend

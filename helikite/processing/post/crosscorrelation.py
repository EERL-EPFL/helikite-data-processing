import pandas as pd
import numpy as np


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


def df_derived_by_shift(df_init, lag=0, NON_DER=[]):
    df = df_init.copy()
    if not lag:
        return df
    cols = {}
    for i in range(1, 2 * lag + 1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ["{}_{}".format(x, i)]
                else:
                    cols[x].append("{}_{}".format(x, i))
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)
        i = -lag
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i += 1
        df = pd.concat([df, dfn], axis=1)  # , join_axes=[df.index])
    return df


def df_findtimelag(df, range_list, instname=""):
    filter_inst = [col for col in df if col.startswith(instname)]
    df_inst = df[filter_inst].iloc[0]

    df_inst = df_inst.set_axis(range_list, copy=False)

    return df_inst


def df_lagshift(df_instrument, df_reference, index, instname=""):
    """
    Shift the instrument dataframe by the specified lag index and reindex
    to match the reference instrument's index.

    Parameters
    ----------
    df_instrument: pd.DataFrame
        The dataframe of the instrument to shift.
    df_reference: pd.DataFrame
        The dataframe of the reference instrument.
    index: int
        The lag index to shift by.
    instname: str
        The name of the instrument (for logging purposes).

    Returns
    -------
    df_shifted: pd.DataFrame
        The shifted instrument dataframe.
    """
    print(f"Shifting {instname} by {index} index")

    df_instrument = df_instrument.copy()

    # Shift the instrument data
    df_shifted = df_instrument.shift(periods=index, axis=0)

    # Remove duplicates in the shifted dataframe
    df_shifted = df_shifted[~df_shifted.index.duplicated(keep="first")]

    # Reindex the instrument data to match the reference instrument's index
    df_shifted = df_shifted.reindex(df_reference.index)

    return df_shifted


# correct the other instrument pressure with the reference pressure
def matchpress(dfpressure, refpresFC, takeofftimeFL, walktime):
    try:
        diffpress = (
            dfpressure.loc[takeofftimeFL - walktime : takeofftimeFL].mean()
            - refpresFC
        )
        dfprescorr = dfpressure.sub(np.float64(diffpress))  # .iloc[0]
    # catch when df1 is None
    except AttributeError:
        pass
    # catch when it hasn't even been defined
    except NameError:
        pass
    return dfprescorr


def presdetrend(dfpressure, takeofftimeFL, landingtimeFL):
    """detrend instrument pressure measurements"""
    print("take off location", dfpressure.loc[takeofftimeFL])
    print("landing location", dfpressure.loc[landingtimeFL])
    print("length of dfpressure", len(dfpressure))

    # Check for NA values and handle them
    start_pressure = dfpressure.loc[takeofftimeFL]
    end_pressure = dfpressure.loc[landingtimeFL]

    # TODO: How to handle NA. Should there even be NA in the pressure data?
    if pd.isna(start_pressure) or pd.isna(end_pressure):
        print(
            "Warning: NA values found in pressure data at takeoff or landing time."
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

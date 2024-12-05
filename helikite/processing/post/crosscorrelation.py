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
                if x not in cols:
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


def df_lagshift(
    df_instrument, df_reference, shift_quantity, instrument_name=""
):
    """
    Shifts the instrument's dataframe by the given quantity.
    First, match the instrument with the index of the reference instrument.
    """
    print(f"\tShifting {instrument_name} by {shift_quantity} index")

    # Add columns to the reference, so we know which to delete later
    # df_reference.columns = [f"{col}_ref" for col in df_reference.columns]
    df_original = df_instrument.copy()

    df_reference_index = df_reference.copy().index.to_frame()

    # Remove index name
    df_reference_index = df_reference_index.rename_axis(None, axis=1)

    df_shifted = df_original.shift(periods=shift_quantity, axis=0)
    # Get only the index of the reference and merge with instrument
    df_synchronised = pd.merge_asof(
        df_reference_index,
        df_shifted,
        left_index=True,
        right_index=True,
    )

    return (df_original, df_synchronised)


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

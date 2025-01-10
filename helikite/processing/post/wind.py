import numpy as np
import pandas as pd


def clip_data(unclipped, HIGH_CLIP, LOW_CLIP):
    """Clip unclipped between high_clip and low_clip.
    unclipped contains a single column of unclipped data."""

    # convert to np.array to access the np.where method
    np_unclipped = np.array(unclipped)
    # clip data above HIGH_CLIP or below LOW_CLIP
    cond_high_clip = (np_unclipped > HIGH_CLIP) | (np_unclipped < LOW_CLIP)
    np_clipped = np.where(cond_high_clip, np.nan, np_unclipped)
    return np_clipped.tolist()


def ewma_fb(df_column, span):
    """Apply forwards, backwards exponential weighted moving average (EWMA)
    to df_column.
    """

    # Forwards EWMA.
    fwd = pd.Series.ewm(df_column, span=span).mean()
    # Backwards EWMA.
    bwd = pd.Series.ewm(df_column[::-1], span=10).mean()
    # Add and take the mean of the forwards and backwards EWMA.
    stacked_ewma = np.vstack((fwd, bwd[::-1]))
    fb_ewma = np.mean(stacked_ewma, axis=0)
    return fb_ewma


def remove_outliers(spikey, fbewma, delta):
    """Remove data from df_spikey that is > delta from fbewma."""

    np_spikey = np.array(spikey)
    np_fbewma = np.array(fbewma)
    cond_delta = np.abs(np_spikey - np_fbewma) > delta
    np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
    return np_remove_outliers


def correction(indexval, df_level0val, HIGH_CLIP, LOW_CLIP, SPAN, DELTA):
    df_level0clipped = clip_data(df_level0val.tolist(), HIGH_CLIP, LOW_CLIP)
    df_level0clip = pd.DataFrame(indexval, df_level0clipped)
    df_level0clip = df_level0clip.set_index(indexval)
    df_level0ewma_fb = ewma_fb(df_level0clip["df_level0clipped"], SPAN)
    df_level0corr = remove_outliers(
        df_level0clipped.tolist(), df_level0ewma_fb.tolist(), DELTA
    )
    # dfhelikite['y_interpolated'] = df['y_remove_outliers'].interpolate()
    return df_level0corr

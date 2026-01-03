import datetime
import logging
import pathlib
import string
from dataclasses import dataclass
from typing import Literal, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helikite.constants import constants

CONC_COLUMN_NAME = "concentration"
GRAD_COLUMN_NAME = "gradient"
FLAG_COLUMN_NAME = "flag"

COLOR_RED = "#d73027"

# Define logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)

@dataclass
class FDAParameters:
    """
    Parameters for Flag Detection Algorithm.

    Parameters
    ----------
    inverse (bool): If True, detect low values/gradients instead of high
    avg_time (str | None): Resampling frequency (e.g., '1s', '5min') before processing
    main_filter (Literal["power_law", "iqr"]): Core detection method
    use_neighbor_filter (bool): If True, flag adjacent points of flagged by main filter data
    use_median_filter (bool): If True, flag points exceeding rolling median * factor
    use_sparse_filter (bool): If True, flag windows with high density of existing flags
    pl_a (float): Multiplier 'a' for power law curve (a * x^m), in case of main_filter='power_law'
    pl_m (float): Exponent 'm' for power law curve (a * x^m), in case of main_filter='power_law'
    iqr_window (str | None): Rolling window size for IQR calculation, in case of main_filter='iqr'
    iqr_factor (float | None): Multiplier for IQR threshold (Q3 + factor * IQR), in case of main_filter='iqr'
    lower_thr (float): The values below lower_thr are considered clean (polluted, if inverse=True) by main filter
    upper_thr (float): The values above upper_thr are considered polluted (clean, if inverse=True) by main filter
    median_window (str | None): Rolling window size for median filter
    median_factor (float | None): Multiplier for median filter threshold
    sparse_window (str | None): Rolling window size to check flag density
    sparse_thr (float | None): Min flagged points in window to trigger sparse filter
    """

    inverse: bool
    avg_time: str | None = None
    main_filter: Literal["power_law", "iqr"] = "power_law"
    use_neighbor_filter: bool = False
    use_median_filter: bool = False
    use_sparse_filter: bool = False
    pl_a: float = np.inf
    pl_m: float = 0
    iqr_window: str | None = None
    iqr_factor: float | None = None
    lower_thr: float = -np.inf
    upper_thr: float = np.inf
    median_window: str | None = None
    median_factor: float | None = None
    sparse_window: int | None = None
    sparse_thr: float | None = None


class FDA:
    """
    Flag Detection Algorithm (FDA) for identifying anomalies in time-series data.

    Based on the algorithm described in:
    "Automated identification of local contamination in remote atmospheric composition time series" by Ivo Beck et al. (2020)
    https://doi.org/10.5194/amt-15-4195-2022

    Parameters
    ----------
    df (pandas.DataFrame): Input dataframe containing concentration and optional flag data
    conc_column_name (str): Name of the column with the values to analyze
    flag_column_name (str | None): Name of the column with ground truth flags
    params (FDAParameters): Configuration object for the detection logic
    """
    def __init__(self, df: pd.DataFrame, conc_column_name: str, flag_column_name: str | None, params: FDAParameters):
        self._title = conc_column_name
        self._params = params
        self._df_orig = df.copy()
        self._conc_orig = conc_column_name

        columns = [conc_column_name]
        if flag_column_name is not None:
            columns.append(flag_column_name)

        self._df = df[columns].copy()
        self._df.rename(columns={conc_column_name: CONC_COLUMN_NAME, flag_column_name: FLAG_COLUMN_NAME}, inplace=True)

        if self._params.avg_time is not None:
            freq = pd.infer_freq(self._df.index)
            if freq is None or pd.to_timedelta(freq) <= pd.to_timedelta(self._params.avg_time):
                self._df = self._df.resample(self._params.avg_time).mean().dropna(how="all")
                if FLAG_COLUMN_NAME in self._df.columns:
                    self._df[FLAG_COLUMN_NAME] = self._df[FLAG_COLUMN_NAME] >= 0.5
            else:
                self._params.avg_time = None
                logger.warning(f"Dataframe has lower frequency ({freq}) than the specified averaging "
                               f"frequency ({self._params.avg_time}). No averaging will be performed.")

        if (~self._df[CONC_COLUMN_NAME].isna()).sum() < 2:
            self._df[GRAD_COLUMN_NAME] = pd.NA
        else:
            self._df[GRAD_COLUMN_NAME] = np.abs(np.gradient(self._df[CONC_COLUMN_NAME]))

        self._filters: list[Callable] | None = None
        self._intermediate_flags: list[pd.Series] | None = None

    def plot_data(self, use_time_index: bool = True, figsize=(18, 10), bins=100, fontsize=22, markersize=3,
                  save_path: str | pathlib.Path | None = None):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)

        data = self._df.copy()
        data[CONC_COLUMN_NAME] = data[CONC_COLUMN_NAME].where(data[CONC_COLUMN_NAME] > 0, np.nan)
        data[GRAD_COLUMN_NAME] = data[GRAD_COLUMN_NAME].where(data[GRAD_COLUMN_NAME] > 0, np.nan)

        # Plot histogram
        sns.histplot(data=data, x=CONC_COLUMN_NAME, y=GRAD_COLUMN_NAME,
                     bins=bins, ax=ax1,
                     cbar=True, cbar_kws={'label': 'Num. of observations'}, log_scale=True, pmax=.6)

        if FLAG_COLUMN_NAME in self._df.columns:
            sns.histplot(data=data, x=CONC_COLUMN_NAME, y=GRAD_COLUMN_NAME, weights=FLAG_COLUMN_NAME,
                         bins=bins, ax=ax1,
                         color="lightyellow", alpha=0.4)

        ax1.set_xlabel('Concentration', fontsize=fontsize)
        ax1.set_ylabel('Gradient', fontsize=fontsize)
        ax1.tick_params(axis='x', labelsize=fontsize)
        ax1.tick_params(axis='y', labelsize=fontsize)

        # Plot time series
        x = data.index if use_time_index else np.arange(len(data))

        ax2.plot(x, data[GRAD_COLUMN_NAME], '.', label='Gradient', color="black", linewidth=0.5,
                 markersize=markersize / 2, alpha=0.5)
        ax2.plot(x, data[CONC_COLUMN_NAME], '.', label='Original data', color=COLOR_RED, markersize=markersize)
        ax2.set_xlabel('Time', fontsize=fontsize)
        ax2.set_ylabel('Concentration', fontsize=fontsize)
        ax2.tick_params(axis='x', labelsize=fontsize / 2)
        ax2.tick_params(axis='y', labelsize=fontsize / 2)
        ax2.set_yscale('log')

        if np.isfinite(self._params.upper_thr) and not self._params.inverse:
            vmax = 1.1 * data[CONC_COLUMN_NAME].max()
            ax1.axvspan(self._params.upper_thr, vmax, facecolor=COLOR_RED, alpha=0.1)
            ax1.axvline(self._params.upper_thr, color='black', linestyle='--', linewidth=1,
                        label=f"threshold: {self._params.upper_thr}")

            ax2.axhspan(self._params.upper_thr, vmax, facecolor=COLOR_RED, alpha=0.1)
            ax2.axhline(self._params.upper_thr, color='black', linestyle='--', linewidth=1,
                        label=f"threshold: {self._params.upper_thr}")

        if np.isfinite(self._params.lower_thr) and self._params.inverse:
            vmin = 0.9 * data[CONC_COLUMN_NAME].min()
            ax1.axvspan(vmin, self._params.lower_thr, facecolor=COLOR_RED, alpha=0.1)
            ax1.axvline(self._params.lower_thr, color='black', linestyle='--', linewidth=1,
                        label=f"threshold: {self._params.lower_thr}")

            ax2.axhspan(vmin, self._params.lower_thr, facecolor=COLOR_RED, alpha=0.1)
            ax2.axhline(self._params.lower_thr, color='black', linestyle='--', linewidth=1,
                        label=f"threshold: {self._params.lower_thr}")

        if self._params.pl_a is not None and self._params.pl_m is not None and np.isfinite(self._params.pl_a):
            # Plot a line in log-loc scale: y = a * x**m
            a, m = self._params.pl_a, self._params.pl_m
            x = np.linspace(data[CONC_COLUMN_NAME].min(), data[CONC_COLUMN_NAME].max(), 100)
            y = a * x ** m
            ax1.plot(x, y, color='red', label='a=' + str(a) + ', m=' + str(m), lw=2)

            vmin, vmax = data[GRAD_COLUMN_NAME].min(), data[GRAD_COLUMN_NAME].max()
            if self._params.inverse:
                ax1.fill_between(x, y1=0.9 * vmin, y2=y, interpolate=True, color=COLOR_RED, alpha=0.1)
            else:
                ax1.fill_between(x, y1=y, y2=1.1 * vmax, interpolate=True, color=COLOR_RED, alpha=0.1)

            ax1.legend()

        ax2.legend()

        for n, ax in enumerate([ax1, ax2]):
            ax.text(0.0, 1.06, '(' + string.ascii_lowercase[n] + ')', transform=ax.transAxes, size=fontsize / 1.1,
                    weight='bold')

        fig.autofmt_xdate()
        fig.suptitle(self._title, fontsize=1.1 * fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def detect(self) -> pd.Series:
        self._filters = []
        match self._params.main_filter:
            case "iqr":
                self._filters.append(FDA.iqr_filter)
            case "power_law":
                self._filters.append(FDA.power_law_filter)
            case _:
                raise ValueError(f"Invalid main filter type: {self._params.main_filter}")

        if self._params.use_neighbor_filter:
            self._filters.append(FDA.neighbor_filter)

        if self._params.use_median_filter:
            self._filters.append(FDA.median_filter)

        if self._params.use_sparse_filter:
            self._filters.append(FDA.sparse_filter)

        self._intermediate_flags = []
        conc, grad = self._df[CONC_COLUMN_NAME], self._df[GRAD_COLUMN_NAME]

        flag = None
        for filter in self._filters:
            flag = filter(conc, grad, flag, self._params)
            self._intermediate_flags.append(flag)

        final_flag = pd.DataFrame(
            data={FLAG_COLUMN_NAME: pd.Series(self._intermediate_flags[-1], dtype="boolean")},
            index=self._df.index,
        )

        if self._params.avg_time is not None:
            final_flag = final_flag.reindex(self._df_orig.index, method="nearest")
            final_flag.where(~self._df_orig[self._conc_orig].isna(), pd.NA, inplace=True)

        return final_flag[FLAG_COLUMN_NAME]


    def plot_detection(self, use_time_index: bool = True, figsize=None, fontsize=14, markersize=3,
                       save_path: str | pathlib.Path | None = None, start_time: datetime.datetime | None = None,
                       end_time: datetime.datetime | None = None):
        sns.set_context(context="paper",
                        rc={"font.size": 14,
                            "axes.titlesize": 14,
                            "xtick.labelsize": 14,
                            "lines.linewidth": 3,
                            "lines.markersize": 6,
                            "ytick.labelsize": 14,
                            "legend.fontsize": 13})
        n = len(self._intermediate_flags)
        figsize = figsize if figsize is not None else (20, n * 4)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize, sharex=True)
        if n == 1: axes = [axes]

        for n, ax in enumerate(axes):
            ax.text(-0.1, 1.1, '(' + string.ascii_lowercase[n] + ')', transform=ax.transAxes, size=20, weight='bold')

        start_time = pd.to_datetime(start_time) if start_time is not None else 0
        end_time = pd.to_datetime(end_time) if end_time is not None else -1

        data = self._df[start_time:end_time]
        data = data[~data[CONC_COLUMN_NAME].isna()]
        conc = data[CONC_COLUMN_NAME]
        x = data.index if use_time_index else np.arange(len(data))

        for i, (flag, ax) in enumerate(zip(self._intermediate_flags, axes)):
            flag = flag[start_time:end_time]
            conc_flagged = conc.where(~flag, pd.NA)

            filter_label = self._filters[i].__name__.replace("_", " ")
            ax.plot(x, conc, '.', label='raw data', color=COLOR_RED, markersize=markersize)
            ax.plot(x, conc_flagged, '.', label=filter_label, color='#4575b4', markersize=markersize)

            ax.set_ylabel('Concentration', fontsize=fontsize)
            ax.set_xlabel('Time', fontsize=fontsize)
            ax.legend(loc=2)
            ax.tick_params(axis='y', labelsize=fontsize / 2)
            ax.tick_params(axis='x', labelsize=fontsize / 2)

            ax.set_yscale("log")

        fig.autofmt_xdate()
        fig.suptitle(self._title, fontsize=1.1 * fontsize)
        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, format='png', dpi=300)
        plt.show()

    @staticmethod
    def power_law_filter(conc: pd.Series, grad: pd.Series, flag_old: pd.Series, params: FDAParameters):
        power_law = params.pl_a * conc ** params.pl_m
        if not params.inverse:
            flag_new = ((grad > power_law) & (conc > params.lower_thr)) | (conc > params.upper_thr)
        else:
            flag_new = ((grad < power_law) & (conc < params.upper_thr)) | (conc < params.lower_thr)

        return flag_new

    @staticmethod
    def iqr_filter(conc: pd.Series, grad: pd.Series, flag_old: pd.Series, params: FDAParameters):
        iqr_window, iqr_factor = params.iqr_window, params.iqr_factor
        q075 = grad.rolling(iqr_window, center=True).quantile(0.75)
        q075 = q075.bfill().ffill()

        q025 = grad.rolling(iqr_window, center=True).quantile(0.25)
        q025 = q025.bfill().ffill()

        iqr = q075 - q025
        iqr_thr = q075 + (iqr_factor * iqr)

        if not params.inverse:
            flag_new = ((grad > iqr_thr) & (conc > params.lower_thr)) | (conc > params.upper_thr)
        else:
            flag_new = ((grad < iqr_thr) & (conc < params.upper_thr)) | (conc < params.lower_thr)

        return flag_new

    @staticmethod
    def neighbor_filter(conc: pd.Series, grad: pd.Series, flag_old: pd.Series, params: FDAParameters):
        flag_fillna = flag_old.fillna(False)

        shift_down = flag_fillna.shift(1)
        shift_up = flag_fillna.shift(-1)
        no_shift = flag_fillna.isna()

        flag_new = (shift_down | shift_up | no_shift).where(~flag_old.isna(), pd.NA)

        return flag_new

    @staticmethod
    def median_filter(conc: pd.Series, grad: pd.Series, flag_old: pd.Series, params: FDAParameters):
        median_window, median_factor = params.median_window, params.median_factor
        median = conc.rolling(median_window, center=True, min_periods=1).median()

        flag_new = flag_old | (conc > median_factor * median)

        return flag_new

    @staticmethod
    def sparse_filter(conc: pd.Series, grad: pd.Series, flag_old: pd.Series, params: FDAParameters):
        sparse_window, sparse_thr = params.sparse_window, params.sparse_thr

        bad_window = flag_old.astype("Int64").rolling(sparse_window, center=True,
                                                      min_periods=sparse_window).sum() >= sparse_thr
        # Propagate window violation to all points in the window
        bad_window = bad_window.rolling(sparse_window, center=True, min_periods=1).max().astype(bool).fillna(False)

        flag_new = flag_old | bad_window

        return flag_new

    @staticmethod
    def evaluate(conc: pd.Series, flag: pd.Series, flag_manual: pd.Series, verbose: bool = False):
        mask = ~conc.isna() & ~flag_manual.isna()
        conc, flag, flag_manual = conc[mask], flag[mask], flag_manual[mask]

        tp = (flag & flag_manual).sum()
        fp = (flag & ~flag_manual).sum()
        tn = (~flag & ~flag_manual).sum()
        fn = (~flag & flag_manual).sum()

        pr = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * pr * r / (pr + r)

        if verbose:
            print(f"TP = {tp}")
            print(f"FP = {fp}")
            print(f"TN = {tn}")
            print(f"FN = {fn}")

            print(f"Precision = {pr}")
            print(f"Recall = {r}")
            print(f"F1 = {f1}")

        return f1

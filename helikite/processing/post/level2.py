import pandas as pd
import numpy as np
import numpy.ma as ma
import os
import glob
import json
import seaborn as sns
from scipy.stats import linregress
import xarray as xr
import matplotlib.ticker as ticker
import math as math
from scipy.stats import gmean
from scipy import stats
from scipy.interpolate import splrep, splev, splint
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from functools import reduce
import itertools
from datetime import timedelta
import datetime as datetime
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcols
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines
from matplotlib.legend import Legend
from matplotlib.text import Text
import warnings
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binned_statistic

# Suppress specific warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Suppress all warnings
warnings.filterwarnings("ignore")

def select_hovering(df_level1_5):
    plt.close('all')

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ["#F54B0F", "#415067"]

    ax.plot(df_level1_5.index, df_level1_5["Altitude"], color=palette[0], linewidth=2)
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_ylim(0, 800)
    ax.set_ylabel("Altitude [m]")
    ax.set_xlabel("Time")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    selected_points = []
    stable_periods = []
    span_artists = []

    def onclick(event):
        if event.inaxes != ax:
            return

        # Skip if zoom or pan is active
        if plt.get_current_fig_manager().toolbar.mode != '':
            return

        click_time = mdates.num2date(event.xdata)
        selected_points.append(click_time)

        ax.plot(event.xdata, event.ydata, 'o', color=palette[1], markersize=8)
        fig.canvas.draw()

        if len(selected_points) == 2:
            start, end = sorted(selected_points)
            stable_periods.append((start, end))
            span = ax.axvspan(start, end, color=palette[1], alpha=0.2)
            span_artists.append(span)
            selected_points.clear()
            fig.canvas.draw()

    def finish_selection(event):
        nonlocal results_df
        if stable_periods:
            results_df = pd.DataFrame(stable_periods, columns=['Start_Time', 'End_Time'])
            results_df['Duration'] = results_df['End_Time'] - results_df['Start_Time']
            results_df['Start_Time'] = results_df['Start_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            results_df['End_Time'] = results_df['End_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

            print("\nSelected Stable Periods:")
            print(results_df.to_string(index=False))
            results_df.to_clipboard(index=False)
            print("\nResults copied to clipboard!")
        else:
            print("No stable periods selected")

    results_df = None

    # Add button
    ax_button = plt.axes([0.82, 0.01, 0.15, 0.05])
    btn = Button(ax_button, 'Finish Selection', color='lightgoldenrodyellow')
    btn.on_clicked(finish_selection)

    # Hook up click handler
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.title("Click to select stable periods")
    plt.show()

    return results_df

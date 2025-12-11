# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 17:15:09 2021
V1.0.0

@author: ivo
"""

# -*- coding: utf-8 -*-
"""
POLLUTION DETECTION ALGORITHM

ToDo: 
    - set more ticks into scatter plot to facilitate the reading of the line
    - time series plot with gradient scatter: Plot both, polluted and clean. 


Created on Wed Oct 13 2021

@author: Ivo Beck: ivo.beck@epfl.ch



"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.dates as mdates
import string

# Visualization Imports
import matplotlib.pyplot as plt
import seaborn as sns


def set_sns_context():
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("paper", rc={"font.size": 22,
                                 "axes.labelsize": 22,  # axis label size
                                 "xtick.labelsize": 22,
                                 "lines.linewidth": 3,
                                 "lines.markersize": 5,
                                 "ytick.labelsize": 22,
                                 "legend.fontsize": 22})


# set_sns_context()
#
# #%% Functions

def power_law_threshold_filter(concentration_series, gradient_series, a, m, upper_threshold, lower_threshold):
    '''
    Input two series, separate polluted and clean data with a separation line
    based on a and m: In log-log scale: y = a*x**m
    Additionally, data points with concentrations higher than upper_trheshold will be flagged as polluted as well.
    Concentrations lower than lower_threshold will be left untouched (assume they are clean)
    Parameters
    ----------
    concentration_series, gradient_series, a, m, upper_threshold, lower_threshold
    Returns
    -------
    threshold_clean: Series with only clean data points after application of both filters, 
    threshold_flag: series with the flags after application of these filters (1 = polluted, 0 = clean)
    
    '''

    # Calculations dataframe for all calculations used for the mask
    # pollution: Flag = 1, normalized >1
    powerlaw_df = pd.DataFrame()
    powerlaw_df['concentration'] = concentration_series.dropna()
    powerlaw_df['gradient'] = gradient_series.dropna()
    powerlaw_df['line'] = a*(powerlaw_df['concentration']**m) # Separation line, based on slope and intersect
    powerlaw_df['gradient_normalized'] = powerlaw_df['gradient']/powerlaw_df['line']
    
    # Flag as polluted (=1) if: gradient too high and concentration > than minimum threshold or concentration > upper threshold
    powerlaw_df['threshold_flag'] = np.where(((powerlaw_df['gradient_normalized'] > 1) & (powerlaw_df['concentration'] > lower_threshold)) | (powerlaw_df['concentration'] > upper_threshold), 1,0) #flag data based on gradient, threshold and keep low concntrations (lower treshold)
    powerlaw_df['threshold_clean'] = powerlaw_df['concentration'][powerlaw_df['threshold_flag']==0] # new column with only clean concentrations after these first two steps        
    
    return(powerlaw_df['threshold_clean'], powerlaw_df['threshold_flag'])



def iqr_filter(concentration_series, gradient_series, iqr_window, iqr_threshold, lower_threshold, upper_threshold):
    """
    Input two series, separate polluted and clean data based on the Interquartile range of the gradients.
    The interquartile range will be calculated for a window of size iqr_window. Datapoints with gradients
    exceeding the 75th percentile by more than the iqr_threshold will be flagged as polluted. 
    
    Additionally, data points with concentrations higher than upper_trheshold will be flagged as polluted as well.
    Concentrations lower than lower_threshold will be left untouched (assume they are clean)
    Parameters
    ----------
    Input: concentration_series, gradient_series, iqr_window, iqr_threshold, upper_threshold, lower_threshold
    Returns
    -------
    clean_threshold: Series with only clean data points after application of both filters, 
    flag_threshold: series with the flags after application of these filters (1 = polluted, 0 = clean)
    
    """
    
    df_iqr = pd.DataFrame()
    df_iqr['concentration'] = concentration_series
    df_iqr['gradient'] = np.abs(np.gradient(concentration_series))
    
    # iqr_window = 1440 # 1440, because 24h is equal to 1440 data points in window
    q075 = df_iqr['gradient'].rolling(iqr_window, center = True).quantile(0.75)
    
    
    q075 = q075.fillna(method='backfill')
    q075 = q075.fillna(method='pad')
    q025 = df_iqr['gradient'].rolling(iqr_window, center = True).quantile(0.25)
    q025 = q025.fillna(method='backfill')
    q025 = q025.fillna(method='pad')
    
    iqr = q075 - q025
    upper_range = q075 + (iqr_threshold * iqr)
    
    
    df_iqr['upper_range'] = upper_range
    df_iqr['iqr_outlier'] = np.where(df_iqr.loc[:,'gradient']>upper_range, 1,0)
    df_iqr['clean_IQR'] = df_iqr['concentration'][df_iqr['iqr_outlier']==0]

    df_iqr['flag_threshold'] =  np.where(((df_iqr['gradient'] > upper_range) & (df_iqr['concentration'] > int(lower_threshold))) | (df_iqr['concentration'] > int(upper_threshold)), 1,0) #flag data based on gradient, threshold and keep low concntrations (lower treshold)
    df_iqr['clean_threshold'] = df_iqr['concentration'][df_iqr['flag_threshold']==0] # new column with only clean concentrations after these first two steps        


    # q075 = gradient_series.rolling(iqr_window, center = True).quantile(0.75)
    # q075 = q075.fillna(method='backfill')
    # q075 = q075.fillna(method='pad')
    # q025 = gradient_series.rolling(iqr_window, center = True).quantile(0.25)
    # q025 = q025.fillna(method='backfill')
    # q025 = q025.fillna(method='pad')
    # print('iqr_window ', iqr_window)
    # print('iqr_trheshold ', iqr_threshold)
    # print('lower_threshold ', lower_threshold)
    # print('upper_threshold ', upper_threshold)
    # print(iqr)
    
    # iqr = q075 - q025
    # upper_range = q075 + (iqr_threshold * iqr)
    
    # thresholds = concentration_series > int(lower_threshold)


    # iqr_flag = np.where(((gradient_series > upper_range) & (concentration_series > int(lower_threshold))) | (concentration_series > int(upper_threshold)), 1,0) #flag data based on gradient, threshold and keep low concntrations (lower treshold)
    # iqr_clean = concentration_series[iqr_flag==0] # new column with only clean concentrations after these first two steps        
    # return(iqr_clean, iqr_flag)
    return(df_iqr['iqr_outlier'], df_iqr['clean_IQR'],df_iqr['flag_threshold'], df_iqr['clean_threshold'])


def neighbor_clean(series):
    """
    Parameters
    ----------
    series: series with pre-cleaned data, pollution should be NaN. 
        
    Returns
    -------
    a dataframe with two columns for the new concentration and the new flag
    """
    shift_down = series.isna().shift(1) #shift all nan(polluted) up and turn them to True
    shift_up = series.isna().shift(-1) #shift all nan (polluted) down and turn them to True
    shift_middle = series.isna() # nan = polluted = turn it into True
    shifted = (shift_down==True) | (shift_up==True) | (shift_middle == True) # shifted = all polluted
    series = series[shifted == False] # clean shift column , rest = nan
    return(series)

    # # return(df['neighbor_cleaned'])
    # print(len(series))
    # shift_down = series.isna().shift(1) #shift all nan(polluted) up and turn them to True
    # shift_up = series.isna().shift(-1) #shift all nan (polluted) down and turn them to True
    # shift_middle = series.isna() # nan = polluted = turn it into True
    # shifted = (shift_down==True) | (shift_up==True) | (shift_middle == True) # shifted = all polluted
    # series = series[shifted == False] # clean shift column , rest = nan
    # neighbor_flag = np.where(series.isna(), 1,0)
    # print(len(series), len(neighbor_flag))
    # return(series, neighbor_flag)
  

def median_filter_rolling(series_name, time, tolerance):
    """

    Parameters
    ----------
    series_name : series
        series of data points from the previous step
    time : integer
        Number of minutes for the centered running median.
    tolerance : float
        Factor. If data ponints exceed the median by this tolerance, they are flagged as polluted.

    Returns
    -------
    Medianfilter: Series
        Series of cleaned data after application of the median filter.
    flag_median: Series
        Series of flags after application of the median filter (1 = polluted, 0 = clean).

    """
    median_df = pd.DataFrame()
    median_df['input'] = series_name    

    median_df['median'] =  median_df['input'].rolling(time+'min', center = True).median()
    median_df['median'] = median_df['median'].fillna(method='pad')    
    
    #  using direct median only
    median_df['flag_median']  =  median_df['input'][ ( median_df['input'] > (median_df['median'] * tolerance) )] # | (df.median_filter_15min < (df.median_30min - df.median_30min * tolerance) )
    median_df['medianfilter'] =  median_df['input'][median_df['flag_median'].isnull()]  
    median_df['flag_median'] = np.where(median_df['medianfilter'].isnull(), 1,0)
  
    return median_df['medianfilter'],  median_df['flag_median']  

def sparse_data(series_name, window_size, min_number): 
    """
        Parameters
    ----------
    series_name : concentration series from previous step
    window_size: Integer
        Number of data points to consider as sparse window.
    min_number : Integer
        Maximum number of polluted data-points that are allowed within the window_size.
        If more points are polluted, all points in window_size will be flagged as polluted
        
    Returns
    -------
    sparse_filter: series of sparse filtered data  
    sparse_flag: series of sparse flag (1 = polluted data point, 0 = clean data point)

    """
    sparse_df = pd.DataFrame()
    sparse_df['input'] = series_name
    sparse_df['flag'] = np.where(sparse_df['input'].isna(), 1,0) #flag of the previous filter, 0 is clean, 1 is polluted
    # df['flagsum'] = df['flag'].rolling('30min', center = True).sum() # sum of clean points within 30min
    sparse_df['sparse_filter'] = sparse_df['input'][sparse_df['flag'].rolling(window_size, center = True).sum() < min_number] #if less flags than threshold (max number of polluted points) --> keep data point
    sparse_df['sparse_flag (1=polluted)'] = np.where(sparse_df['sparse_filter'].isna(), 1,0) # sparse flag: 1 if polluted, 0 if not
    return(sparse_df['sparse_filter'], sparse_df['sparse_flag (1=polluted)'])




    #Plot Gradient of Number concentration vs. Number concentration with straight line
    # scatter plot of mean gradient (dN/dt) vs Number concentration N 
def plot_grad_timeseries(data_df, raw_data, pathname_to_save, line=False, a=None, m=None):
    ###############################################################################
    # Plot one Time series and the gradient scatter in a 2*1 plot
    ###############################################################################
    ms = 3
    
    fig, axs = plt.subplots(figsize=(18,10),ncols = 2, constrained_layout = False)
      
    # im=axs[0].scatter(x, y, s=1)
    data_df['gradient'] = data_df['gradient'].replace(0, np.nan)
    data_df['concentration'] = data_df['concentration'].replace(0, 0.1)
    sns.histplot(data = data_df, x='concentration', y = 'gradient', bins = 200, cbar = True, cbar_kws={'label': 'Num. of observations'}, log_scale = True, pmax=.6, ax = axs[0]) #  cbar_kws=dict(shrink=.75), , pthresh=.05
    axs[0].set_xlabel('Concentration', fontsize=22)
    axs[0].tick_params(axis='x', labelsize = 22)
    axs[0].tick_params(axis='y', labelsize =22)
    axs[0].set_ylabel('Gradient', fontsize=22)
    # axs[0].set_yscale('log')
    # axs[0].set_xscale('log')
    # xticks = axs[0].get_xticks()
    
    
    # axs[0].locator_params(axis = 'both', nbins=7)
    # axs[0].xaxis.set_major_locator(MaxNLocator(7))     
    # axs[0].xaxis.set_major_locator(MultipleLocator(base = 1))
    # axs[0].set_xticks(np.logspace(0, np.log10(avg_data['concentration'].max()), num = 10))
    # axs[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # print(line)
    # axs[0].xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    # axs[0].xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    # axs[0].yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    # axs[0].yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)
       
    axs[1].plot(raw_data['concentration'],'.', label = 'Original data', color = '#d73027', markersize = ms )
    axs[1].set(xlabel='Time', ylabel='Concentration')
    axs[1].set_yscale('log')
    axs[1].xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    axs[1].xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    axs[1].yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    axs[1].yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)

    fig.autofmt_xdate()
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()
        
    for n,ax in enumerate(axs.flat):
        ax.text(0.0, 1.06, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, 
                size=20, weight='bold')
    if line is True:
        # Plot a line in log-loc scale: y = a*x**m
        x = np.linspace(10,50000,100)
        axs[0].plot(x,a*x**m, color='red', label = 'a='+str(a)+', m='+str(m), lw = 2)
        axs[0].text(0.5, 0.3, 'a = ' +str(a)+' m = '+ str(m))
        axs[1].plot(data_df['threshold_clean'],'.', label = 'Gradient, threshold filtered', color = '#4575b4', markersize = ms )
        axs[1].legend()
        plt.savefig(pathname_to_save / 'gradient_line_and_timeseries.png', format = 'png')

    else: 
        plt.savefig(pathname_to_save / 'gradient_and_timeseries.png', format = 'png')


def plot_power_law_scatter(data_df, a, m, pathname_to_save):
    """
    Scatter plot of gradient vs number concentrations, colored with wind direction data
    
    Parameters
    ----------
    data_df : dataframe: 
       contains concentration and gradient column
    a, m: floats
        intercept and slope of the separation line

    Returns
    -------
    Scatter plot of gradients vs concentrations, with separation line in log-log scale
    """
    
    
    fig, ax = plt.subplots(figsize=(18,10), constrained_layout = True)
    #plot lines in the graph
    sns.histplot(data = data_df, x='concentration', y = 'gradient', bins = 200, cbar = True, cbar_kws={'label': 'Num. of observations'}, log_scale = True, pmax=.6, ax = ax) #  cbar_kws=dict(shrink=.75), , pthresh=.05
    x = data_df['concentration']
    ax.plot(x,a*(x**m), color='red', label = 'a='+str(a)+', m='+str(m),lw = 2)

    ax.set_xlabel('Concentration', fontsize=22)
    ax.tick_params(axis='x', labelsize = 22)
    ax.tick_params(axis='y', labelsize =22)
    ax.set_ylabel('Gradient', fontsize=22)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    # ax.set_ylim(0.01,100000)
    # ax.set_xlim(0.5, 100000)
    fig.text(0.6, 0.25, 'a='+str(a)+', m='+str(m), style = 'italic', fontsize = 15, color = "red") 

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.savefig(pathname_to_save / 'gradient_and_line.png', format = 'png', dpi=300)
    plt.savefig(str(pathname_to_save) + '/' + 'gradient_and_line_a=' + str(a) + '_m='+str(m)+ '.png', format = 'png', dpi=300)



    
def final_four_plots(starttime, endtime, series1, series2, series3, series4, series5, pathname_to_save):
    sns.set_context("paper", rc={"font.size":14,
                                 "axes.titlesize":14,
                                 "xtick.labelsize":14,
                                 "lines.linewidth" : 3,
                                 "lines.markersize" : 6,   
                                 "ytick.labelsize":14,
                                 "legend.fontsize":13})
    # Show the different mask parameters
    start_time = pd.to_datetime(starttime)
    end_time = pd.to_datetime(endtime)    
    
    ds1= series1.loc[start_time:end_time]
    ds2= series2.loc[start_time:end_time]
    ds3= series3.loc[start_time:end_time]
    ds4= series4.loc[start_time:end_time]
    ds5= series5.loc[start_time:end_time]
    
    
   
    
    fig, axs = plt.subplots(4,1, figsize=(20,15)) # , figsize=(20,15)([ax1, ax2], [ax3, ax4])
    
    for n, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.1, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, 
                size=20, weight='bold')
        
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax4 = axs[3]
    
    
    
    ax1.plot(ds1, '.', label = 'Raw data',  color = '#d73027')
    ax1.plot(ds2, '.', label = 'gradient and threshold filtered', color = '#4575b4')
    # ax1.set_yscale('log')
    ax1.set_ylabel('Concentration', fontsize =14)
    ax1.legend(loc=2)
    ax1.tick_params(axis='y')
    # ax1.text(0.5, 0.5, 'A', size=10)
    
    # Edit the major and minor ticks of the x and y axes
    # top / right — whether there will be ticks on the secondary axes (top/right)
    ax1.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    ax1.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    ax1.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    ax1.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)
    
    
    ax2.plot(ds1, '.', label = 'Raw data',  color = '#d73027')
    ax2.plot(ds3, '.', label = 'neighbors filtered', color = '#4575b4')
    ax2.set_ylabel('Concentration', fontsize =14)
    # ax2.set_yscale('log')
    ax2.legend(loc=2)
    # ax2.text(0.97, 0.8, 'd)', size=10, weight='bold')
    
    # Edit the major and minor ticks of the x and y axes
    # top / right — whether there will be ticks on the secondary axes (top/right)
    ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    ax2.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    ax2.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)
    
    
    ax3.plot(ds1, '.', label = 'Raw data',  color = '#d73027') #d73027 (red)
    ax3.plot(ds4, '.', label = 'median filtered', color = '#4575b4') #4575b4 ((blue))
    # ax3.set_yscale('log')
    ax3.set_ylabel('Concentration', fontsize =14)
    ax3.set_xlabel(start_time.strftime('%Y-%m-%d'), fontsize = 14)
    ax3.legend(loc=2)
    ax3.tick_params(axis='y')
    # ax3.text(0.97, 0.8, 'B', size=10, weight='bold')
    
    # Edit the major and minor ticks of the x and y axes
    # top / right — whether there will be ticks on the secondary axes (top/right)
    ax3.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    ax3.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    ax3.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    ax3.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)
     
    
    ax4.plot(ds1, '.', label = 'Raw data',  color = '#d73027') #d73027 (red)
    ax4.plot(ds5, '.', label = 'All PDA filtered', color = '#4575b4') #4575b4 ((blue))ax4.set_xlabel('Concentration')
    ax4.set_ylabel('concentration', fontsize =14)
    ax4.legend(loc = 3)
    # ax4.text(0.97, 0.8, 'c', size=10, weight='bold')
    
    # Edit the major and minor ticks of the x and y axes
    # top / right — whether there will be ticks on the secondary axes (top/right)
    ax4.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    ax4.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    ax4.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    ax4.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)
    
    fig.autofmt_xdate()
    xfmt = mdates.DateFormatter('%Hh') # format the x axis strings
    ax1.xaxis.set_major_formatter(xfmt)
    ax2.xaxis.set_major_formatter(xfmt)
    ax3.xaxis.set_major_formatter(xfmt)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.savefig(pathname_to_save / 'PDA_timeseries_four_filters.png', format = 'png', dpi=300 )

def final_three_plots(starttime, endtime, series1, series2, series3, series4, pathname_to_save):
    
    sns.set_context("paper", rc={"font.size":14,
                                 "axes.titlesize":14,
                                 "xtick.labelsize":14,
                                 "lines.linewidth" : 3,
                                 "lines.markersize" : 6,   
                                 "ytick.labelsize":14,
                                 "legend.fontsize":13})
        
    # Show the different mask parameters
    start_time = pd.to_datetime(starttime)
    end_time = pd.to_datetime(endtime)
    
    
    ds1= series1.loc[start_time:end_time]
    ds2= series2.loc[start_time:end_time]
    ds3= series3.loc[start_time:end_time]
    ds4= series4.loc[start_time:end_time]
    
    
  
    fig, axs = plt.subplots(3,1, figsize=(20,15)) #([ax1, ax2], [ax3, ax4])
    
    for n, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.1, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, 
                size=20, weight='bold')
        
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

 
    
    ax1.plot(ds1, '.', label = 'Raw data',  color = '#d73027')
    ax1.plot(ds2, '.', label = 'gradient and threshold filtered', color = '#4575b4')
    ax1.set_yscale('log')
    ax1.set_ylabel('Concentration', fontsize =14)
    ax1.legend(loc=2)
    ax1.tick_params(axis='y')
    # ax1.text(0.5, 0.5, 'A', size=10)
    
    # Edit the major and minor ticks of the x and y axes
    # top / right — whether there will be ticks on the secondary axes (top/right)
    ax1.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    ax1.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    ax1.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    ax1.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)
    
    
    ax2.plot(ds1, '.', label = 'Raw data',  color = '#d73027')
    ax2.plot(ds3, '.', label = 'neighbors filtered', color = '#4575b4')
    ax2.set_ylabel('Concentration', fontsize =14)
    # ax2.set_yscale('log')
    ax2.legend(loc=2)
    # ax2.text(0.97, 0.8, 'd)', size=10, weight='bold')
    
    # Edit the major and minor ticks of the x and y axes
    # top / right — whether there will be ticks on the secondary axes (top/right)
    ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    ax2.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    ax2.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)
    
    
    ax3.plot(ds1, '.', label = 'Raw data',  color = '#d73027') #d73027 (red)
    ax3.plot(ds4, '.', label = 'All PDA filtered', color = '#4575b4') #4575b4 ((blue))
    # ax3.set_yscale('log')
    ax3.set_ylabel('Concentration', fontsize =14)
    ax3.set_xlabel(start_time.strftime('%Y-%m-%d'), fontsize = 14)
    ax3.legend(loc=2)
    ax3.tick_params(axis='y')
    # ax3.text(0.97, 0.8, 'B', size=10, weight='bold')
    
    # Edit the major and minor ticks of the x and y axes
    # top / right — whether there will be ticks on the secondary axes (top/right)
    ax3.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    ax3.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    ax3.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    ax3.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)
    

    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.savefig(pathname_to_save / 'PDA_timeseries_3filters.png', format = 'png', dpi=300 )


def final_two_plots(starttime, endtime, series1, series2, series3, pathname_to_save):
    # Show the different mask parameters
    start_time = pd.to_datetime(starttime)
    end_time = pd.to_datetime(endtime)
    
    
    ds1= series1.loc[start_time:end_time]
    ds2= series2.loc[start_time:end_time]
    ds3= series3.loc[start_time:end_time]
    
    
    sns.set_context("paper", rc={"font.size":14,
                                 "axes.titlesize":14,
                                 "xtick.labelsize":14,
                                 "lines.linewidth" : 3,
                                 "lines.markersize" : 6,   
                                 "ytick.labelsize":14,
                                 "legend.fontsize":13})
    
    
    fig, axs = plt.subplots(2,1, figsize=(20,15)) # ([ax1, ax2], [ax3, ax4])
    
    for n, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.1, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, 
                size=20, weight='bold')
        
    ax1 = axs[0]
    ax2 = axs[1]

 
    
    ax1.plot(ds1, '.', label = 'Raw data',  color = '#d73027')
    ax1.plot(ds2, '.', label = 'gradient and threshold filtered', color = '#4575b4')
    # ax1.set_yscale('log')
    ax1.set_ylabel('Concentration', fontsize =14)
    ax1.legend(loc=2)
    ax1.tick_params(axis='y')
    # ax1.text(0.5, 0.5, 'A', size=10)
    
    # Edit the major and minor ticks of the x and y axes
    # top / right — whether there will be ticks on the secondary axes (top/right)
    ax1.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    ax1.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    ax1.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    ax1.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)
    
    
    ax2.plot(ds1, '.', label = 'Raw data',  color = '#d73027')
    ax2.plot(ds3, '.', label = 'All PDA filtered', color = '#4575b4')
    ax2.set_ylabel('Concentration', fontsize =14)
    # ax2.set_yscale('log')
    ax2.legend(loc=2)
    # ax2.text(0.97, 0.8, 'd)', size=10, weight='bold')
    ax2.set_xlabel(start_time.strftime('%Y-%m-%d'), fontsize = 14)

    
    # Edit the major and minor ticks of the x and y axes
    # top / right — whether there will be ticks on the secondary axes (top/right)
    ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom= 'on') # which — whether we are editing major , minor , or both ticks
    ax2.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', bottom= 'on') # size — length of ticks in points
    ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left= 'on') #width — line width of ticks (we can set this to the same as our axis line width)
    ax2.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', left= 'on') # direction — whether ticks will face in , out , or inout (both)

    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
  
    plt.savefig(pathname_to_save / 'PDA_timeseries_two_filters.png', format = 'png', dpi=300 )



def averaging(raw_data, avg_time):
    averages = pd.DataFrame()
    averages['concentration'] = raw_data['concentration'].resample(str(avg_time)+'s').mean()
    averages['gradient'] = raw_data['gradient'].resample(str(avg_time)+'s').mean()              
    
    return(averages)
        
def pathname():       
    pathname_to_file = input("Enter the path to your datafile: ")
    pathname_to_save = input("Enter the path to your target directory to save plots and output data: ")
    filename = input("Enter name of your datafile (incl. ending): ")
    
    path_to_file = Path(pathname_to_file)
    datafile = path_to_file/filename
    output_path = Path(pathname_to_save)
    return(datafile, output_path)

def load_data(datafile):
    raw_data = pd.read_csv(datafile,sep = ',', header = 0,names = ['date', 'concentration'], parse_dates = ['date'], infer_datetime_format=(True), index_col = ['date'] )#, names = ['date', 'concentration'], parse_dates = ['date'], index_col = ['date']
    raw_data = raw_data.dropna()
    raw_data['gradient'] = np.abs(np.gradient(raw_data['concentration']))
    return(raw_data)
    


# #%%
# MAIN FUNCTION
# -------------------------
# LOAD DATASETS
# ------------------
#
# pathname_to_file = r'C:\Users\ivo\Documents\08_MOSAiC Pollution Mask\Files\Test_PDA_folder'
# pathname_to_save = r'C:\Users\ivo\Documents\08_MOSAiC Pollution Mask\Files\Test_PDA_folder'
# filename = 'cpc3025_feb_mar_cleaned_raw_test.txt'
# averaging_time = '60'
# gradient_filter_method = 'a'
# upper_threshold = 10000
# lower_threshold = 60
# neighbor_decision = 'y'
# median_time = '60'
# median_factor = 1.4
# iqr_window = 1440
#
# #%% Start the Script with questions
# ############################################
# a = None
# m = None
#
# # Ask for pathnames and filenames
# file_path, pathname_to_save = pathname()
# pathname_to_save = Path(pathname_to_save)
# file_path = Path(file_path)
#
# # Load data
# print('Load dataset...')
# raw_data = load_data(file_path)
# print('Dataset path: ' + str(file_path))
# print('Done loading!')
#
# #%%
# #################################################################
# # Start here to leave the filepath and filename as it was before
# #############################################################
# # Averaging data
# avg_data = pd.DataFrame()
# while True:
#     averaging_time = input("Do you wish to average your data? If NO, type ENTER. If YES, type the averaging time in seconds for the gradient plot: ")
#     if averaging_time == '':
#         print('Proceed without averaging...')
#         avg_data['concentration'] = raw_data['concentration']
#         avg_data['gradient'] = raw_data['gradient']
#         break
#     else:
#         try:
#             avg_time = int(averaging_time)
#             avg_data = averaging(raw_data, avg_time)  # run the averaging function
#             break
#         except ValueError:
#             print('Please press ENTER if you wish to proceed without averaging or type in an averaging time (integer number)!')
#             continue
#
# # Create gradient and time series figure
# print('A new figure with the gradient vs concentration and a time series of your data was saved in your target directory. Please open it to make a decision for the next step.')
# plot_grad_timeseries(data_df=avg_data, raw_data=raw_data, pathname_to_save=pathname_to_save, line=False, a=a, m=m)
#
# # Step 1
# print('Step1: Check out the gradient vs concentration figure and decide whether you wish to separate polluted from clean data based on a power law (straight line in the logarithmnic gradient plot), or based on the interquartile range (IQR).')
# line_decision = ''
# while True:
#     if line_decision == 'n':
#         break
#     gradient_filter_method = input("Step 1: Continue with power law filter (a) or interquartile range filter (b)? (a/b) ")
#
#     # Ask for which step1 to choose
#     if gradient_filter_method == 'a':
#         print('Your choice: Continue with power law filter')
#         while True:
#             try:
#                 x1, y1, x2, y2 = [float(s) for s in input('Step 1A (power law filter): Type two fix points (x1, y1) and (x2, y2) from the graph to find a separation line. The points dont have to be very accurate and the separtation line can be adjusted later. Type x1 y1 x2 y2 ').split()]
#             except ValueError:
#                 print('Please enter FOUR float values!')
#                 continue
#             else:
#                  break # without break it would jump back to the while True loop
#         m = np.abs(np.log10(y2/y1)/np.log10(x2/x1))
#         a = y1/(x1**m)
#
#         while True:
#             if line_decision =='n':
#                 break
#             plot_power_law_scatter(data_df=avg_data, a=a, m=m, pathname_to_save=pathname_to_save)
#             print('PDA step 1A: Power law separation line slope m = '+str(m)+', intercept at a = '+str(a)+'. A new figure (gradient_and_line.png) with a potential separation line was saved to your target directory.',
#                       'Please open it and decide whether you want to correct the line.')
#             try:
#                 while True:
#                     if line_decision == 'n':
#                         break
#
#                     line_decision = input('Step 1A: Do you wish to enter a new slope and a new intercept for the separation line (y/n)?')
#                     if line_decision =='n':
#                         break
#                     elif line_decision == 'y':
#                         while True:
#                             try:
#                                 a, m = [float(s) for s in input('Step 1A. Actual intercept a = '+str(a)+', and slope m = '+str(m)+'. Type in the new values (a m):').split()]
#                                 plot_power_law_scatter(data_df=avg_data, a=a, m=m, pathname_to_save=pathname_to_save)
#                                 print('New figure with separation line saved to target directory. ')
#
#                             except ValueError:
#                                 print('Please enter two float numbers, separated by a space')
#
#                             else:
#                                 break
#                     else:
#                         print('Please type y or n')
#
#             except ValueError:
#                 print('Please enter y or n')
#
#             else:
#                 continue
#
#     elif gradient_filter_method == 'b':
#         while True:
#             try:
#                 iqr_timewindow, iqr_factor = [float(s) for s in input("Step 1B (IQR filter): Choose the iqr window size (in minutes) and the iqr factor: ").split()]
#             except ValueError:
#                 print('Please enter TWO values!')
#             else:
#                 break
#         break
#     else:
#         print('Please choose a method for step 1')
#
#
#
# # Step 2. Threshold filter
# # -------------------------
# while True:
#     try:
#         upper_threshold, lower_threshold = [int(s) for s in input("Step 2 (threshold filter): Choose uppper threshold and lower threshold (upper_threshold lower_threshold): ").split()]
#     except ValueError:
#         print('Please enter TWO values!')
#     else:
#         break
#
#
# # Run Gradient and threshold filter
# if gradient_filter_method == 'a':
#     avg_data['threshold_clean'], avg_data['threshold_flag (1=polluted)'] = power_law_threshold_filter(avg_data['concentration'], avg_data['gradient'], a=a, m=m, upper_threshold=upper_threshold, lower_threshold = lower_threshold)
#     print('A plot of the gradient vs concentration of your dataset with the separarion line in your target folder')
#     # Todo: Change the plot so that the separated data can bee seen in time series.
#     plot_grad_timeseries(data_df = avg_data, line = True, a=a, m=m)
# elif gradient_filter_method == 'b':
#     # avg_data['threshold_clean'], avg_data['threshold_flag (1=polluted)'] = iqr_filter(avg_data['concentration'], avg_data['gradient'], iqr_window = int(iqr_timewindow), iqr_threshold=iqr_factor, lower_threshold = lower_threshold, upper_threshold = upper_threshold)
#     avg_data['iqr_outlier'], avg_data['clean_IQR'],  avg_data['threshlold_flag (1=polluted)'],  avg_data['threshold_clean'] = iqr_filter(avg_data['concentration'], avg_data['gradient'], iqr_window = int(iqr_timewindow), iqr_threshold=iqr_factor, lower_threshold = lower_threshold, upper_threshold = upper_threshold)
#
#
# # Step3. Neighbor Filter
# # ------------------------------------
# neighbor_decision = input("Step 3: Do you want to apply the neighboring points filter? Type y for yes, anything else for no: ")
# if neighbor_decision == 'y':
#     print('Apply step 3: Neighboring points filter')
#     avg_data['neighbor_clean'] = neighbor_clean(avg_data['threshold_clean']) #, 'neighbor_flag'
#
# median_decision = input("Step 4: Do you want to apply the median filter? Type y for yes, anything else for no: ")
# if median_decision == 'y':
#     while True:
#         try:
#             median_time, median_factor = input("Step 4: Enter median time (minutes) and median factor (m_t m_f): ").split()
#             median_factor = float(median_factor)
#         except ValueError:
#             print('Please enter TWO variables!')
#         else:
#             break
#
#     if neighbor_decision == 'y':
#         print('Step 4: Apply Median filter, based on neighbor filter')
#         avg_data['median_clean'], avg_data['median_flag (1=polluted)'] = median_filter_rolling(avg_data['neighbor_clean'], time = median_time, tolerance = median_factor)
#     elif neighbor_decision != 'y':
#         print('Step 4: Apply Median filter, based on treshold filter')
#         avg_data['median_clean'], avg_data['median_flag (1=polluted)'] = median_filter_rolling(avg_data['threshold_clean'], time = median_time, tolerance = median_factor)
#
# while True:
#     try:
#         sparse_window, sparse_threshold = [int(s) for s in input('Step 5: Choose a sparse window size (# of datapoints) and threshold (max. allowed number of polluted data points within sparse window) (sparse_window sparse_threshold): ').split()]
#     except ValueError:
#         print('Please enter TWO variables!')
#     else:
#         break
# if neighbor_decision == 'y' and median_decision == 'y':
#     print('Step 5: Apply sparse data filter, based on median filter')
#     avg_data['sparse_clean'], avg_data['sparse_flag'] = sparse_data(avg_data['median_clean'], window_size = int(sparse_window), min_number = sparse_threshold)
#     starttime, endtime = input('PDA done. To check out the result, type in a start time and an end time of a plot (YYYY-MM-DD YYYY-MM-DD): ').split()
#     final_four_plots(starttime, endtime, series1=avg_data['concentration'], series2=avg_data['threshold_clean'], series3=avg_data['neighbor_clean'], series4=avg_data['median_clean'], series5=avg_data['sparse_clean'], pathname_to_save=pathname_to_save)
#     print('A figure with four time series of different filtering steps was saved to your target directory.')
#
# elif neighbor_decision == 'y' and median_decision != 'y':
#     print('Step 5: Apply sparse data filter, based on neighbor filter')
#     avg_data['sparse_clean'], avg_data['sparse_flag'] = sparse_data(avg_data['neighbor_clean'], window_size = int(sparse_window), min_number = sparse_threshold)
#     starttime, endtime = input('PDA done. To check out the result, type in a start time and an end time of a plot (YYYY-MM-DD YYYY-MM-DD): ').split()
#     final_three_plots(starttime, endtime, series1=avg_data['concentration'], series2=avg_data['threshold_clean'], series3=avg_data['neighbor_clean'], series4=avg_data['sparse_clean'], pathname_to_save=pathname_to_save)
#     print('A figure with three time series of different filtering steps was saved to your target directory.')
#
# elif neighbor_decision != 'y' and median_decision == 'y':
#     print('Step 5: Apply sparse data filter, based on median filter witout neighbor filter')
#     avg_data['sparse_clean'], avg_data['sparse_flag'] = sparse_data(avg_data['median_clean'], window_size = int(sparse_window), min_number = sparse_threshold)
#     starttime, endtime = input('PDA done. To check out the result, type in a start time and an end time of a plot (YYYY-MM-DD YYYY-MM-DD): ').split()
#     final_three_plots(starttime, endtime, series1=avg_data['concentration'], series2=avg_data['threshold_clean'], series3=avg_data['median_clean'], series4=avg_data['sparse_clean'], pathname_to_save=pathname_to_save)
#     print('A figure with three time series of different filtering steps was saved to your target directory.')
#
# elif neighbor_decision != 'y' and median_decision != 'y':
#     print('Step 5: Apply sparse data filter, based on threshold filter')
#     avg_data['sparse_clean'], avg_data['sparse_flag'] = sparse_data(avg_data['threshold_clean'], window_size = int(sparse_window), min_number = sparse_threshold)
#     starttime, endtime = input('PDA done. To check out the result, type in a start time and an end time of a plot (YYYY-MM-DD YYYY-MM-DD): ').split()
#     final_two_plots(starttime, endtime, series1=avg_data['concentration'], series2=avg_data['threshold_clean'], series3=avg_data['sparse_clean'], pathname_to_save=pathname_to_save)
#     print('A figure with two time series of different filtering steps was saved to your target directory.')
#
#
# # Save data to your directory
# avg_data.to_csv(pathname_to_save / 'PDA_pollution_flag.csv', index = True, header = True, index_label = 'Date/Time')
# print('Your pollution flagged dataset is saved in your target directory.')
#
# data_stats = avg_data.describe()
# data_stats.to_excel(pathname_to_save / 'PDA_steps_statistics.xlsx')


"""
CPC3007
Total particle concentration in size range of 7 - 2000 nm.
"""

from helikite.instruments.base import Instrument
from helikite.constants import constants
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcols
import matplotlib.dates as mdates

class CPC(Instrument):
    """
    Instrument definition for the cpc3007 sensor system.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "cpc"
        
    
    def CPC_STP_normalization(df):
        """
        Normalize CPC3007 concentrations to STP conditions and insert the results
        right after the existing CPC columns.
    
        Parameters:
        df (pd.DataFrame): DataFrame containing CPC measurements and metadata.
    
        Returns:
        df (pd.DataFrame): Updated DataFrame with STP-normalized columns inserted.
        """
        plt.close('all')
    
        # Constants for STP
        P_STP = 1013.25  # hPa
        T_STP = 273.15   # Kelvin
    
        # Measured conditions
        P_measured = df["flight_computer_pressure"]
        T_measured = df["Average_Temperature"] + 273.15  # Convert °C to Kelvin
    
        # Calculate STP correction
        correction_factor = (P_measured / P_STP) * (T_STP / T_measured)
        normalized_column = df['cpc_totalconc_raw'] * correction_factor
    
        # Prepare to insert
        cpc_columns = [col for col in df.columns if col.startswith('cpc_')]
        if cpc_columns:
            last_cpc_index = df.columns.get_loc(cpc_columns[-1]) + 1
        else:
            last_cpc_index = len(df.columns)
    
        # Insert STP-normalized column (only if it doesn't already exist)
        if 'cpc_totalconc_stp' in df.columns:
            df = df.drop(columns='cpc_totalconc_stp')
    
        df = pd.concat(
            [df.iloc[:, :last_cpc_index],
             pd.DataFrame({'cpc_totalconc_stp': normalized_column}, index=df.index),
             df.iloc[:, last_cpc_index:]],
            axis=1
        )
        
        # PLOT
        plt.figure(figsize=(8, 6))
        plt.plot(df['cpc_totalconc_raw'], df['Altitude'], label='Measured', color='blue', marker='.', linestyle='none')
        plt.plot(df['cpc_totalconc_stp'], df['Altitude'], label='STP-normalized', color='red', marker='.', linestyle='none')
        plt.xlabel('CPC3007 total concentration (cm$^{-3}$)', fontsize=12)
        plt.ylabel('Altitude (m)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        return df
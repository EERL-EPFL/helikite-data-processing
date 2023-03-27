from typing import Dict, Any, List, Optional, Union, Callable
from pydantic import BaseModel
from pandas import DataFrame
from plotly.graph_objects import Figure
from datetime import datetime
import pandas as pd


class Instrument:
    def __init__(
        self,
        dtype: Dict[Any, Any]={},             # Mapping of column to data type
        na_values: List[Any] | None = None,   # List of values to consider null
        header: int | None = 0,               # Row ID for the header
        delimiter: str = ",",                 # String delimiter
        lineterminator: str | None = None,    # The character to define EOL
        comment: str | None = None,           # Ignore anything after set char
        names: List[str] | None = None,       # Names of headers if non existant
        index_col: bool | int | None = None,  # The column ID of the index
        cols_export: List[str] = [],          # Columns to export 
        cols_housekeeping: List[str] = [],    # Columns to use for housekeeping
        export_order: int | None = None,      # Order hierarchy in export file
        pressure_variable: str | None = None  # The variable measuring pressure
    ) -> None:

        self.dtype = dtype
        self.na_values = na_values
        self.header = header
        self.delimiter = delimiter
        self.lineterminator = lineterminator
        self.comment = comment
        self.names = names
        self.index_col = index_col
        self.cols_export = cols_export
        self.cols_housekeeping = cols_housekeeping
        self.export_order = export_order
        self.pressure_variable = pressure_variable
        
        # Properties that are not part of standard config, can be added
        self.filename: str | None = None
        self.date: datetime | None = None
        self.pressure_offset_housekeeping: float | None = None
        self.time_offset: Dict[str, int] = {}



    def data_corrections(self, df):
        ''' Default callback function for data corrections.

        Return with no changes
        '''

        return df

    def create_plots(self, df: DataFrame):
        ''' Default callback for generated figures from dataframes

        Return nothing, as anything else will populate the list that is written out
        to HTML.
        '''

        return

    def file_identifier(self, first_lines_of_csv: List[str]):
        ''' Default file identifier callback

        Must return false. True would provide false positives.
        '''

        return False

    def date_extractor(self, first_lines_of_csv: List[str]):
        ''' Returns the date of the data sample from a CSV header

        Can be used for an instrument that reports the date in header
        instead of in the data field.

        Return None if there is nothing to do here
        '''

        return None
    
    def get_housekeeping_data(
            self,
            df, 
            pressure_housekeeping_var='housekeeping_pressure'
        ) -> pd.DataFrame:
        ''' Returns the dataframe of housekeeping variables 
        
        If there are no housekeeping variables, return the original dataframe
        '''
        
        if self.cols_housekeeping:
            return df.copy()[self.cols_housekeeping]
        
        return df.copy()
        
        
    def get_export_data(self, df) -> pd.DataFrame:
        ''' Returns the dataframe of only the columns to export 
        
        If there are no columns set in the Instrument class, the default
        behaviour is to return the dataframe with all of the columns
        '''
        
        if self.cols_export:
            return df.copy()[self.cols_export]
        else:
            print("There are no export variables set for this instrument, "
                  "returning all")
            return df.copy()
            
    def correct_from_time_offset(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        ''' Using values in the time_offset variable, correct DateTime index '''
        
        if (
            self.time_offset['hour'] != 0 
            or self.time_offset['minute'] != 0
            or self.time_offset['second'] != 0
        ):
            print(f"Shifting the time offset by {self.time_offset}")
            
            df.index = df.index + pd.DateOffset(
                hours=self.time_offset['hour'], 
                minutes=self.time_offset['minute'], 
                seconds=self.time_offset['second'])
            
        
        return df
    
    def set_housekeeping_pressure_offset_variable(
        self, 
        df: pd.DataFrame,
        column_name="housekeeping_pressure"
    ) -> pd.DataFrame:
        ''' Generate variable to offset pressure value for housekeeping
        
        Using an offset in the configuration, a new variable is created
        that offset's the instruments pressure variable. This is used to align
        the pressure value on the plot to help align pressure. 
        '''
        
        if self.pressure_variable is not None:
            if self.pressure_offset_housekeeping is None:
                # If no offset, but a pressure var exists add column of same val
                df[column_name] = df[self.pressure_variable] 
            else:
                df[column_name] = df[self.pressure_variable] + self.pressure_offset_housekeeping
        
        return df
        
        
    def read_data(
        self
    ) -> pd.DataFrame:
        ''' Read data into dataframe

        This allows a custom read function to parse the CSV/TXT into a
        dataframe, for example cleaning dirty data at the end of the file
        in memory without altering the input file (see flight computer conf).

        '''

        df = pd.read_csv(
            self.filename,
            dtype=self.dtype,
            na_values=self.na_values,
            header=self.header,
            delimiter=self.delimiter,
            lineterminator=self.lineterminator,
            comment=self.comment,
            names=self.names,
            index_col=self.index_col,
        )
        
        
        return df

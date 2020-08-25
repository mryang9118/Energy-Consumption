import pandas as pd
import numpy as np


def clean_ev_data(old_path, new_path):
    """remove missing values (comment it after the first run)"""
    ds = pd.read_csv(filepath_or_buffer=old_path)
    ds = ds[pd.notnull(obj=ds['quantity(kWh)'])]
    ds = ds[pd.notnull(obj=ds['tire_type'])]
    ds = ds[pd.notnull(obj=ds['driving_style'])]
    ds = ds[pd.notnull(obj=ds['consumption(kWh/100km)'])]
    ds = ds[pd.notnull(obj=ds['avg_speed(km/h)'])]
    ds = ds[pd.notnull(obj=ds['trip_distance(km)'])]
    ds.to_csv(path_or_buf=new_path, index=False)

    """load the data"""
    dataframe = pd.read_csv(filepath_or_buffer=new_path)
    # clean some abnormal data
    filter_condition = np.abs(dataframe['quantity(kWh)'] / dataframe['trip_distance(km)'] * 100
                              - dataframe['consumption(kWh/100km)']) < dataframe['consumption(kWh/100km)'] / 2
    return dataframe[filter_condition]


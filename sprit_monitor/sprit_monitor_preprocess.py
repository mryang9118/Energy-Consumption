import pandas as pd
import numpy as np


def clean_ev_data(old_path, new_path):
    """remove missing values (comment it after the first run)"""
    ds = pd.read_csv(filepath_or_buffer=old_path)
    ds = ds[pd.notnull(obj=ds['power(kW)'])]
    ds = ds[pd.notnull(obj=ds['quantity(kWh)'])]
    ds = ds[pd.notnull(obj=ds['tire_type'])]
    ds = ds[pd.notnull(obj=ds['driving_style'])]
    ds = ds[pd.notnull(obj=ds['consumption(kWh/100km)'])]
    ds = ds[pd.notnull(obj=ds['avg_speed(km/h)'])]
    ds = ds[pd.notnull(obj=ds['trip_distance(km)'])]
    ds.to_csv(path_or_buf=new_path, index=False)

    """load the data"""
    data_frame = pd.read_csv(filepath_or_buffer=new_path)
    # clean some abnormal data
    filter_condition = np.abs(data_frame['quantity(kWh)'] / data_frame['trip_distance(km)'] * 100
                              - data_frame['consumption(kWh/100km)']) < data_frame['consumption(kWh/100km)'] / 2
    return data_frame[filter_condition]

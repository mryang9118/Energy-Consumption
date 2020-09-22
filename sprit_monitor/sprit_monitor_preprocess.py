import pandas as pd
import numpy as np


def clean_ev_data(file_path):
    """remove missing values (comment it after the first run)"""
    ds = pd.read_csv(filepath_or_buffer=file_path)
    ds = ds[pd.notnull(obj=ds['power(kW)'])]
    ds = ds[pd.notnull(obj=ds['quantity(kWh)'])]
    ds = ds[pd.notnull(obj=ds['tire_type'])]
    ds = ds[pd.notnull(obj=ds['driving_style'])]
    ds = ds[pd.notnull(obj=ds['consumption(kWh/100km)'])]
    ds = ds[pd.notnull(obj=ds['avg_speed(km/h)'])]
    ds = ds[pd.notnull(obj=ds['trip_distance(km)'])]
    # ds.to_csv(path_or_buf=new_path, index=False)

    """load the data"""
    # data_frame = pd.read_csv(filepath_or_buffer=new_path)
    # clean some abnormal data
    return ds[get_sprit_monitor_filter_condition(ds)]


def get_sprit_monitor_filter_condition(dataset):
    return np.abs(dataset['quantity(kWh)'] / dataset['trip_distance(km)'] * 100 - \
                  dataset['consumption(kWh/100km)']) < dataset['consumption(kWh/100km)'] / 2

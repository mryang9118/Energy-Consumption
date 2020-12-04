import pandas as pd
import numpy as np


def clean_ev_data(file_path, sep=","):
    """remove missing values (comment it after the first run)"""
    ds = pd.read_csv(filepath_or_buffer=file_path, sep=sep)
    drop_set_ = ['power(kW)', 'quantity(kWh)', 'tire_type', 'driving_style', 'consumption(kWh/100km)', 'avg_speed(km/h)', 'trip_distance(km)']
    ds = ds.dropna(axis=0, subset=drop_set_)
    # clean some abnormal data
    return ds[get_sprit_monitor_filter_condition(ds)]


def get_sprit_monitor_filter_condition(dataset):
    return np.abs(dataset['quantity(kWh)'] / dataset['trip_distance(km)'] * 100 - \
                  dataset['consumption(kWh/100km)']) < dataset['consumption(kWh/100km)'] / 2

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class SpritMonitorPreProcess:

    @staticmethod
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

    @staticmethod
    def preprocess_ev_data(data_frame):
        X = data_frame[['power(kW)', 'quantity(kWh)', 'tire_type', 'city',
                        'motor_way', 'country_roads', 'driving_style',
                        'consumption(kWh/100km)', 'A/C', 'park_heating', 'avg_speed(km/h)']].values
        y = data_frame[['trip_distance(km)']].values
        # calculate the distinct values
        power_num = data_frame.groupby(['power(kW)']).ngroups
        tire_num = data_frame.groupby(['tire_type']).ngroups
        label_encoder_1 = LabelEncoder()
        X[:, 0] = label_encoder_1.fit_transform(y=X[:, 0])
        label_encoder_1 = LabelEncoder()
        X[:, 2] = label_encoder_1.fit_transform(y=X[:, 2])
        label_encoder_2 = LabelEncoder()
        X[:, 6] = label_encoder_2.fit_transform(y=X[:, 6])
        # onehot encoding for categorical features with more than 2 categories
        onehot_encoder = OneHotEncoder(categorical_features=[0, 2, 6])
        X = onehot_encoder.fit_transform(X=X).toarray()
        # delete the first column code of each encoded feature to avoid the dummy variable
        X = np.delete(X, [0, power_num, power_num + tire_num], 1)
        return X, y

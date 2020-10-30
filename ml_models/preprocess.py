import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(data_frame, x_columns_name, y_column_name, require_encoded_columns):
    """
    :param data_frame:
    :type 'pandas.core.frame.DataFrame':
    :param x_columns_name, eg.['a', 'b', 'c']:
    :type list:
    :param y_column_name, eg. ['y']:
    :type list:
    :param require_encoded_columns, eg.['a', 'b']:
    :type list, note: assign by order:
    :return tuple X, y:
    :rtype:
    """
    x_data_frame = data_frame[x_columns_name]
    y = data_frame[y_column_name].values
    column_transformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), require_encoded_columns)], remainder='passthrough')
    X = column_transformer.fit_transform(X=x_data_frame)
    return X, y

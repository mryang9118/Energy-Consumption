import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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
    X = x_data_frame.values
    y = data_frame[y_column_name].values
    # calculate the distinct values, please assign require_encoded_columns by order
    onehot_index_list = []
    feature_count_list = []
    for single_column in require_encoded_columns:
        distinct_count = x_data_frame.groupby([single_column]).ngroups
        if distinct_count < 2:
            continue
        column_index = x_data_frame.columns.get_loc(single_column)
        label_encoder = LabelEncoder()
        X[:, column_index] = label_encoder.fit_transform(y=X[:, column_index])
        if distinct_count > 2:
            onehot_index_list.append(column_index)
            feature_count_list.append(distinct_count)
    # onehot encoding for categorical features with more than 2 categories
    onehot_encoder = OneHotEncoder(categorical_features=onehot_index_list)
    X = onehot_encoder.fit_transform(X=X).toarray()
    # delete the first column code of each encoded feature to avoid the dummy variable
    delete_column_list = []
    start = 0
    for i in range(0, len(feature_count_list)):
        delete_column_list.append(start)
        start += feature_count_list[i]
    X = np.delete(X, delete_column_list, 1)
    return X, y

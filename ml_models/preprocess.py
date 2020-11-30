import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings


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
    :return [X, feature_names], y:
    :rtype: array: [X, feature_names] X: arrya input matrix, feature_names: encoded feature names
    """
    x_data_frame = data_frame[x_columns_name]
    y = data_frame[y_column_name].values
    column_transformer = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), require_encoded_columns)],
                                           remainder='passthrough')
    X = column_transformer.fit_transform(X=x_data_frame)
    feature_names = get_feature_names(column_transformer)
    return [feature_names, X], y


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings Names of the features produced by transform.
    """

    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                          "provide get_feature_names. "
                          "Will return input column names if available"
                          % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]

    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    return feature_names

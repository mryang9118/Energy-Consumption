from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


def random_forest_regress_model(n_estimators, criterion="mae", oob_score=True, warm_start=False):
    return RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, oob_score=oob_score,
                                 warm_start=warm_start)


def ada_boost_regress_model(n_estimators, learning_rate):
    return AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)


def mlp_regress_model(hidden_layer_sizes, max_iter, n_iter_no_change, activation='relu',
                      solver='adam', verbose=False, warm_start=False):
    return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
                        n_iter_no_change=n_iter_no_change, activation=activation,
                        solver=solver, verbose=verbose, warm_start=warm_start)


def deep_mlp_model(build_fn, param_dict):
    return KerasRegressor(build_fn=build_fn, batch_size=param_dict.get('batch_size'),
                          epochs=param_dict.get('epochs'), verbose=False)




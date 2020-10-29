from sklearn.model_selection import ShuffleSplit, cross_validate, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def evaluate_general_model(estimator, x, y, scoring, n_splits=10, test_size=0.5, verbose=0):
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=2)
    scoring_array = cross_validate(estimator=estimator, X=x, y=y, cv=cv,
                                   scoring=scoring, verbose=verbose)
    for fun in scoring:
        fun_key = 'test_%s' % fun
        if fun_key in scoring_array:
            fun_values = scoring_array[fun_key]
            print('average %s value is: %s' % (fun, abs(round(number=fun_values.mean(), ndigits=3))))
            best_value = sorted(fun_values, reverse=False)[-1]
            print("best %s value is: %s" % (fun, abs(round(number=best_value, ndigits=3))))
            print("-------------------------------")


def evaluate_deep_mlp(model, x_matrix, y_matrix, metrics):
    dmlp_error = model.evaluate(x_matrix, y_matrix, verbose=0, return_dict=True)
    for single_metric in metrics:
        print('%s value is: %.3f' % (single_metric, dmlp_error[single_metric]))


def evaluate_predict_result(y_true, y_pred):
    print("RMSE value: %.3f" % np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)))
    print("MAE value: %.3f" % mean_absolute_error(y_true=y_true, y_pred=y_pred))
    print("Variance score: %.3f" % (r2_score(y_true=y_true, y_pred=y_pred)))



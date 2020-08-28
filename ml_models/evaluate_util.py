from sklearn.model_selection import ShuffleSplit, cross_validate


def evaluate_model(estimator, x, y, scoring):
    cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
    results = cross_validate(estimator=estimator, X=x, y=y, cv=cv, scoring=scoring, verbose=2)
    return results


def report_results(results, scoring_array):
    for fun in scoring_array:
        fun_key = 'test_%s' % fun
        if fun_key in results:
            fun_values = results[fun_key]
            print('average %s value is: %s' % (fun, abs(round(number=fun_values.mean(), ndigits=3))))
            best_value = sorted(fun_values, reverse=False)[-1]
            print("best %s value is:", abs(round(number=best_value, ndigits=3)))
            print("-------------------------------")

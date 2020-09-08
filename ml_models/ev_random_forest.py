from sklearn.ensemble import RandomForestRegressor


def get_rf_model():
    return RandomForestRegressor(n_estimators=200, criterion="mae", oob_score=True, warm_start=False)

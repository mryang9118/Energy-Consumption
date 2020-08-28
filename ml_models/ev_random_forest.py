from sklearn.ensemble import RandomForestRegressor


class EVRandomForestModel:

    @staticmethod
    def get_model():
        return RandomForestRegressor(n_estimators=200, criterion="mae", oob_score=True, warm_start=False)



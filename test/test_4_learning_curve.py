"""
@Project: Energy-Consumption   
@Description: test the learning curve with random forest model
@Time:2020/10/27 18:20                      
 
"""
from sklearn.pipeline import Pipeline

from utils import *
from sklearn.preprocessing import StandardScaler
from ml_models import ModelsFitter
import warnings

warnings.filterwarnings(action="ignore")
data_path = "../data/tesla_model3_tn.csv"
after_clean = clean_ev_data(data_path)
X, y = preprocess_data(after_clean, SPRIT_MONITOR_X_COLUMN_NAMES, SPRIT_MONITOR_TARGET_COLUMN_NAME,
                       SPRIT_MONITOR_REQUIRE_ENCODED_COLUMNS)
model_fitter = ModelsFitter(RF)
pipeline = Pipeline([
        (STANDARD_SCALAR, StandardScaler()),
        (ESTIMATOR, model_fitter)
    ])
pipeline.fit(X[1], y)
# model_fitter.plot_learning_curve()
model_fitter.calculate_feature_importance(X[0])

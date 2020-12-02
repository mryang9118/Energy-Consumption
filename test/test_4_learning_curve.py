"""
@Project: Energy-Consumption   
@Description: test the learning curve with random forest model
@Time:2020/10/27 18:20                      
 
"""
from sklearn.pipeline import make_pipeline

from ml_models.preprocess import *
from utils.constants import *
from sprit_monitor.sprit_monitor_preprocess import *
from sklearn.preprocessing import StandardScaler
from ml_models.parse_models import ModelsFitter
import warnings

warnings.filterwarnings(action="ignore")
data_path = "../data/tesla_model3_tn.csv"
after_clean = clean_ev_data(data_path)
X, y = preprocess_data(after_clean, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)
model_fitter = ModelsFitter(RF)
pipeline = make_pipeline(StandardScaler(), model_fitter)
pipeline.fit(X[1], y)
# model_fitter.plot_learning_curve()
model_fitter.calculate_feature_importance(X[0])

"""
@Project: Energy-Consumption   
@Description: test the learning curve with random forest model
@Time:2020/10/27 18:20                      
 
"""
from ml_models.preprocess import *
from utils.constants import *
from sprit_monitor.sprit_monitor_preprocess import *
from sklearn.preprocessing import StandardScaler
from ml_models.parse_models import ModelsFitter
import warnings

warnings.filterwarnings(action="ignore")
data_path = "../data/volkswagen_e_golf_85_power.csv"
after_clean = clean_ev_data(data_path)
X, y = preprocess_data(after_clean, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)
sc = StandardScaler()
X_matrix = sc.fit_transform(X=X[1])
model_fitter = ModelsFitter(RF, X_matrix, y)
model_fitter.fit_model()
model_fitter.plot_learning_curve()
model_fitter.calculate_feature_importance(X[0])

"""
@Project: Energy-Consumption   
@Description: an application of ev
@Time:2020/9/21 16:29                      
 
"""
import warnings
from sprit_monitor.sprit_monitor_preprocess import *
from sklearn.preprocessing import StandardScaler
from ml_models.preprocess import *
from utils.constants import *
from ml_models.parse_models import ModelsGetter
from ml_models.evaluate_util import *

warnings.filterwarnings(action="ignore")
file_path = "../data/volkswagen_e_golf_85_power.csv"
# preprocess ev data from sprit monitor
after_clean = clean_ev_data(file_path)
X, y = preprocess_data(after_clean, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)
# scale the values
sc = StandardScaler()
X = sc.fit_transform(X=X)
getter = ModelsGetter(DEEP_MLP, X, y)
# train the model, and evaluate
getter.process()


model = getter.get_model()
# use the test data for predict, just for example
y_pred = model.predict(X[0:2000, ])
evaluate_predict_result(y[0:2000, ], y_pred)


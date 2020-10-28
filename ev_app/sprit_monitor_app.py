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
from ml_models.parse_models import ModelsFitter
from ml_models.evaluate_util import *
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action="ignore")
file_path = "../data/Volkswagen_100.csv"
# preprocess ev data from sprit monitor
after_clean = clean_ev_data(file_path)
X, y = preprocess_data(after_clean, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)
    # scale the values
    sc = StandardScaler()
    X_train = sc.fit_transform(X=X_train)
    X_test = sc.transform(X=X_test)
    # train the model, and evaluate
    print('---------------------------%s time Start---------------------------'% i)
    getter = ModelsFitter(DEEP_MLP, X_train, y_train)
    getter.process()
    model = getter.get_model()
    # use the test data for predict, just for example
    y_pred = model.predict(X_test)
    print('y test: %s' % str(y_test).replace('\n', ' '))
    print('y prediction: %s' % str(y_pred))
    evaluate_predict_result(y_test, y_pred)
    getter.save_model()
    print('---------------------------%s time End---------------------------'% i)


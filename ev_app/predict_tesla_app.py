"""
@Project: Energy-Consumption   
@Description: an application of mixed data with tesla and volkswagen
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
train_path = "../data/mix_e-golf_tesla.csv"
test_path = "../data/half_tesla_data.csv"
# preprocess ev data from sprit monitor
train_data = clean_ev_data(train_path)
test_data = clean_ev_data(test_path)
X, y = preprocess_data(train_data, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)
X_test, y_test = preprocess_data(test_data, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)

for i in range(10):
    # scale the values
    # train the model, and evaluate
    X_train, X_abandon_test, y_train, y_abandon_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    X_init_test = X_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X=X_train)
    X_init_test = sc.transform(X=X_init_test)
    print('---------------------------%s time Start---------------------------' % i)
    getter = ModelsFitter(RF, X_train, y_train)
    getter.process()
    model = getter.get_model()
    # use the test data for predict, just for example
    y_pred = model.predict(X_init_test)
    print('y test: %s' % str(y_test).replace('\n', ' '))
    print('y prediction: %s' % str(y_pred))
    evaluate_predict_result(y_test, y_pred)
    print('---------------------------%s time End---------------------------' % i)

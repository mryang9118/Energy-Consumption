"""
@Project: Energy-Consumption   
@Description: an application of mixed data with tesla and volkswagen
@Time:2020/9/21 16:29

"""

from sklearn.pipeline import make_pipeline

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

for i in range(1):
    # scale the values
    # train the model, and evaluate
    X_train, X_abandon_test, y_train, y_abandon_test = train_test_split(X[1], y, test_size=0.1, shuffle=True)
    X_init_test = X_test[1]

    pipe_line = make_pipeline(StandardScaler(), ModelsFitter(RF))
    pipe_line.fit(X_train, y_train)
    y_pred = pipe_line.predict(X_init_test)
    print('y test: %s' % str(y_test).replace('\n', ' '))
    print('y prediction: %s' % str(y_pred))
    evaluate_predict_result(y_test, y_pred)

    print('---------------------------%s time End---------------------------' % i)

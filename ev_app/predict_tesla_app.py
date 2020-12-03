"""
@Project: Energy-Consumption   
@Description: an application of mixed data with tesla and volkswagen
@Time:2020/9/21 16:29

"""

from sklearn.pipeline import make_pipeline
from utils import *
from sklearn.preprocessing import StandardScaler
from ml_models import ModelsFitter
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action="ignore")
train_path = "../data/mix_e-golf_tesla.csv"
test_path = "../data/half_tesla_data.csv"
# preprocess ev data from sprit monitor
train_data = clean_ev_data(train_path)
test_data = clean_ev_data(test_path)
X = train_data[SPRIT_MONITOR_X_COLUMN_NAMES]
y = train_data[SPRIT_MONITOR_TARGET_COLUMN_NAME].values
X_test = test_data[SPRIT_MONITOR_X_COLUMN_NAMES]
y_test = test_data[SPRIT_MONITOR_TARGET_COLUMN_NAME].values

for i in range(1):
    # scale the values
    # train the model, and evaluate
    X_train, X_abandon_test, y_train, y_abandon_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    X_init_test = X_test
    pipe_line = make_pipeline(get_column_transformer(SPRIT_MONITOR_REQUIRE_ENCODED_COLUMNS), StandardScaler(), ModelsFitter(RF))
    pipe_line.fit(X_train, y_train)
    y_pred = pipe_line.predict(X_init_test)
    print('y test: %s' % str(y_test).replace('\n', ' '))
    print('y prediction: %s' % str(y_pred))
    evaluate_predict_result(y_test, y_pred)
    print('---------------------------%s time End---------------------------' % i)

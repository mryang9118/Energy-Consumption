"""
@Project: Energy-Consumption   
@Description: tesla application
@Time:2020/10/14 15:41                      
 
"""
import warnings
from sprit_monitor.sprit_monitor_preprocess import *
from sklearn.preprocessing import StandardScaler
from ml_models.preprocess import *
from utils.constants import *
from ml_models.evaluate_util import *
from sklearn.model_selection import train_test_split
from ml_models.models_getter import *

warnings.filterwarnings(action="ignore")
file_path = "../data/tesla_trace.csv"
SOURCE_COLUMN_NAMES = ['battery_range', 'speed', 'temperature', 'temperature_setting', 'battery_level']
TARGET_COLUMN_NAMES = ['odometer']
# preprocess ev data from sprit monitor
raw_data = pd.read_csv(filepath_or_buffer=file_path, delimiter=' ')
X, y = preprocess_data(raw_data, SOURCE_COLUMN_NAMES, TARGET_COLUMN_NAMES, [])
rf_model = get_model(RF, True)
# deep_mlp_model = get_model(DEEP_MLP, True)

for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X[1], y, test_size=0.4, shuffle=True)
    # scale the values
    sc = StandardScaler()
    X_train = sc.fit_transform(X=X_train)
    X_test = sc.transform(X=X_test)
    # RF Model
    print('---------------------------%s time Start---------------------------'% i)
    print('-------Random Forest Start-------')
    model = get_model(RF, tuned=False, save_model=True, x_matrix=X_train, y_matrix=y_train)
    print('y test: %s' % str(y_test).replace('\n', ' '))
    y_pred_fit = model.predict(X_test)
    print('1. Fit model real time: y prediction: %s' % str(y_pred_fit))
    evaluate_predict_result(y_test, y_pred_fit)

    y_pred_tuned = rf_model.predict(X_test)
    print('2. Use tuned model: y prediction: %s' % str(y_pred_tuned))
    evaluate_predict_result(y_test, y_pred_tuned)

    print('-------Random Forest End-------')
    # Deep MLP Model
    # print('-------Deep MLP Start-------')
    # model = get_model(DEEP_MLP, tuned=False, save_model=True, x_matrix=X_train, y_matrix=y_train)
    # print('y test: %s' % str(y_test).replace('\n', ' '))
    # y_pred_fit = model.predict(X_test)
    # print('1. Fit model real time: y prediction: %s' % str(y_pred_fit))
    # evaluate_predict_result(y_test, y_pred_fit)
    #
    # y_pred_tuned = deep_mlp_model.predict(X_test)
    # print('2. Use tuned model: y prediction: %s' % str(y_pred_tuned))
    # evaluate_predict_result(y_test, y_pred_tuned)
    # print('-------Deep MLP End-------')
    print('---------------------------%s time End---------------------------'% i)


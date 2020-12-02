"""
@Project: Energy-Consumption   
@Description: an application of ev
@Time:2020/9/21 16:29                      
 
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_models import ModelsFitter
from utils import *
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action="ignore")
file_path = "../data/mix_e-golf_tesla.csv"
# preprocess ev data from sprit monitor
after_clean = clean_ev_data(file_path)
X, y = preprocess_data(after_clean, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X[1], y, test_size=0.2, shuffle=True)
    # train the model, and evaluate
    print('---------------------------%s time Start---------------------------'% i)
    fitter = ModelsFitter(DEEP_MLP)
    # use the test data for predict, just for example
    pipeline = make_pipeline(StandardScaler(), fitter)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print('y test: %s' % str(y_test).replace('\n', ' '))
    print('y prediction: %s' % str(y_pred))
    evaluate_predict_result(y_test, y_pred)
    fitter.calculate_feature_importance(X[0])
    fitter.save_model()
    print('---------------------------%s time End---------------------------'% i)


# load and evaluate a saved model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import load_model
from ml_models.preprocess import *
from sprit_monitor.sprit_monitor_constants import *
# load model
model = load_model('volkswagen_e_golf_model.h5')
# summarize model.
model.summary()
# load dataset
dataset = pd.read_csv(filepath_or_buffer ='../data/volkswagen_e_golf_data.csv')
filter_condition = np.abs(dataset['quantity(kWh)']/dataset['trip_distance(km)'] * 100
                          - dataset['consumption(kWh/100km)']) < dataset['consumption(kWh/100km)']/2
dataset = dataset[filter_condition]
X, y = preprocess_data(dataset, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)

# split the dataset into training-set and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# scale the values
sc = StandardScaler()
X_test = sc.fit_transform(X=X_test)
test_pred = model.predict(X_test)
print("RMSE on test data: %.3f" % np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_pred)))
print("MAE on test data: %.3f" % mean_absolute_error(y_true=y_test, y_pred=test_pred))
print("variance score on test data: %.3f" % r2_score(y_true=y_test, y_pred=test_pred))

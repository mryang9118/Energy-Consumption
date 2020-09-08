import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ml_models.preprocess import *
from sprit_monitor.sprit_monitor_constants import *


def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=100, kernel_initializer='uniform', activation='relu', input_dim=len(X[0])))
    regressor.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(units=25, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(units=13, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
    # activation func of the output layer must be 'linear' for regression tasks
    regressor.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))
    regressor.compile(optimizer='adam', loss='mean_absolute_error')
    return regressor


new_path = "../data/volkswagen_e_golf_data.csv"
dataset = pd.read_csv(filepath_or_buffer=new_path)
filter_condition = np.abs(dataset['quantity(kWh)'] / dataset['trip_distance(km)'] * 100
                          - dataset['consumption(kWh/100km)']) < dataset['consumption(kWh/100km)'] / 2
dataset = dataset[filter_condition]
X, y = preprocess_data(dataset, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)
# split the dataset into training-set and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# scale the values
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
# X_test = sc.fit_transform(X=X_test)
deep_mlp = KerasRegressor(build_fn=build_regressor, batch_size=8, epochs=50, verbose=False)
deep_mlp.fit(X_train, y_train)
deep_mlp.model.save("./volkswagen_e_golf_model_u.h5")
print("Saved model to disk")

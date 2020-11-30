# Use scikit-learn to grid search the batch size and epochs
import numpy
import pandas as pd
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from ml_models.preprocess import *
from utils.constants import *

# Function to create model, required for KerasClassifier
from tensorflow.keras.layers import Dropout


def build_regressor(dropout_rate=0.0):
    regressor = Sequential()
    regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(units=100, kernel_initializer='uniform', activation='relu', input_dim=len(X[0])))
    regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(units=25, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(units=13, kernel_initializer='uniform', activation='relu'))
    regressor.add(Dropout(dropout_rate))
    regressor.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
    # activation func of the output layer must be 'linear' for regression tasks
    regressor.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))
    regressor.compile(optimizer='adam', loss='mean_absolute_error')
    return regressor


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
new_path = "../data/volkswagen_e_golf_data.csv"
# load dataset
dataset = pd.read_csv(filepath_or_buffer=new_path)
# preprocess dataset
X, y = preprocess_data(dataset, X_COLUMN_NAMES, Y_COLUMN_NAME, REQUIRE_ENCODED_COLUMNS)

# create model
model = KerasRegressor(build_fn=build_regressor, batch_size=8, epochs=50, verbose=0)
# define the grid search parameters
dropout_rate = [0.0, 0.2, 0.3, 0.4, 0.5]
param_grid = dict(dropout_rate=dropout_rate)
cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
grid = GridSearchCV(estimator=model, scoring='r2', param_grid=param_grid, n_jobs=1, cv=cv)
grid_result = grid.fit(X[1], y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

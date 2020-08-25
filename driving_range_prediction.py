import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import util


def do_kfold(model):
    cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
    results = cross_validate(estimator=model, X=X, y=y, cv=cv, scoring=['neg_mean_absolute_error', 'r2'])
    mae_values = results['test_neg_mean_absolute_error']
    r2_scores = results['test_r2']
    return mae_values, r2_scores


def do_fit_predict(model):
    model.fit(X_train, y_train)
    training_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return training_pred, test_pred


def report_cross_val_results(mae_values, r2_scores):
    print("average MAE values (bias) is:", abs(round(number=mae_values.mean(), ndigits=3)))
    print("std deviation of MAE values (variance) is:", round(number=mae_values.std(), ndigits=3))
    best_mae = sorted(mae_values, reverse=False)[-1]
    print("best MAE value is:", abs(round(number=best_mae, ndigits=3)))

    print("average r2 scores (bias) is:", round(number=r2_scores.mean() * 100, ndigits=3))
    print("std deviation of r2 scores (variance) is:", round(number=r2_scores.std() * 100, ndigits=3))
    best_r2 = sorted(r2_scores, reverse=False)[-1] * 100
    print("best r2 score is:", round(number=best_r2, ndigits=3))
    print("-------------------------------")


def report_results(training_pred, test_pred):
    print("RMSE on training data: %.3f" % np.sqrt(mean_squared_error(y_true=y_train, y_pred=training_pred)))
    print("RMSE on test data: %.3f" % np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_pred)))
    print("MAE on training data: %.3f" % mean_absolute_error(y_true=y_train, y_pred=training_pred))
    print("MAE on test data: %.3f" % mean_absolute_error(y_true=y_test, y_pred=test_pred))
    print("variance score on training data: %.3f" % r2_score(y_true=y_train, y_pred=training_pred))
    print("variance score on test data: %.3f" % r2_score(y_true=y_test, y_pred=test_pred))
    print("-------------------------------")

warnings.filterwarnings(action="ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

old_path = "./data/volkswagen_e_golf_data.csv"
new_path = "./data/volkswagen_e_golf_data_clean.csv"
data_frame = util.clean_ev_data(old_path, new_path)
X = data_frame[['power(kW)', 'quantity(kWh)', 'tire_type', 'city',
               'motor_way', 'country_roads', 'driving_style',
               'consumption(kWh/100km)', 'A/C', 'park_heating', 'avg_speed(km/h)']].values
y = data_frame[['trip_distance(km)']].values

# calculate the distinct values
power_num = data_frame.groupby(['power(kW)']).ngroups
tire_num = data_frame.groupby(['tire_type']).ngroups
dstyle_num = data_frame.groupby(['driving_style']).ngroups

"""do the preprocessing tasks on the data"""
# encode categorical features
label_encoder_1 = LabelEncoder()
X[:, 0] = label_encoder_1.fit_transform(y=X[:, 0])
label_encoder_1 = LabelEncoder()
X[:, 2] = label_encoder_1.fit_transform(y=X[:, 2])
label_encoder_2 = LabelEncoder()
X[:, 6] = label_encoder_2.fit_transform(y=X[:, 6])

# onehot encoding for categorical features with more than 2 categories
onehot_encoder = OneHotEncoder(categorical_features=[0, 2, 6])
X = onehot_encoder.fit_transform(X=X).toarray()

# delete the first column code of each encoded feature to avoid the dummy variable
X = np.delete(X, [0, power_num, power_num + tire_num], 1)

# split the dataset into training-set and test-set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# scale the values
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
X_test = sc.fit_transform(X=X_test)
# y_train = sc.fit_transform(X=y_train)
# y_test = sc.fit_transform(X=y_test)


"""define the linear regression model"""
linear_regressor = LinearRegression()

"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
print("\n ------ Linear Regression CrossVal ------")
reg_mae_values, reg_r2_scores = do_kfold(model=linear_regressor)
report_cross_val_results(mae_values=reg_mae_values, r2_scores=reg_r2_scores)

"""train the linear regression model and print the results on the never-seen-before test data"""
print("\n ------ Linear Regression TrainTest ------")
reg_training_pred, reg_test_pred = do_fit_predict(model=linear_regressor)
report_results(training_pred=reg_training_pred, test_pred=reg_test_pred)

"""define the random forest ensemble model"""
rf = RandomForestRegressor(n_estimators=200, criterion="mae", oob_score=True, warm_start=False)

"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
print("\n ------ Random Forest CrossVal ------")
rf_mae_values, rf_r2_scores = do_kfold(model=rf)
report_cross_val_results(mae_values=rf_mae_values, r2_scores=rf_r2_scores)

"""train the RF model and print the results on the never-seen-before test data"""
print("\n ------ Random Forest TrainTest ------")
rf_train_pred, rf_test_pred = do_fit_predict(model=rf)
report_results(training_pred=rf_train_pred, test_pred=rf_test_pred)

"""define the deep multi-layer perceptron model"""


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


deep_mlp = KerasRegressor(build_fn=build_regressor, batch_size=8, epochs=50, verbose=False)

"""do the KFold cross-validation both with MAE values and r2 scores criteria"""
print("\n ------ Deep MLP CrossVal ------")
deep_mlp_mae_values, deep_mlp_r2_scores = do_kfold(model=deep_mlp)
report_cross_val_results(mae_values=deep_mlp_mae_values, r2_scores=deep_mlp_r2_scores)

"""train the deep mlp model and print the results on the never-seen-before test data"""
print("\n ------ Deep MLP TrainTest ------")
deep_mlp_train_pred, deep_mlp_test_pred = do_fit_predict(model=deep_mlp)
report_results(training_pred=deep_mlp_train_pred, test_pred=deep_mlp_test_pred)

"""plot driving range based on the battery quantity"""
quantity_index = power_num + tire_num + dstyle_num - 3
quantity = X[:, quantity_index]
distance = y
quantity = np.reshape(quantity, newshape=(-1, 1))
distance = np.reshape(distance, newshape=(-1, 1))

quantity_linear_reg = LinearRegression()
quantity_linear_reg.fit(X=quantity, y=distance)
q_slope = quantity_linear_reg.coef_[0]
q_intercept = quantity_linear_reg.intercept_
q_predicted_distances = q_intercept + q_slope * quantity

fig = plt.figure()
plt.scatter(x=quantity, y=distance, s=15, c='black', linewidths=0.1)
plt.plot(quantity, q_predicted_distances, c='red', linewidth=2)
plt.legend(('fitted line', 'data records'), loc='lower right')
plt.title(label='Linear Regression Plot')
plt.xlabel(xlabel='quantity (kWh)'), plt.ylabel(ylabel='driving range (km)')
plt.show()
# fig.savefig('range_to_quantity.png')


"""plot driving range based on the average speed"""
avg_speed = X[:, quantity_index + 7]
avg_speed = np.reshape(avg_speed, newshape=(-1, 1))

speed_linear_reg = LinearRegression()
speed_linear_reg.fit(X=avg_speed, y=distance)
s_slope = speed_linear_reg.coef_[0]
s_intercept = speed_linear_reg.intercept_
s_predicted_distances = s_intercept + s_slope * quantity

fig = plt.figure()
plt.scatter(x=avg_speed, y=distance, s=15, c='orange', linewidths=0.1)
plt.plot(quantity, s_predicted_distances, c='blue', linewidth=2)
plt.legend(('fitted line', 'data records'), loc='upper left')
plt.title(label='Linear Regression Plot')
plt.xlabel(xlabel='average speed (km/h)'), plt.ylabel(ylabel='driving range (km)')
plt.xlim(-5, 110), plt.ylim(-30, 650)
plt.show()
# fig.savefig('range_to_speed.png')
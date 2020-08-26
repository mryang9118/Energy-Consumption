from sprit_monitor.preprocess_sprit_monitor_ev_data import *
from ml_models.ev_deep_mlp import DeepMLPModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy import sqrt

old_path = "../data/volkswagen_e_golf_85_power.csv"
new_path = "../data/volkswagen_e_golf_85_power_test.csv"
# preprocess ev data from sprit monitor
data_preprocess = SpritMonitorPreProcess()
data_frame, X, y = data_preprocess.preprocess_ev_data(old_path, new_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
# scale the values
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
X_test = sc.fit_transform(X=X_test)
# use the deep dnn model
model = DeepMLPModel(len(X[0]))
model.compile(optimizer='adam', loss='mean_absolute_error')
# fit the model
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))


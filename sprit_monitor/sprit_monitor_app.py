import warnings

from sprit_monitor.preprocess_sprit_monitor_ev_data import *
from ml_models.ev_deep_mlp import DeepMLPModel
from ml_models.ev_random_forest import EVRandomForestModel
from sklearn.preprocessing import StandardScaler
from numpy import sqrt
from ml_models.evaluate_util import *
from sklearn.metrics import SCORERS

warnings.filterwarnings(action="ignore")
old_path = "../data/volkswagen_e_golf_85_power.csv"
new_path = "../data/volkswagen_e_golf_85_power_test.csv"
# preprocess ev data from sprit monitor
after_clean = SpritMonitorPreProcess.clean_ev_data(old_path, new_path)
X, y = SpritMonitorPreProcess.preprocess_ev_data(after_clean)
# scale the values
sc = StandardScaler()
X = sc.fit_transform(X=X)
# refer the scoring function
print(sorted(SCORERS.keys()))
scoring_methods = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
print('----1. Using deep MLP Model to Evaluate----')
# use the deep mlp model
dmlp_model = DeepMLPModel(len(X[0]))
# you can customize your own loss, metrics function
dmlp_model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse', 'mae'])
dmlp_model.fit(X, y, batch_size=8, epochs=50, verbose=2)
# evaluate the model
dmlp_error = dmlp_model.evaluate(X, y, verbose=0, return_dict=True)
print('RMSE: %.3f, MAE: %.3f' % (sqrt(dmlp_error['mse']), dmlp_error['mae']))
# save model to disk or just save weights by save_weights
# model.save(filepath="./volkswagen_model", save_format="tf")
print('----2. Using Random Forest Model to Evaluate----')
rf = EVRandomForestModel.get_model()
rf_error = evaluate_model(rf, X, y, scoring_methods)
report_results(rf_error, scoring_methods)




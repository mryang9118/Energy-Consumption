from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from utils import *


warnings.filterwarnings(action="ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)


new_path = "../data/single_power_Renault_ZOE.csv"

"""load the data"""
dataset = pd.read_csv(filepath_or_buffer=new_path)
dataset = dataset[get_sprit_monitor_filter_condition(dataset)]
X, y = preprocess_data(dataset, SPRIT_MONITOR_X_COLUMN_NAMES, SPRIT_MONITOR_TARGET_COLUMN_NAME, SPRIT_MONITOR_REQUIRE_ENCODED_COLUMNS)
# split the dataset into training-set and test-set
X_train, X_test, y_train, y_test = train_test_split(X[1], y, test_size=0.2, shuffle=True)

# scale the values
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
X_test = sc.transform(X=X_test)

"""define the random forest ensemble model"""
rf = RandomForestRegressor(criterion="mae", warm_start=False)
param_grid={'n_estimators': range(100, 500, 20)}
cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=2)
grid = GridSearchCV(estimator=rf, scoring='r2', param_grid=param_grid, n_jobs=1, cv=cv)
grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

"""
@Project: Energy-Consumption
@Description: Constants for Sprit monitor data
@Time:2020/9/8 17:15

"""

SPRIT_MONITOR_X_COLUMN_NAMES = ['power(kW)', 'quantity(kWh)', 'tire_type', 'city',
                                'motor_way', 'country_roads', 'driving_style',
                                'consumption(kWh/100km)', 'A/C', 'park_heating', 'avg_speed(km/h)']
SPRIT_MONITOR_TARGET_COLUMN_NAME = ['trip_distance(km)']
SPRIT_MONITOR_REQUIRE_ENCODED_COLUMNS = ['tire_type', 'driving_style']
SPRIT_MONITOR = 'sprit monitor'
DEEP_MLP = 'DeepMLP'
RF = 'RandomForest'
MLP = 'MLP'
ADA_BOOST = 'AdaBoost'
PREPROCESS = 'preprocess'
DECOMPOSER = 'decomposer'
STANDARD_SCALAR = 'scalar'
ESTIMATOR = 'estimator'
PREDICT = 'predict_value'
HYPER_PARAMETERS = 'hyperparameters'
PIPELINE_FILE_SUFFIX = 'pipeline.joblib'
MODEL_SUFFIX = 'model'
LAYERS = "layers"
DENSE = 'Dense'
OPTIMIZER = 'optimizer'
METRICS = 'metrics'
EPOCHS = 'epochs'
BATCH_SIZE = 'batch_size'
SCORING_METHODS = 'scoring_methods'
N_ESTIMATORS = 'n_estimators'
LOSS = 'loss'
CRITERION = 'criterion'
N_SPLITS = 'n_splits'
KERNEL_INITIALIZER = 'kernel_initializer'
ACTIVATION = 'activation'
UNITS = 'units'
INPUT_DIM = 'input_dim'
TEST_SIZE = 'test_size'
COMMA = ','
MODEL_SAVED_PATH = '../output'
SPRIT_MONITOR_HEADER = ['manufacturer', 'version', 'fuel_date', 'trip_distance(km)', 'power(kW)', 'quantity(kWh)',
                        'tire_type', 'city', 'motor_way', 'country_roads', 'driving_style', 'consumption(kWh/100km)',
                        'A/C', 'park_heating', 'avg_speed(km/h)']

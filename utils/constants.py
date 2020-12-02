"""
@Project: Energy-Consumption
@Description: Constants for Sprit monitor data
@Time:2020/9/8 17:15

"""

X_COLUMN_NAMES = ['power(kW)', 'quantity(kWh)', 'tire_type', 'city',
                  'motor_way', 'country_roads', 'driving_style',
                  'consumption(kWh/100km)', 'A/C', 'park_heating', 'avg_speed(km/h)']
Y_COLUMN_NAME = ['trip_distance(km)']
REQUIRE_ENCODED_COLUMNS = ['tire_type', 'driving_style']
SPRIT_MONITOR = 'sprit monitor'

DEEP_MLP = 'DeepMLP'
RF = 'RandomForest'
MLP = 'MLP'
ADA_BOOST = 'AdaBoost'

MODEL_SAVED_PATH = '../output'

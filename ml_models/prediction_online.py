"""
@Project: Energy-Consumption   
@Description: predict the test result, and return the value
@Time:2020/12/1 16:02                      
 
"""
import os
import joblib
import pandas as pd

from utils.constants import MODEL_SAVED_PATH

X_COLUMN_NAMES = ['power(kW)', 'quantity(kWh)', 'tire_type', 'city',
                  'motor_way', 'country_roads', 'driving_style',
                  'consumption(kWh/100km)', 'A/C', 'park_heating', 'avg_speed(km/h)']


def predict_result(pipeline_path, test_data_frame):
    """
    :param pipeline_path: the pipeline file path
    :type pipeline_path: str
    :param test_data_frame: input predict matrix X
    :type test_data_frame: data frame
    :return: predict value of input data
    :rtype: array
    """
    if not os.path.exists(pipeline_path):
        print("Not found the file, please check the pipeline file path!")
        return
    pipeline = joblib.load(pipeline_path)
    X_test = test_data_frame[X_COLUMN_NAMES]
    return pipeline.predict(X_test)


if __name__ == '__main__':
    input_array = [['Volkswagen', 'E-Golf 300', '06.01.2019', 28, 100, 10.2, 'Winter tires', 1,	0,	1, 'Normal',
                   24.6, 1, 1, 29]]
    header = ['manufacturer', 'version', 'fuel_date', 'trip_distance(km)', 'power(kW)', 'quantity(kWh)', 'tire_type',
              'city', 'motor_way', 'country_roads',	'driving_style', 'consumption(kWh/100km)',
              'A/C', 'park_heating', 'avg_speed(km/h)']
    test_frame = pd.DataFrame(data=input_array, columns=header)
    pipeline_path = '%s/sprit_monitor_pipeline.joblib' % MODEL_SAVED_PATH
    y_predict = predict_result(pipeline_path, test_frame)
    print('The predict result is %s, the actual value is %s' % (y_predict, test_frame['trip_distance(km)'].values))
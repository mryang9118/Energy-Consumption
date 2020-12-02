"""
@Project: Energy-Consumption   
@Description: predict the test data, and return the value
@Time:2020/12/1 16:02                      
 
"""
import os
import joblib
import pandas as pd

X_COLUMN_NAMES = ['power(kW)', 'quantity(kWh)', 'tire_type', 'city',
                  'motor_way', 'country_roads', 'driving_style',
                  'consumption(kWh/100km)', 'A/C', 'park_heating', 'avg_speed(km/h)']


def predict(pipeline_path, test_data_frame):
    """
    :param pipeline_path: the pipeline file path
    :type pipeline_path: str
    :param test_data_frame: input data
    :type test_data_frame: DataFrame of shape [n_samples, n_features]
    :return: predict value of input data, and the hyperparameters of the trained model
    :rtype: dict{'predict': array, 'hyperparameters': dict}
    """
    if not os.path.exists(pipeline_path):
        print("Not found the file, please check the pipeline file path!")
        return
    pipeline = joblib.load(pipeline_path)
    X_test = test_data_frame[X_COLUMN_NAMES]
    steps_object = pipeline.get_params()['steps']
    hyper_params = steps_object[len(steps_object) - 1][1].get_parameters()
    return {'predict': pipeline.predict(X_test), 'hyperparameters': hyper_params}


if __name__ == '__main__':
    input_array = [['Volkswagen', 'E-Golf 300', '06.01.2019', 28, 100, 10.2, 'Summer tires', 1,	0,	1, 'Moderate',
                   24.6, 1, 1, 29]]
    header = ['manufacturer', 'version', 'fuel_date', 'trip_distance(km)', 'power(kW)', 'quantity(kWh)', 'tire_type',
              'city', 'motor_way', 'country_roads',	'driving_style', 'consumption(kWh/100km)',
              'A/C', 'park_heating', 'avg_speed(km/h)']
    MODEL_SAVED_PATH = '../output'
    test_frame = pd.DataFrame(data=input_array, columns=header)
    pipeline_path = '%s/mix_e-golf_tesla_pipeline.joblib' % MODEL_SAVED_PATH
    result = predict(pipeline_path, test_frame)
    print('The hyperparameters of the RF model: %s' % result['hyperparameters'])
    print('The predict result is %s, the actual value is %s' % (result['predict'], test_frame['trip_distance(km)'].values))
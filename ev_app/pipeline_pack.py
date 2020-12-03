"""
@Project: Energy-Consumption   
@Description: save the pipeline of preprocess, standardization, model
@Time:2020/12/1 15:12                      
 
"""
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml_models import ModelsFitter
from utils import *
import os

warnings.filterwarnings(action="ignore")


def save_pipeline(model_name, data_path, X_column_names, Y_column_names, require_encode_columns, output_path):
    train_data = clean_ev_data(data_path)
    X = train_data[X_column_names]
    y = train_data[Y_column_names]
    column_transformer = get_column_transformer(require_encode_columns)
    pipeline = make_pipeline(column_transformer, StandardScaler(), ModelsFitter(model_name))
    pipeline.fit(X, y)
    joblib.dump(pipeline, output_path, compress=3)


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
    steps_object = pipeline.get_params()['steps']
    hyper_params = steps_object[len(steps_object) - 1][1].get_parameters()
    X_column_names = steps_object[0][1]._feature_names_in.tolist()
    X_test = test_data_frame[X_column_names]
    return {'predict': pipeline.predict(X_test), 'hyperparameters': hyper_params}


if __name__ == '__main__':
    # save pipeline case
    input_path = "../data/all_all_mixed_data_11_12.csv"
    input_column_names = SPRIT_MONITOR_X_COLUMN_NAMES
    target_column_names = SPRIT_MONITOR_TARGET_COLUMN_NAME
    encode_column_names = SPRIT_MONITOR_REQUIRE_ENCODED_COLUMNS
    saved_path = '%s/mix_manufactures_pipeline.joblib' % MODEL_SAVED_PATH
    save_pipeline(RF, input_path, input_column_names, target_column_names, encode_column_names, saved_path)
    # load pipeline case
    input_array = [['Volkswagen', 'E-Golf 300', '06.01.2019', 28, 100, 10.2, 'Summer tires', 1,	0,	1, 'Moderate',
                   24.6, 1, 1, 29]]
    header = ['manufacturer', 'version', 'fuel_date', 'trip_distance(km)', 'power(kW)', 'quantity(kWh)', 'tire_type',
              'city', 'motor_way', 'country_roads',	'driving_style', 'consumption(kWh/100km)',
              'A/C', 'park_heating', 'avg_speed(km/h)']
    test_frame = pd.DataFrame(data=input_array, columns=header)
    pipeline_path = '%s/mix_manufactures_pipeline.joblib' % MODEL_SAVED_PATH
    result = predict(pipeline_path, test_frame)
    print('The hyperparameters of the RF model: %s' % result['hyperparameters'])
    print('The predict result is %s, the actual value is %s' % (result['predict'], test_frame['trip_distance(km)'].values))
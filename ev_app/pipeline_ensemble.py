"""
@Project: Energy-Consumption   
@Description: save the pipeline of preprocess, standardization, model
@Time:2020/12/1 15:12                      
 
"""
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml_models import ModelsFitter
from tensorflow.keras.models import load_model
from utils import *
import os

warnings.filterwarnings(action="ignore")


def save_pipeline(model_name, data_path, X_column_names, Y_column_names, require_encode_columns, output_path):
    train_data = clean_ev_data(data_path)
    X = train_data[X_column_names]
    y = train_data[Y_column_names]
    pipeline = Pipeline([
        ('preprocess', get_column_transformer(require_encode_columns)),
        ('scaler', StandardScaler()),
        ('estimator', ModelsFitter(model_name))
    ])
    pipeline.fit(X, y)
    __pickle_pipeline(model_name, pipeline, output_path)


def __pickle_pipeline(model_name, pipeline, output_path):
    if model_name == DEEP_MLP:
        pipeline.named_steps['estimator'].save_model()
        pipeline.named_steps['estimator'].model = None
    joblib.dump(pipeline, output_path, compress=3)


def predict(model_name, pipeline_path, test_data_frame):
    """
    :param model_name: choose which model
    :type model_name: str
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
    if model_name == DEEP_MLP:
        pipeline.named_steps['estimator'].model = load_model('%s/%s_model' % (MODEL_SAVED_PATH, str(DEEP_MLP).lower()))
    steps_object = pipeline.get_params()['steps']
    hyper_params = steps_object[len(steps_object) - 1][1].get_parameters()
    X_column_names = steps_object[0][1]._feature_names_in.tolist()
    X_test = test_data_frame[X_column_names]
    return {'predict': pipeline.predict(X_test), 'hyperparameters': hyper_params}


if __name__ == '__main__':
    # save pipeline case
    input_path = "../data/mix_e-golf_tesla.csv"
    input_column_names = SPRIT_MONITOR_X_COLUMN_NAMES
    target_column_names = SPRIT_MONITOR_TARGET_COLUMN_NAME
    encode_column_names = SPRIT_MONITOR_REQUIRE_ENCODED_COLUMNS
    saved_path = '%s/mix_e-golf_tesla_deep_mlp.joblib' % MODEL_SAVED_PATH
    save_pipeline(DEEP_MLP, input_path, input_column_names, target_column_names, encode_column_names, saved_path)
    # load pipeline case
    input_array = [['Volkswagen', 'E-Golf 300', '06.01.2019', 28, 100, 10.2, 'Summer tires', 1,	0,	1, 'Moderate',
                   24.6, 1, 1, 29]]
    header = ['manufacturer', 'version', 'fuel_date', 'trip_distance(km)', 'power(kW)', 'quantity(kWh)', 'tire_type',
              'city', 'motor_way', 'country_roads',	'driving_style', 'consumption(kWh/100km)',
              'A/C', 'park_heating', 'avg_speed(km/h)']
    test_frame = pd.DataFrame(data=input_array, columns=header)
    pipeline_path = '%s/mix_e-golf_tesla_deep_mlp.joblib' % MODEL_SAVED_PATH
    result = predict(DEEP_MLP, pipeline_path, test_frame)
    print('The hyperparameters of the RF model: %s' % result['hyperparameters'])
    print('The predict result is %s, the actual value is %s' % (result['predict'], test_frame['trip_distance(km)'].values))
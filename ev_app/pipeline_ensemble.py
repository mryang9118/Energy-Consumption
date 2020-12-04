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


def save_pipeline(model_name, data_path, sep=",",
                  X_column_names=SPRIT_MONITOR_X_COLUMN_NAMES,
                  Y_column_names=SPRIT_MONITOR_TARGET_COLUMN_NAME,
                  require_encode_columns=SPRIT_MONITOR_REQUIRE_ENCODED_COLUMNS,
                  output_path=MODEL_SAVED_PATH):
    """
    :param model_name: selective model name
    :type model_name: str
    :param data_path: the path of training data, only support csv format now
    :type data_path: str
    :param sep: the separator of input csv file
    :type sep: str
    :param X_column_names: input data header
    :type X_column_names: array
    :param Y_column_names: target column name
    :type Y_column_names: array
    :param require_encode_columns: require encode columns
    :type require_encode_columns: array
    :param output_path: output folder path for the pipeline
    :type output_path: str
    """
    train_data = clean_ev_data(data_path, sep)
    X = train_data[X_column_names]
    y = train_data[Y_column_names]
    pipeline = Pipeline([
        (PREPROCESS, get_column_transformer(require_encode_columns)),
        (STANDARD_SCALAR, StandardScaler()),
        (ESTIMATOR, ModelsFitter(model_name))
    ])
    pipeline.fit(X, y)
    __pickle_pipeline(model_name, pipeline, output_path)


def __pickle_pipeline(model_name, pipeline, output_path):
    if model_name == DEEP_MLP:
        pipeline.named_steps[ESTIMATOR].save_model()
        pipeline.named_steps[ESTIMATOR].model = None
    joblib.dump(pipeline, '%s/%s_%s' % (output_path, model_name, PIPELINE_FILE_SUFFIX), compress=3)


def predict(model_name, pipeline_folder_path, test_data_frame):
    """
    :param model_name: choose which model
    :type model_name: str
    :param pipeline_folder_path: the pipeline folder path
    :type pipeline_folder_path: str
    :param test_data_frame: input data
    :type test_data_frame: DataFrame of shape [n_samples, n_features]
    :return: predict value of input data, and the hyperparameters of the trained model
    :rtype: dict{'predict': array, 'hyperparameters': dict}
    """
    pipeline_file = '%s/%s_%s' % (pipeline_folder_path, model_name, PIPELINE_FILE_SUFFIX)
    if not os.path.exists(pipeline_file):
        print("Not found the file, please check the pipeline file path!")
        return
    pipeline = joblib.load(pipeline_file)
    if model_name == DEEP_MLP:
        pipeline.named_steps[ESTIMATOR].model = load_model('%s/%s_%s' % (MODEL_SAVED_PATH, str(DEEP_MLP).lower(),
                                                                         MODEL_SUFFIX))
    steps_object = pipeline.get_params()['steps']
    hyper_params = steps_object[len(steps_object) - 1][1].get_parameters()
    X_column_names = steps_object[0][1]._feature_names_in.tolist()
    X_test = test_data_frame[X_column_names]
    return {PREDICT: pipeline.predict(X_test), HYPER_PARAMETERS: hyper_params}


if __name__ == '__main__':
    # save pipeline to disk
    selected_model = DEEP_MLP
    input_path = "../data/mix_e-golf_tesla.csv"
    save_pipeline(selected_model, input_path)
    # prepare test data
    input_array = [['Volkswagen', 'E-Golf 300', '06.01.2019', 28, 100, 10.2, 'Summer tires', 1,	0,	1, 'Moderate',
                   24.6, 1, 1, 29]]
    test_frame = pd.DataFrame(data=input_array, columns=SPRIT_MONITOR_HEADER)
    result = predict(selected_model, MODEL_SAVED_PATH, test_frame)
    print('The hyperparameters of the %s model: %s' % (selected_model, result[HYPER_PARAMETERS]))
    print('The predict result is %s, the actual value is %s' % (result[PREDICT],
                                                                test_frame[SPRIT_MONITOR_TARGET_COLUMN_NAME].values))
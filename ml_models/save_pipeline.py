"""
@Project: Energy-Consumption   
@Description: save the pipeline of preprocess, standardization, model
@Time:2020/12/1 15:12                      
 
"""
import joblib
from sklearn.pipeline import make_pipeline
from sprit_monitor.sprit_monitor_preprocess import *
from sklearn.preprocessing import StandardScaler
from ml_models.preprocess import *
from utils.constants import *
from ml_models.parse_models import ModelsFitter


warnings.filterwarnings(action="ignore")


def save_model(model_name, data_path, X_column_names, Y_column_names, require_encode_columns, output_path):
    train_data = clean_ev_data(data_path)
    X = train_data[X_column_names]
    y = train_data[Y_column_names]
    column_transformer = get_column_transformer(require_encode_columns)
    pipe_line = make_pipeline(column_transformer, StandardScaler(), ModelsFitter(model_name))
    pipe_line.fit(X, y)
    joblib.dump(pipe_line, output_path, compress=3)


if __name__ == '__main__':
    input_path = "../data/mix_e-golf_tesla.csv"
    input_column_names = X_COLUMN_NAMES
    target_column_names = Y_COLUMN_NAME
    encode_column_names = REQUIRE_ENCODED_COLUMNS
    saved_path = '%s/mix_e-golf_tesla_pipeline.joblib' % MODEL_SAVED_PATH
    save_model(RF, input_path, input_column_names, target_column_names, encode_column_names, saved_path)

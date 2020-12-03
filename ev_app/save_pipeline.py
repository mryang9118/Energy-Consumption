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


warnings.filterwarnings(action="ignore")


def save_model(model_name, data_path, X_column_names, Y_column_names, require_encode_columns, output_path):
    train_data = clean_ev_data(data_path)
    X = train_data[X_column_names]
    y = train_data[Y_column_names]
    column_transformer = get_column_transformer(require_encode_columns)
    pipeline = make_pipeline(column_transformer, StandardScaler(), ModelsFitter(model_name))
    pipeline.fit(X, y)
    joblib.dump(pipeline, output_path, compress=3)


if __name__ == '__main__':
    input_path = "../data/all_all_mixed_data_11_12.csv"
    input_column_names = SPRIT_MONITOR_X_COLUMN_NAMES
    target_column_names = SPRIT_MONITOR_TARGET_COLUMN_NAME
    encode_column_names = SPRIT_MONITOR_REQUIRE_ENCODED_COLUMNS
    saved_path = '%s/mix_manufactures_pipeline.joblib' % MODEL_SAVED_PATH
    save_model(RF, input_path, input_column_names, target_column_names, encode_column_names, saved_path)

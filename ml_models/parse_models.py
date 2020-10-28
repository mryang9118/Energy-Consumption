"""
@Project: Energy-Consumption
@Description: support deep MLP, RF... algorithm, read the config from ml_models/models_config.xml
@Time:2020/9/16 18:07

"""
import time
from utils.xml_reader import *
from utils.constants import *
from ml_models.deep_mlp_models import DeepMLPModel
from ml_models.machine_learning_models import *
from ml_models.evaluate_util import *
import joblib

MODEL_CONFIG_PATH = "../config/models_config.xml"


class ModelsFitter(object):
    def __init__(self, model, x_matrix, y_matrix):
        self.model = model
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.model_used = None

    def process(self):
        print('------------- Start %s' % time.strftime('%X %x %Z') + ' -------------')
        layers_list, params_dict = read_xml_config(MODEL_CONFIG_PATH, self.model)
        if self.model == DEEP_MLP:
            self.model_used = DeepMLPModel(layers_list, len(self.x_matrix[0]))
            self.model_used.compile(optimizer=params_dict.get('optimizer'), loss=params_dict.get('loss'),
                                    metrics=params_dict.get('metrics'))
            self.model_used.fit(self.x_matrix, self.y_matrix, batch_size=params_dict.get('batch_size'),
                                epochs=params_dict.get('epochs'), verbose=0)
            evaluate_deep_mlp(self.model_used, self.x_matrix, self.y_matrix, params_dict.get('metrics'))
        elif self.model == RF:
            self.model_used = random_forest_regress_model(params_dict.get('n_estimators'), params_dict.get('criterion'))
            self.model_used.fit(self.x_matrix, self.y_matrix)
            evaluate_general_model(self.model_used, x=self.x_matrix, y=self.y_matrix,
                                   scoring=params_dict.get('scoring_methods'))
        print('------------- End %s' % time.strftime('%X %x %Z') + ' -------------')

    def get_model(self):
        return self.model_used

    def save_model(self):
        if self.model == DEEP_MLP:
            # sub model should assign parameter save_format
            self.model_used.save('%s/%s_model' % (MODEL_SAVED_PATH, str(self.model).lower()), save_format="tf")
        else:
            # apply to any model from scikit-learn
            joblib.dump(self.model_used, '%s/%s_model.joblib' % (MODEL_SAVED_PATH, str(self.model).lower()), compress=3)
        print("---Save %s model to %s_model.* file.---" % (self.model, str(self.model).lower()))

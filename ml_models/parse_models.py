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
from ml_models.visualize_result import *
import joblib

MODEL_CONFIG_PATH = "../config/models_config.xml"


class ModelsFitter(object):
    def __init__(self, model_name, x_matrix, y_matrix):
        self.model_name = model_name
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.model = None

    def process(self):
        print('------------- Start %s' % time.strftime('%X %x %Z') + ' -------------')
        layers_list, params_dict = read_xml_config(MODEL_CONFIG_PATH, self.model_name)
        if self.model_name == DEEP_MLP:
            self.model = DeepMLPModel(layers_list, len(self.x_matrix[0]))
            self.model.compile(optimizer=params_dict.get('optimizer'), loss=params_dict.get('loss'),
                               metrics=params_dict.get('metrics'))
            self.model.fit(self.x_matrix, self.y_matrix, batch_size=params_dict.get('batch_size'),
                           epochs=params_dict.get('epochs'), verbose=0)
            evaluate_deep_mlp(self.model, self.x_matrix, self.y_matrix, params_dict.get('metrics'))
        elif self.model_name == RF:
            self.model = random_forest_regress_model(params_dict.get('n_estimators'), params_dict.get('criterion'))
            self.model.fit(self.x_matrix, self.y_matrix)
            evaluate_general_model(self.model, x=self.x_matrix, y=self.y_matrix,
                                   scoring=params_dict.get('scoring_methods'))
        print('------------- End %s' % time.strftime('%X %x %Z') + ' -------------')

    def get_model_name(self):
        return self.model_name

    def get_model(self):
        return self.model

    def save_model(self):
        if self.model_name == DEEP_MLP:
            # sub model should assign parameter save_format
            self.model.save('%s/%s_model' % (MODEL_SAVED_PATH, str(self.model_name).lower()), save_format="tf")
        else:
            # apply to any model from scikit-learn
            joblib.dump(self.model, '%s/%s_model.joblib' % (MODEL_SAVED_PATH, str(self.model_name).lower()), compress=3)
        print("---Save %s model to %s_model.* file.---" % (self.model_name, str(self.model_name).lower()))

    def plot_learning_curve(self, cv=10):
        if self.model_name == DEEP_MLP:
            print('Not support Deep MLP model yet.')
            return
        sizes, training_scores, testing_scores = learning_curve(self.model, self.x_matrix, self.y_matrix, cv=cv,
                                                                train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=1)
        plot_curve(self.model_name, sizes, training_scores, testing_scores)

    def calculate_feature_importance(self):
        if self.model_name == RF:
            importance = self.model.feature_importances_
            plot_feature_importance(importance)
        else:
            print('Not support this model yet.')

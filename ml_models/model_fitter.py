"""
@Project: Energy-Consumption
@Description: support deep MLP, RF... algorithm, read the config from config/models_config.xml
@Time:2020/9/16 18:07

"""
import time
from sklearn.model_selection import learning_curve
from ml_models.machine_learning_models import random_forest_regress_model
from ml_models.deep_mlp_models import DeepMLPModel
from utils.evaluate_util import evaluate_deep_mlp, evaluate_general_model
from utils.visualize_result import plot_curve, plot_feature_importance
from utils.xml_reader import read_xml_config
from utils import *
import joblib


MODEL_CONFIG_PATH = "../config/models_config.xml"


class ModelsFitter(object):
    def __init__(self, model_name, x_matrix=None, y_matrix=None):
        self.model_name = model_name
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.model = None
        self.params_dict = None

    def fit(self, x_matrix, y_matrix):
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.__fit_model()

    def __fit_model(self):
        print('------------- Start fitting model %s' % time.strftime('%X %x %Z') + ' -------------')
        layers_list, params_dict = read_xml_config(MODEL_CONFIG_PATH, self.model_name)
        self.params_dict = params_dict
        if self.model_name == DEEP_MLP:
            self.model = DeepMLPModel(layers_list, len(self.x_matrix[0]))
            self.model.compile(optimizer=params_dict.get(OPTIMIZER), loss=params_dict.get(LOSS),
                               metrics=params_dict.get(METRICS))
            self.model.fit(self.x_matrix, self.y_matrix, batch_size=params_dict.get(BATCH_SIZE),
                           epochs=params_dict.get(EPOCHS), verbose=0)
        elif self.model_name == RF:
            self.model = random_forest_regress_model(params_dict.get(N_ESTIMATORS), params_dict.get(CRITERION))
            self.model.fit(self.x_matrix, self.y_matrix)
        print('------------- End fitting model %s' % time.strftime('%X %x %Z') + ' -------------')
        return self.model

    def predict(self, x_matrix):
        return self.model.predict(x_matrix)

    def evaluate_model(self):
        scores_dict = {}
        if self.model_name == DEEP_MLP:
            scores_dict = evaluate_deep_mlp(self.model, self.x_matrix, self.y_matrix, self.params_dict.get(METRICS))
        elif self.model_name == RF:
            scores_dict = evaluate_general_model(self.model, x=self.x_matrix, y=self.y_matrix,
                                                 scoring=self.params_dict.get(SCORING_METHODS))
        return scores_dict

    def get_model_name(self):
        return self.model_name

    def get_model(self):
        return self.model

    def get_parameters(self):
        return self.params_dict

    def save_model(self, output_path=MODEL_SAVED_PATH):
        if self.model_name == DEEP_MLP:
            # sub model should assign parameter save_format
            self.model.save('%s/%s_%s' % (output_path, str(self.model_name).lower(),
                                          MODEL_SUFFIX), save_format="tf")
        else:
            # apply to any model from scikit-learn
            joblib.dump(self.model, '%s/%s_%s' % (output_path, str(self.model_name).lower(),
                                                  MODEL_SUFFIX), compress=3)
        print("---Save %s model to %s_model.* file.---" % (self.model_name, str(self.model_name).lower()))

    def plot_learning_curve(self, cv=10):
        if self.model_name == DEEP_MLP:
            print('Not support to plot Deep MLP model learning curve yet.')
            return
        sizes, training_scores, testing_scores = learning_curve(self.model, self.x_matrix, self.y_matrix, cv=cv,
                                                                train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=1)
        plot_curve(self.model_name, sizes, training_scores, testing_scores)

    def calculate_feature_importance(self, feature_names):
        if self.model_name == RF:
            importance = self.model.feature_importances_
            plot_feature_importance(importance, feature_names)
        else:
            print('Not support to evaluate %s model feature importance yet.' % self.model_name)

"""
@Project: Energy-Consumption   
@Description: Load different tuned models based on parameters
@Time:2020/10/28 16:04                      
 
"""
from utils.constants import *
import joblib
from tensorflow.keras.models import load_model


class TunedModelLoader(object):

    def __init__(self, model):
        self.model_name = model
        if self.model_name == DEEP_MLP:
            self.model_used = load_model('%s/%s_model' % (MODEL_SAVED_PATH, str(DEEP_MLP).lower()))
        else:
            self.model_used = joblib.load('%s/%s_model.joblib' % (MODEL_SAVED_PATH, str(self.model_name).lower()))

    def load_model(self):
        return self.model_used

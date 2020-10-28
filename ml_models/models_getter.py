"""
@Project: Energy-Consumption   
@Description: get tuned model or real time fitted model
@Time:2020/10/28 16:39                      
 
"""
from ml_models.load_models import TunedModelLoader
from ml_models.parse_models import ModelsFitter


def get_model(model_name, tuned=True, save_model=True, x_matrix=None, y_matrix=None):
    """
    :param model_name:
    :type str:
    :param tuned:
    :type bool: tuned=True load the existing model to memory directly, tuned=False means tuning the model real time
    :param save_model:
    :type bool: this parameter works when tuned=False, meaning save the tuned model to disk or not
    :param x_matrix:
    :type array:
    :param y_matrix:
    :type array:
    :return:
    :rtype:
    """
    if tuned:
        model_loader = TunedModelLoader(model_name)
        model = model_loader.load_model()
    else:
        getter = ModelsFitter(model_name, x_matrix, y_matrix)
        getter.process()
        model = getter.get_model()
        if save_model:
            getter.save_model()
    return model

"""
@Project: Energy-Consumption   
@Description: read configuration of ml/dl models        
@Time:2020/9/17 09:48
 
"""
import xml.dom.minidom
import re
from utils import *


def read_xml_config(xml_path, model_name):
    dom = xml.dom.minidom.parse(xml_path)
    model_root = dom.documentElement.getElementsByTagName(model_name)
    if model_root:
        if model_name == DEEP_MLP:
            return get_deep_mlp_config(model_root)
        if model_name == RF:
            return get_rf_config(model_root)


def get_deep_mlp_config(model_root):
    layers_list = []
    hyperparam_dict = {}
    for model_element in model_root:
        denses_root = model_element.getElementsByTagName('Denses')
        for dense_element in denses_root:
            attributes = dense_element.getElementsByTagName(DENSE)
            for attr in attributes:
                layer_dict = {}
                units = attr.getAttribute(UNITS)
                kernel_initializer = attr.getAttribute(KERNEL_INITIALIZER)
                activation = attr.getAttribute(ACTIVATION)
                input_dim = attr.getAttribute(INPUT_DIM)
                layer_dict[UNITS] = int(units)
                layer_dict[KERNEL_INITIALIZER] = kernel_initializer
                layer_dict[ACTIVATION] = activation
                if input_dim:
                    layer_dict[INPUT_DIM] = input_dim
                layers_list.append(layer_dict)
        hyperparameters = model_element.getElementsByTagName(HYPER_PARAMETERS)
        for param in hyperparameters:
            optimizer = param.getAttribute(OPTIMIZER)
            loss = param.getAttribute(LOSS)
            batch_size = param.getAttribute(BATCH_SIZE)
            epochs = param.getAttribute(EPOCHS)
            metrics = param.getAttribute(METRICS)
            hyperparam_dict[OPTIMIZER] = optimizer
            hyperparam_dict[LOSS] = loss
            hyperparam_dict[BATCH_SIZE] = int(batch_size)
            hyperparam_dict[EPOCHS] = int(epochs)
            hyperparam_dict[METRICS] = re.split(COMMA, metrics)
    return layers_list, hyperparam_dict


def get_rf_config(model_root):
    hyperparam_dict = {}
    for model_element in model_root:
        hyperparameters = model_element.getElementsByTagName(HYPER_PARAMETERS)
        for param in hyperparameters:
            n_estimators = param.getAttribute(N_ESTIMATORS)
            criterion = param.getAttribute(CRITERION)
            n_splits = param.getAttribute(N_SPLITS)
            test_size = param.getAttribute(TEST_SIZE)
            scoring_methods = param.getAttribute(SCORING_METHODS)
            hyperparam_dict[N_ESTIMATORS] = int(n_estimators)
            hyperparam_dict[CRITERION] = criterion
            hyperparam_dict[N_SPLITS] = int(n_splits)
            hyperparam_dict[TEST_SIZE] = float(test_size)
            hyperparam_dict[SCORING_METHODS] = re.split(COMMA, scoring_methods)
    return list(), hyperparam_dict


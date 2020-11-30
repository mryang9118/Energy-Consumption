"""
@Project: Energy-Consumption   
@Description: read configuration of ml/dl models        
@Time:2020/9/17 09:48
 
"""
import xml.dom.minidom
import re


def read_xml_config(xml_path, model_name):
    dom = xml.dom.minidom.parse(xml_path)
    model_root = dom.documentElement.getElementsByTagName(model_name)
    if model_root:
        if model_name == 'DeepMLP':
            return get_deep_mlp_config(model_root)
        if model_name == 'RandomForest':
            return get_rf_config(model_root)


def get_deep_mlp_config(model_root):
    layers_list = []
    hyperparam_dict = {}
    for model_element in model_root:
        denses_root = model_element.getElementsByTagName('Denses')
        for dense_element in denses_root:
            attributes = dense_element.getElementsByTagName('Dense')
            for attr in attributes:
                layer_dict = {}
                units = attr.getAttribute('units')
                kernel_initializer = attr.getAttribute('kernel_initializer')
                activation = attr.getAttribute('activation')
                input_dim = attr.getAttribute('input_dim')
                layer_dict['units'] = int(units)
                layer_dict['kernel_initializer'] = kernel_initializer
                layer_dict['activation'] = activation
                if input_dim:
                    layer_dict['input_dim'] = input_dim
                layers_list.append(layer_dict)
        hyperparameters = model_element.getElementsByTagName('hyperparameters')
        for param in hyperparameters:
            optimizer = param.getAttribute('optimizer')
            loss = param.getAttribute('loss')
            batch_size = param.getAttribute('batch_size')
            epochs = param.getAttribute('epochs')
            metrics = param.getAttribute('metrics')
            hyperparam_dict['optimizer'] = optimizer
            hyperparam_dict['loss'] = loss
            hyperparam_dict['batch_size'] = int(batch_size)
            hyperparam_dict['epochs'] = int(epochs)
            hyperparam_dict['metrics'] = re.split(',', metrics)
    return layers_list, hyperparam_dict


def get_rf_config(model_root):
    hyperparam_dict = {}
    for model_element in model_root:
        hyperparameters = model_element.getElementsByTagName('hyperparameters')
        for param in hyperparameters:
            n_estimators = param.getAttribute('n_estimators')
            criterion = param.getAttribute('criterion')
            n_splits = param.getAttribute('n_splits')
            test_size = param.getAttribute('test_size')
            scoring_methods = param.getAttribute('scoring_methods')
            hyperparam_dict['n_estimators'] = int(n_estimators)
            hyperparam_dict['criterion'] = criterion
            hyperparam_dict['n_splits'] = int(n_splits)
            hyperparam_dict['test_size'] = float(test_size)
            hyperparam_dict['scoring_methods'] = re.split(',', scoring_methods)
    return list(), hyperparam_dict


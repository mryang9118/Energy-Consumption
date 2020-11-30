from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

refs = locals()


class DeepMLPModel(Model):

    def set_vars(self, dense_object_list, vars_list):
        global refs
        for i in range(len(vars_list)):
            refs[vars_list[i]] = dense_object_list[i]

    def get_config(self):
        return {"input_dim": self.input_dim, "layers": self.dense_object_list}

    def __init__(self, layers_list, input_dim):
        super(DeepMLPModel, self).__init__(name="custom_deep_mlp_model")
        self.input_dim = input_dim
        self.vars_list = []
        self.dense_object_list = []
        for single_layer in layers_list:
            index = layers_list.index(single_layer)

            dense = Dense(units=single_layer.get('units'),
                          kernel_initializer=single_layer.get('kernel_initializer'),
                          activation=single_layer.get('activation'), input_dim=input_dim) if \
                index == 0 else Dense(units=single_layer.get('units'),
                                      kernel_initializer=single_layer.get(
                                          'kernel_initializer'),
                                      activation=single_layer.get('activation'))
            self.dense_object_list.append(dense)
            self.vars_list.append('layer_%s' % (index + 1))
        self.set_vars(self.dense_object_list, self.vars_list)

    def call(self, inputs, training=None, mask=None):
        x = refs[self.vars_list[0]](inputs)
        for i in range(1, len(self.vars_list)):
            x = refs[self.vars_list[i]](x)
        return x

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class DeepMLPModel(Model):

    def get_config(self):
        return {"input_dim": self.input_dim}

    def __init__(self, input_dim):
        super(DeepMLPModel, self).__init__(name="custom_deep_mlp_model")
        self.input_dim = input_dim
        self.hidden_layer_1 = Dense(100, kernel_initializer='uniform', activation='relu', input_dim=input_dim)
        self.hidden_layer_2 = Dense(50, kernel_initializer='uniform', activation='relu')
        self.hidden_layer_3 = Dense(25, kernel_initializer='uniform', activation='relu')
        self.hidden_layer_4 = Dense(13, kernel_initializer='uniform', activation='relu')
        self.hidden_layer_5 = Dense(7, kernel_initializer='uniform', activation='relu')
        self.output_layer = Dense(1, kernel_initializer='uniform', activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.hidden_layer_1(inputs)
        x = self.hidden_layer_2(x)
        x = self.hidden_layer_3(x)
        x = self.hidden_layer_4(x)
        x = self.hidden_layer_5(x)
        return self.output_layer(x)



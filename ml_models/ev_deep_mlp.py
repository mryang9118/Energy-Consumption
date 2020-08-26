import tensorflow as tf


class DeepMLPModel(tf.keras.Model):

    def get_config(self):
        pass

    def __init__(self, input_dim):
        super(DeepMLPModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100, kernel_initializer='uniform', activation='relu', input_dim=input_dim)
        self.dense2 = tf.keras.layers.Dense(50, kernel_initializer='uniform', activation='relu')
        self.dense3 = tf.keras.layers.Dense(25, kernel_initializer='uniform', activation='relu')
        self.dense4 = tf.keras.layers.Dense(13, kernel_initializer='uniform', activation='relu')
        self.dense5 = tf.keras.layers.Dense(7, kernel_initializer='uniform', activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer='uniform', activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.output_layer(x)

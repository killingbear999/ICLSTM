import tensorflow as tf
from keras.layers import Input
from keras import Model

class residualLSTM(tf.keras.Model):
    def __init__(self, num_neurons, input_shape, output_shape):
        super(residualLSTM, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(num_neurons, activation='elu', return_sequences=True, kernel_constraint=tf.keras.constraints.NonNeg())
        self.dense1 = tf.keras.layers.Dense(input_shape, activation='elu', kernel_constraint=tf.keras.constraints.NonNeg())
        self.skip1 = tf.keras.layers.Add()

        self.lstm2 = tf.keras.layers.LSTM(num_neurons, activation='elu', return_sequences=True, kernel_constraint=tf.keras.constraints.NonNeg())
        self.dense2 = tf.keras.layers.Dense(input_shape, activation='elu', kernel_constraint=tf.keras.constraints.NonNeg())
        self.skip2 = tf.keras.layers.Add()

        self.dense3 = tf.keras.layers.Dense(output_shape, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())

    def call(self, input_tensor):
        x = self.lstm1(input_tensor)
        x = self.dense1(x)
        x = self.skip1([x, input_tensor])

        x = self.lstm2(x)
        x = self.dense2(x)
        x = self.skip2([x, input_tensor])

        output = self.dense3(x)
        return output

class residualRNN(tf.keras.Model):
    def __init__(self, num_neurons, input_shape, output_shape):
        super(residualRNN, self).__init__()
        self.rnn1 = tf.keras.layers.SimpleRNN(num_neurons, activation='elu', return_sequences=True, kernel_constraint=tf.keras.constraints.NonNeg())
        self.dense1 = tf.keras.layers.Dense(input_shape, activation='elu', kernel_constraint=tf.keras.constraints.NonNeg())
        self.skip1 = tf.keras.layers.Add()

        self.rnn2 = tf.keras.layers.SimpleRNN(num_neurons, activation='elu', return_sequences=True, kernel_constraint=tf.keras.constraints.NonNeg())
        self.dense2 = tf.keras.layers.Dense(input_shape, activation='elu', kernel_constraint=tf.keras.constraints.NonNeg())
        self.skip2 = tf.keras.layers.Add()

        self.dense3 = tf.keras.layers.Dense(output_shape, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())

    def call(self, input_tensor):
        x = self.rnn1(input_tensor)
        x = self.dense1(x)
        x = self.skip1([x, input_tensor])

        x = self.rnn2(x)
        x = self.dense2(x)
        x = self.skip2([x, input_tensor])

        output = self.dense3(x)
        return output

# input = Input(shape=(X_train.shape[1],X_train.shape[2]))
# x = residualLSTM(num_neurons=64, input_shape=8, output_shape=2)(input)
# # x = residualRNN(num_neurons=64, input_shape=8, output_shape=2)(input)
# model = Model(input, x)

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
# history = model.fit(X_train, y_train, epochs=500, batch_size=256, validation_split=0.25, verbose=2)
# model.summary()
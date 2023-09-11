import tensorflow as tf

class MyLSTMCell(tf.keras.layers.Layer):

    def __init__(self, units, input_shape_custom, **kwargs):
        self.units = units
        self.input_shape_custom = input_shape_custom
        self.state_size = [self.units, self.units]
        super(MyLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1] + self.units, 4 * self.units),
                                      initializer='uniform',
                                      name='kernel',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      trainable=True)
        self.built = True

    def call(self, inputs, states):
        h_tm1 = states[0]  # previous hidden state
        c_tm1 = states[1]  # previous cell state

        combined_inputs = tf.matmul(tf.concat([inputs, h_tm1], axis=-1), self.kernel)
        i, f, o, g = tf.split(combined_inputs, num_or_size_splits=4, axis=1)

        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        g = tf.nn.tanh(g)

        new_c = f * c_tm1 + i * g
        new_h = o * tf.nn.tanh(new_c)

        return new_h, [new_h, new_c]

    def get_config(self):
        config = super(MyLSTMCell, self).get_config()
        config.update({"units": self.units, "input_shape_custom": self.input_shape_custom})
        return config

# model = tf.keras.Sequential([
#     tf.keras.layers.RNN(MyLSTMCell(units=64, input_shape_custom=X_train.shape[2]),return_sequences=True),
#     tf.keras.layers.RNN(MyLSTMCell(units=64, input_shape_custom=64),return_sequences=True),
#     tf.keras.layers.Dense(2, activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())
# ])

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
# history = model.fit(X_train, y_train, epochs=1000, batch_size=256, validation_split=0.25, verbose=2)
# model.summary()
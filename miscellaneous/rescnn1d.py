import tensorflow as tf

class resCNN1D(tf.keras.Model):
  def __init__(self, num_neurons, kernel_size, len_sequence, num_feature, output_shape):
    super(resCNN1D, self).__init__()
    self.cnn_input = tf.keras.layers.Conv1D(filters=num_neurons, kernel_size=1, padding='same', input_shape=(len_sequence,num_feature))
    self.cnn1 = tf.keras.layers.Conv1D(filters=num_neurons, kernel_size=kernel_size, padding='same', activation='relu', input_shape=(len_sequence,num_feature))
    self.skip1 = tf.keras.layers.Add()
    self.cnn2 = tf.keras.layers.Conv1D(filters=num_neurons, kernel_size=kernel_size, padding='same', activation='relu')
    self.skip2 = tf.keras.layers.Add()

    self.flatten1 = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(256,activation='relu')
    self.dense2 = tf.keras.layers.Dense(len_sequence * output_shape, activation='linear')
    self.reshape1 = tf.keras.layers.Reshape((len_sequence, output_shape))

  def call(self, input_tensor):
    input = self.cnn_input(input_tensor)
    x = self.cnn1(input_tensor)
    x = self.skip1([x, input])

    x = self.cnn2(x)
    x = self.skip2([x, input])

    x = self.flatten1(x)
    x = self.dense1(x)
    x = self.dense2(x)

    output = self.reshape1(x)
    return output

# model = resCNN1D(num_neurons=64, kernel_size=3, len_sequence=X_train.shape[1], num_feature=X_train.shape[2], output_shape=y_train.shape[2])
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
# history = model.fit(X_train, y_train, epochs=500, batch_size=256, validation_split=0.25, verbose=2)
# model.summary()

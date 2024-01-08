import tensorflow as tf

class CNN1D(tf.keras.Model):
  def __init__(self, num_neurons, kernel_size, len_sequence, num_feature, output_shape):
    super(CNN1D, self).__init__()
    self.cnn1 = tf.keras.layers.Conv1D(filters=num_neurons, kernel_size=kernel_size, activation='relu', input_shape=(len_sequence,num_feature))
    self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=2)
    
    self.cnn2 = tf.keras.layers.Conv1D(filters=num_neurons, kernel_size=kernel_size, activation='relu')
    self.maxpool2 = tf.keras.layers.MaxPooling1D(pool_size=2)
    
    self.flatten1 = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(128,activation='relu')
    self.dense2 = tf.keras.layers.Dense(len_sequence * output_shape, activation='linear')
    
    self.reshape1 = tf.keras.layers.Reshape((len_sequence, output_shape))

  def call(self, input_tensor):
    x = self.cnn1(input_tensor)
    x = self.maxpool1(x)

    x = self.cnn2(x)
    x = self.maxpool2(x)

    x = self.flatten1(x)
    x = self.dense1(x)
    x = self.dense2(x)

    output = self.reshape1(x)
    return output

# model = CNN1D(num_neurons=64, kernel_size=3, len_sequence=X_train.shape[1], num_feature=X_train.shape[2], output_shape=y_train.shape[2])
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
# history = model.fit(X_train, y_train, epochs=500, batch_size=256, validation_split=0.25, verbose=2)
# model.summary()

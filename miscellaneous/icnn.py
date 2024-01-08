import tensorflow as tf
from keras import backend as K
from keras.layers import Model, Dense, Input

class MyICNN(tf.keras.layers.Layer):

    def __init__(self, units, myInputShape, isFirstLayer, **kwargs):
        self.units = units
        self.state_size = self.units
        self.myInputShape = myInputShape
        self.isFirstLayer = isFirstLayer
        super(MyICNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.myInputShape, self.units),
                                  initializer='uniform',
                                  name='kernel',
                                  constraint=tf.keras.constraints.NonNeg(),
                                  trainable=True)
        self.W = self.add_weight(shape=(input_shape[0][-1], self.units),
                                  initializer='uniform',
                                  name='W',
                                  constraint=tf.keras.constraints.NonNeg(),
                                  trainable=True)
        self.built = True

    def call(self, inputs):
        x, input = inputs
        if self.isFirstLayer is False:
          h = K.dot(x, self.W) + K.dot(input, self.kernel)
        else: 
          h = K.dot(x, self.W)
        h = tf.nn.relu(h)
        return h

    def get_config(self):
        config = super(MyICNN, self).get_config()
        config.update({"units": self.units, "myInputShape": self.myInputShape, "isFirstLayer": self.isFirstLayer})
        return config

# myInputShape = X_train.shape[1]*X_train.shape[2]
# X_train_ICNN = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2])
# y_train_ICNN = y_train.reshape(-1, y_train.shape[1]*y_train.shape[2])
# input = Input(shape=(X_train.shape[1]*X_train.shape[2],))
# x = MyICNN(64, myInputShape, True)([input, input])
# x = MyICNN(64, myInputShape, False)([x, input])
# x = Dense(y_train.shape[1] * y_train.shape[2], activation='linear', kernel_constraint=tf.keras.constraints.NonNeg())(x)
# model = Model(input, x)

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
# history = model.fit(X_train_ICNN, y_train_ICNN, epochs=500, batch_size=256, validation_split=0.25, verbose=2)
# model.summary()
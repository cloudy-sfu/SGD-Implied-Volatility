import tensorflow as tf
from tensorflow.keras.layers import *
import pickle
import numpy as np


class Pricing(Layer):
    def __init__(self, rf_rate):
        super(Pricing, self).__init__()
        self.r = tf.convert_to_tensor(rf_rate, dtype=tf.float32)

    @staticmethod
    def norm_cdf(x):
        return .5 * (1. + tf.math.erf(x / tf.math.sqrt(2.)))

    def call(self, inputs, **kwargs):
        sigma_, constants = inputs[0], inputs[1]
        s_, k_, t_ = constants[:, 0], constants[:, 1], constants[:, 2]
        d1 = (tf.math.log(s_ / k_) + (self.r + .5 * tf.math.square(sigma_)) * t_) / (sigma_ * tf.math.sqrt(t_) + 1e-13)
        d2 = d1 - sigma_ * tf.math.sqrt(t_)
        return s_ * self.norm_cdf(d1) - k_ * tf.math.exp(-self.r * t_) * self.norm_cdf(d2)


input_1 = Input((3,))  # ANN, Reference: risks, Shuaiqiang Liu, et al.
l1 = Dense(400, activation="relu")(input_1)
l2 = Dense(400, activation="relu")(l1)
l3 = Dense(400, activation="relu")(l2)
l4 = Dense(400, activation="relu")(l3)
sigma = Dense(1, activation="relu")(l4)
p = Pricing(rf_rate=.02433)([sigma, input_1])

mdl = tf.keras.Model(input_1, p)
mdl.compile(optimizer="adam", loss="mse", metrics=["mae"])
stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=60)
save_best = tf.keras.callbacks.ModelCheckpoint('raw/dense-3.h5', monitor="val_loss", save_best_only=True)

with open("raw/options", "rb") as f:
    s, k, _, c, t = pickle.load(f)
X = np.vstack([s, k, t]).T
mdl.fit(X, c, epochs=1000, batch_size=1024, callbacks=[stop_early, save_best])


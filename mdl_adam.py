import pickle
import tensorflow as tf
import numpy as np

@tf.function
def norm_cdf(x):
    return .5 * (1. + tf.math.erf(x / tf.math.sqrt(2.)))

with open("raw/options", "rb") as f:
    s, k, r, c, t = pickle.load(f)
n_samples = s.shape[0]
n_steps = 2000

s = tf.convert_to_tensor(s, dtype=tf.float32)
k = tf.convert_to_tensor(k, dtype=tf.float32)
r = tf.convert_to_tensor(r, dtype=tf.float32)
c = tf.convert_to_tensor(c, dtype=tf.float32)
t = tf.convert_to_tensor(t, dtype=tf.float32)
adam = tf.keras.optimizers.Adam()
c_hat_np = np.zeros((n_samples, n_steps))

for init_sigma in np.linspace(.1, 1, 7):
    sigma = tf.Variable(initial_value=tf.ones([n_samples]) * init_sigma, dtype=tf.float32,
                        constraint=lambda z: tf.clip_by_value(z, 0, 1e6))
    for step in range(n_steps):
        with tf.GradientTape(persistent=True) as tape:
            d1 = (tf.math.log(s / k) + (r + .5 * tf.square(sigma)) * t) / (sigma * tf.math.sqrt(t) + 1e-13)
            d2 = d1 - sigma * tf.math.sqrt(t)
            c_hat = s * norm_cdf(d1) - k * tf.exp(-r * t) * norm_cdf(d2)
            h = tf.square(c_hat - c)
        adam.minimize(h, [sigma], tape=tape)
        c_hat_np[:, step] = c_hat.numpy()
    with open(f"raw/c-hat/adam-{n_steps}-{format(init_sigma, '.2f')}", "wb") as f:
        pickle.dump(c_hat_np, f)

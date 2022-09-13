import tensorflow as tf
import sys

inputs = tf.keras.layers.Input(shape=1, name="feature1", dtype=tf.float32)
outputs = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=[outputs])

save_as_type = sys.argv[1]
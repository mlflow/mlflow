import os

import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants


@pytest.fixture
def boston_housing_model(tmpdir):
    tf.reset_default_graph()
    saved_model_path = os.path.join(str(tmpdir), "saved_model")

    # model training
    boston_housing = keras.datasets.boston_housing
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    model.fit(train_data, train_labels, epochs=100, verbose=0)

    # model saving
    input_tensor_name = 'input'
    output_tensor_name = 'output'
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        {input_tensor_name: model.input}, {output_tensor_name: model.output})

    builder = saved_model_builder.SavedModelBuilder(saved_model_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    # Initialize global variables and the model
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        # Add the meta_graph and the variables to the builder
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature,
            },
            legacy_init_op=legacy_init_op)

        # save the graph
        builder.save()

    return {
        'path': saved_model_path,
        'meta_graph_tags': [tf.saved_model.tag_constants.SERVING],
        'signature_def_key': signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        'test_data': test_data,
        'input_tensor_name': input_tensor_name,
        'output_tensor_name': output_tensor_name,
    }

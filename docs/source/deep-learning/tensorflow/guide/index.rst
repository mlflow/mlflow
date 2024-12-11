Tensorflow within MLflow
=========================

In this guide we will walk you through how to use Tensorflow within MLflow. We will demonstrate
how to track your Tensorflow experiments and log your Tensorflow models to MLflow.

Autologging Tensorflow Experiments
-----------------------------------

.. attention::
    Autologging is only supported when you are using the ``model.fit()`` Keras API to train
    the model. Additionally only Tensorflow >= 2.3.0 is supported. If you are using an older version
    of Tensorflow or Tensorflow without Keras, please use manual logging.

MLflow can automatically log metrics and parameters from your Tensorflow training. To enable
autologging, simply run :py:func:`mlflow.tensorflow.autolog()` or :py:func:`mlflow.autolog()`.

.. code-block:: python

    import mlflow
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    mlflow.tensorflow.autolog()

    # Prepare data for a 2-class classification.
    data = np.random.uniform(size=[20, 28, 28, 3])
    label = np.random.randint(2, size=20)

    model = keras.Sequential(
        [
            keras.Input([28, 28, 3]),
            keras.layers.Conv2D(8, 2),
            keras.layers.MaxPool2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
            keras.layers.Softmax(),
        ]
    )

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(0.001),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    with mlflow.start_run():
        model.fit(data, label, batch_size=5, epochs=2)

What is Logged by Autologging?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, autologging logs the following to MLflow:

- The model summary as returned by ``model.summary()``.
- Training hyperparamers, e.g., batch size and epochs.
- Optimizer configs, e.g., optimizer name and learning rate.
- Dataset information.
- Training and validation metrics, including loss and any metrics specified in ``model.compile()``.
- Saved model after training completes in the format of TF saved model (compiled graph).

You can customize autologging behavior by passing arguments to :py:func:`mlflow.tensorflow.autolog()`,
for example if you don't want to log the dataset information, then you can run
``mlflow.tensorflow.autolog(log_dataset_info=False)``. Please refer to the API documentation
:py:func:`mlflow.tensorflow.autolog()` for full customization options.


Understanding Autologging
^^^^^^^^^^^^^^^^^^^^^^^^^

The way we autolog Tensorflow is by registering a custom callback to the Keras model via monkey patch.
Briefly we attach a MLflow callback to the Keras model that works similarly to normal Keras callbacks.
At training start, training parameters including epochs, batch_size, learning_rate and model information
such as model summary will be logged. In addition, the callback will be triggered per ``every_n_iter``
epochs to log the training metrics, and after the training finishes, the trained model will be saved to MLflow.


Logging to MLflow with Keras Callback
--------------------------------------

As discussed in the previous section, MLflow autologging for Tensorflow is simply using a Keras
callback. If you wish to log additional information that isn't provided by the base autologging
implementation via this default callback, you can write your own callback to log custom information.

Using the Predefined Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MLflow offers a predefined callback :py:class:`mlflow.tensorflow.MlflowCallback` that you can use or
extend to log information to MLflow. The callback function provides the same functionality as autologging
and is suitable for users willing to have a better control of the experiment. Using ``mlflow.tensorflow.MlflowCallback``
is the same as other Keras callbacks:

.. code-block:: python

    with mlflow.start_run():
        model.fit(
            data,
            label,
            batch_size=5,
            epochs=2,
            callbacks=[mlflow.tensorflow.MlflowCallback()],
        )

You can change the logging frequency in :py:class:`mlflow.tensorflow.MlflowCallback` by setting
``log_every_epoch`` and ``log_every_n_steps``, by default metrics are logged per epoch. Please refer to
the API documentation for more details.

Customizing MLflow Logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also write your own callback to log information to MLflow. To do that, you need to define
a class subclassing from `keras.callbacks.Callback <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback>`_,
which provides hooks at various stages of training and validation, e.g., ``on_epoch_end`` and
``on_train_end`` are called separately at the end of each epoch and when the training is finished.
You can then use the callback in ``model.fit()``. Here is a simple example for logging the training metrics
in log scale:

.. code-block::

    from tensorflow import keras
    import math
    import mlflow

    class MlflowCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            for k, v in logs.items():
                mlflow.log_metric(f"log_{k}", math.log(v), step=epoch)

At the conclusion of each epoch, the ``logs`` object will contain ``loss`` and ``metrics`` as defined
in ``model.compile()``. For full documentation of the Keras callback API, please
read `keras.callbacks.Callback <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback>`_.

Saving Your Tensorflow Model to MLflow
--------------------------------------

If you have turned on the autologging, your Tensorflow model will be automatically saved after the training
is done. If you prefer to explicitly save your model, you can instead manually call
:py:func:`mlflow.tensorflow.log_model()`. After saving, you can load back the model using
:py:func:`mlflow.tensorflow.load_model()`. The loaded model can be used for inference by calling
the ``predict()`` method.

.. code-block:: python

    import mlflow
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.Input([28, 28, 3]),
            keras.layers.Conv2D(8, 2),
            keras.layers.MaxPool2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
            keras.layers.Softmax(),
        ]
    )

    save_path = "model"
    with mlflow.start_run() as run:
        mlflow.tensorflow.log_model(model, "model")

    # Load back the model.
    loaded_model = mlflow.tensorflow.load_model(f"runs:/{run.info.run_id}/{save_path}")

    print(loaded_model.predict(tf.random.uniform([1, 28, 28, 3])))


Diving into Saving
^^^^^^^^^^^^^^^^^^

Under the hood of saving, we are converting the Tensorflow model into a pyfunc model, which is a generic
type of model in MLflow. The pyfunc model is saved to MLflow. You don't need to learn the basics of pyfunc
model to use Tensorflow flavor, but if you are interested, please refer to `MLflow pyfunc model <https://mlflow.org/docs/latest/models.html#how-to-load-and-score-python-function-models>`_.

Saving Format
~~~~~~~~~~~~~

By default, MLflow saves your Tensorflow model in the format of a TF saved model (compiled graph), which is
suitable for deployment across platforms. You can also save your model in other formats, i.e., ``h5`` and
``keras`` by setting the ``keras_model_kwargs`` parameter in :py:func:`mlflow.tensorflow.log_model()`. For
example, if you want to save your model in ``h5`` format (which only saves model weights instead of the
compiled graph) you can run:

.. code-block:: python

    import mlflow
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.Input([28, 28, 3]),
            keras.layers.Conv2D(8, 2),
            keras.layers.MaxPool2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
            keras.layers.Softmax(),
        ]
    )

    save_path = "model"
    with mlflow.start_run() as run:
        mlflow.tensorflow.log_model(
            model, "model", keras_model_kwargs={"save_format": "h5"}
        )

    # Load back the model.
    loaded_model = mlflow.tensorflow.load_model(f"runs:/{run.info.run_id}/{save_path}")

    print(loaded_model.predict(tf.random.uniform([1, 28, 28, 3])))

For difference between the formats, please refer to `Tensorflow Save and Load Guide <https://www.tensorflow.org/guide/keras/save_and_serialize>`_.
Please note that if you want to deploy your model, you will need to save your model in the TF saved model format.

Model Signature
~~~~~~~~~~~~~~~

A model signature is a description of a model's input and output. If you have enabled autologging and provided
a dataset, then the signature will be automatically inferred from the dataset. Otherwise, you need to provide
a signature in order to have the signature information viewable within the MLflow UI. A model signature will be
shown in the MLflow UI as follows:

.. figure:: ../../../_static/images/deep-learning/tensorflow/guide/tensorflow-model-signature.png
   :alt: Tensorflow Model Signature
   :width: 90%
   :align: center

To manually set the signature for your model, you can pass a ``signature`` parameter to
:py:func:`mlflow.tensorflow.log_model()`. You will need to set the input schema by specifying the ``dtype``
and ``shape`` of the input tensors, and wrap it with :py:func:`mlflow.types.TensorSpec`. For example,

.. code-block::

    import mlflow
    import tensorflow as tf
    import numpy as np

    from tensorflow import keras
    from mlflow.types import Schema, TensorSpec
    from mlflow.models import ModelSignature

    model = keras.Sequential([
        keras.Input([28, 28, 3]),
        keras.layers.Conv2D(8, 2),
        keras.layers.MaxPool2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(2),
        keras.layers.Softmax(),
    ])

    input_schema = Schema(
        [
            TensorSpec(np.dtype(np.float32), (-1, 28, 28, 3), "input"),
        ]
    )
    signature = ModelSignature(inputs=input_schema)

    with mlflow.start_run() as run:
        mlflow.tensorflow.log_model(model, "model", signature=signature)

    # Load back the model.
    loaded_model = mlflow.tensorflow.load_model(f"runs:/{run.info.run_id}/{save_path}")

    print(loaded_model.predict(tf.random.uniform([1, 28, 28, 3])))

Please note that a model signature is not necessary for loading a model. You can still load the model
and perform inferenece if you know the input format. However, it's a good practice to include the signature
for better model understanding.

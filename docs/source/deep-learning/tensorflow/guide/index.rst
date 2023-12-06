Tensorflow within MLflow
=========================

In this guide we will walk you through how to use Tensorflow within MLflow. We will demonstrate
how to track your Tensorflow experiments and log your Tensorflow models to MLflow.

Autologging Tensorflow Experiments
-----------------------------------

.. attention::
    Autologging is only supported for Tensorflow versions >= 2.3.0, and when you are using the
    ``model.fit()`` Keras API to train the model. If you are using an older version of Tensorflow
    or Tensorflow without Keras, please log to MLflow manually.

MLflow can automatically log metrics and parameters from your Tensorflow training runs. To enable
autologging, simply run :py:func:`mlflow.tensorflow.autolog()`
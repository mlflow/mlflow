MLflow Keras 3.0 Integration
============================

Introduction
------------

Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. 
It was developed with a focus on enabling fast experimentation.

Keras 3.0 (Keras Core) makes it possible to run Keras workflows on top of TensorFlow, JAX, and PyTorch. 
It also enables you to seamlessly integrate Keras components (like layers, models, or metrics) as part of 
low-level TensorFlow, JAX, and PyTorch workflows.

MLflow provides built-in support for Keras 3.0 workflows. It provides a callback that allows you to 
log parameters and metrics during model training. Model logging is not currently supported.

5 Minute Quick Start with MLflow + Keras 3.0
--------------------------------------------

To get a quick overview of how to use MLflow + Keras 3.0, please read the quickstart guide. It will walk
you through how to use the callback for tracking experiments, as well as how to customize it.

.. raw:: html

    <a href="quickstart/quickstart_keras_core.html" class="download-btn">View the Quickstart</a>

To download the Keras 3.0 tutorial notebook to run in your environment, click the link below:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/deep-learning/keras/quickstart/quickstart_keras_core.ipynb"
    class="notebook-download-btn">Download the Quickstart of MLflow Keras Integration</a><br>


.. toctree::
    :maxdepth: 1
    :hidden:

    quickstart/quickstart_keras_core.ipynb

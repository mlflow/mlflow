MLflow Tensorflow Integration
==============================

Introduction
------------

TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible
ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art
in ML and developers easily build and deploy ML-powered applications.

MLflow has built-in support (we call it MLflow Tensorflow flavor) for Tensorflow workflow, at a high level
in MLflow we provide a set of APIs for:

- **Simplified Experiment Tracking**: Log parameters, metrics, and models during model training.
- **Experiments Management**: Store your Tensorflow experiments in MLflow server, and you can view and share them from MLflow UI.
- **Effortless Deployment**: Deploy Tensorflow models with simple API calls, catering to a variety of production environments.

5 Minute Quick Start with the MLflow Tensorflow Flavor
-------------------------------------------------------

To get a quick overview of how to use the MLflow Tensorflow flavor, please read the quickstart guide. It
will walk you through the basics of tracking Tensorflow experiments.

.. raw:: html

    <a href="quickstart/quickstart_tensorflow.html" class="download-btn">View the Quickstart</a>

To download the tensorflow tutorial notebooks to run in your environment, click the respective links below:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/deep-learning/tensorflow/quickstart/quickstart_tensorflow.ipynb"
    class="notebook-download-btn">Download the Quickstart of MLflow Tensorflow Integration</a><br>

.. toctree::
    :maxdepth: 1
    :hidden:

    quickstart/quickstart_tensorflow.ipynb

`Developer Guide of Tensorflow with MLflow <guide/index.html>`_
----------------------------------------------------------------

To learn more about the nuances of the `tensorflow` flavor in MLflow, please read the developer guide. It will walk you
through the following topics:

.. raw:: html

    <a href="guide/index.html" class="download-btn">View the Developer Guide</a>

- **Autologging Tensorflow Experiments with MLflow**: How to left MLflow autolog Tensorflow experiments, and what
  metrics are logged.
- **Control MLflow Logging with Keras Callback**: For people who don't like autologging, we offer an option to log
  experiments to MLflow using a custom Keras callback.
- **Log Your Tensorflow Models with MLflow**: How to log your Tensorflow models with MLflow and how to load them back
  for inference.


.. toctree::
    :maxdepth: 2
    :hidden:

    guide/index.rst
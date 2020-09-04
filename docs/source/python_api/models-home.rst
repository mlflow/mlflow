:toc-description: APIs for saving machine learning models.
:toc-icon: /_static/icons/deploy.png

Models
========

In this section, you'll find functions and modules related to saving and managing models. For more information on MLflow
Models and common workflows, see `Mlflow Models <../models.html>`_.

Log a model
-----------

.. tabs::

    .. tab:: H20

        .. autofunction:: mlflow.h2o.log_model

    .. tab:: Keras

        .. autofunction:: mlflow.keras.log_model

    .. tab:: TensorFlow

        .. autofunction:: mlflow.tensorflow.log_model

    .. tab:: MLeap

        .. autofunction:: mlflow.mleap.log_model

    .. tab:: PyTorch

        .. autofunction:: mlflow.pytorch.log_model

    .. tab:: Scikit-learn

        .. autofunction:: mlflow.sklearn.log_model

    .. tab:: Spark MLlib

        .. autofunction:: mlflow.spark.log_model

    .. tab:: ONNX

        .. autofunction:: mlflow.onnx.log_model

    .. tab:: MXNet Gluon

        .. autofunction:: mlflow.gluon.log_model

    .. tab:: XGBoost

        .. autofunction:: mlflow.xgboost.log_model

    .. tab:: LightGBM

        .. autofunction:: mlflow.lightgbm.log_model

    .. tab:: Spacy

        .. autofunction:: mlflow.spacy.log_model

    .. tab:: Fastai

        .. autofunction:: mlflow.fastai.log_model

Other modules
-------------
.. toctree::
    :maxdepth: 1

    mlflow.models
    mlflow.pyfunc
:toc-description: Library-specific APIs and MlflowClient.
:toc-icon: /_static/icons/deploy.png

Tracking
=========

MLflow supports a variety of machine learning libraries, or "flavors". In this section, you'll find functions and
modules containing library-specific functionality for each of them, as well as `MlflowClient`, a lower-level class for
experiment and run management.

Autologging (experimental)
--------------------------

.. tabs::

    .. tab:: Keras

        .. autofunction:: mlflow.keras.autolog

    .. tab:: TensorFlow

        .. autofunction:: mlflow.tensorflow.autolog

    .. tab:: Scikit-learn

        .. autofunction:: mlflow.sklearn.autolog

    .. tab:: MXNet Gluon

        .. autofunction:: mlflow.gluon.autolog

    .. tab:: LightGBM

        .. autofunction:: mlflow.lightgbm.autolog

    .. tab:: Fastai

        .. autofunction:: mlflow.fastai.autolog

MLflow Client
--------------------

.. toctree::
    :maxdepth: 1

    mlflow.tracking

Flavors
---------------------

.. toctree::
    :maxdepth: 1

    mlflow.sklearn
    mlflow.spacy
    mlflow.spark
    mlflow.tensorflow
    mlflow.fastai
    mlflow.gluon
    mlflow.h2o
    mlflow.keras
    mlflow.lightgbm
    mlflow.mleap
    mlflow.xgboost
    mlflow.pytorch
    mlflow.onnx
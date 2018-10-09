"""
The ``mlflow.spacy`` module provides an API for logging and loading SpaCy
models. This module exports SpaCy models with the following flavors:

SpaCy (native) format
    This is the main flavor that can be loaded back into SpaCy.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os

import pandas as pd

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking


FLAVOR_NAME = "spacy"


def log_model(spacy_model, artifact_path, conda_env=None, **kwargs):
    """
    Log a SpaCy model as an MLflow artifact for the current run.

    :param spacy_model: SpaCy model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Path to a Conda environment file. If provided, this
                      defines the environment for the model. At minimum, it
                      should specify python, spacy, and mlflow with
                      appropriate versions.
    :param kwargs: kwargs to pass to ``spacy_model.to_disk()`` method.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.spacy,
              spacy_model=spacy_model, conda_env=conda_env, **kwargs)


def save_model(spacy_model, path, conda_env=None, mlflow_model=Model(), **kwargs):
    """
    Save a SpaCy model to a path on the local file system.

    :param spacy_model: SpaCy model to be saved.
                          input and produce a single output tensor.
    :param path: Local path where the model is to be saved.
    :param conda_env: Path to a Conda environment file. If provided, this
                      decribes the environment this model should be run in.
                      At minimum, it should specify python, spacy, and mlflow
                      with appropriate versions.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param kwargs: kwargs to pass to ``spacy_model.to_disk`` method.
    """
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise RuntimeError("Path '{}' already exists".format(path))
    os.makedirs(path)
    model_path = os.path.join(path, "model.pth")

    # Save spacy model
    spacy_model.to_disk(model_path, **kwargs)
    model_file = os.path.basename(model_path)

    mlflow_model.add_flavor(FLAVOR_NAME, model_data=model_file,
                            spacy_version=spacy.__version__)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.spacy",
                        data=model_file, env=conda_env)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def _load_model(path, **kwargs):
    import spacy

    mlflow_model_path = os.path.join(path, "MLmodel")
    if not os.path.exists(mlflow_model_path):
        raise RuntimeError("MLmodel is not found at '{}'".format(path))

    mlflow_model = Model.load(mlflow_model_path)

    if FLAVOR_NAME not in mlflow_model.flavors:
        raise ValueError("Could not find flavor '{}' amongst available flavors {}, "
                         "unable to load stored model"
                         .format(FLAVOR_NAME, list(mlflow_model.flavors.keys())))

    flavor = mlflow_model.flavors[FLAVOR_NAME]
    if spacy.__version__ != flavor["spacy_version"]:
        raise ValueError("Stored model version '{}' does not match "
                         "installed SpaCy version '{}'"
                         .format(flavor["spacy_version"], spacy.__version__))

    path = os.path.abspath(path)
    path = os.path.join(path, mlflow_model.flavors[FLAVOR_NAME]['model_data'])
    return spacy.load(path, **kwargs)


def load_model(path, run_id=None, **kwargs):
    """
    Load a SpaCy model from a local file (if ``run_id`` is ``None``) or a run.

    :param path: Local filesystem path or run-relative artifact path to the model saved
                 by :py:func:`mlflow.spacy.log_model`.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.
    :param kwargs: kwargs to pass to ``spacy.load`` method.
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)

    return _load_model(path, **kwargs)


def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return _SpaCyWrapper(_load_model(os.path.dirname(path), **kwargs))


class _SpaCyWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """
    def __init__(self, spacy_model):
        self.spacy_model = spacy_model

    def predict(self, data, mode='classification'):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data should be pandas.DataFrame")
        if mode != "classification":
            raise ValueError("{} is not a supported mode".format(mode))

        return data.ix[:, 0].apply(lambda text: self.spacy_model(text).cats)

"""
The ``mlflow.spacy`` module provides an API for logging and loading spaCy models.
This module exports spacy models with the following flavors:

spaCy (native) format
    This is the main flavor that can be loaded back into spaCy.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference, this
    flavor is created only if spaCy's model pipeline has at least one
    `TextCategorizer <https://spacy.io/api/textcategorizer>`_.
"""
import logging
import os
import sys
from typing import IO, Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "spacy"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("spacy")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    spacy_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save a spaCy model to a path on the local file system.

    :param spacy_model: spaCy model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    """
    import spacy

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException(
            "Unable to save MLflow model to {path} - path '{path}' "
            "already exists".format(path=path)
        )

    model_data_subpath = "model.spacy"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # Save spacy-model
    spacy_model.to_disk(path=model_data_path)

    # Save the pyfunc flavor if at least one text categorizer in spaCy pipeline
    if any(
        [
            isinstance(pipe_component[1], spacy.pipeline.TextCategorizer)
            for pipe_component in spacy_model.pipeline
        ]
    ):
        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.spacy",
            data=model_data_subpath,
            env=_CONDA_ENV_FILE_NAME,
        )
    else:
        _logger.warning(
            "Generating only the spacy flavor for the provided spacy model. This means the model "
            "can be loaded back via `mlflow.spacy.load_model`, but cannot be loaded back using "
            "pyfunc APIs like `mlflow.pyfunc.load_model` or via the `mlflow models` CLI commands. "
            "MLflow will only generate the pyfunc flavor for spacy models containing a pipeline "
            "component that is an instance of spacy.pipeline.TextCategorizer."
        )

    mlflow_model.add_flavor(FLAVOR_NAME, spacy_version=spacy.__version__, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_path, FLAVOR_NAME, fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    spacy_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs
):
    """
    Log a spaCy model as an MLflow artifact for the current run.

    :param spacy_model: spaCy model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to ``spacy.save_model`` method.
    """
    Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.spacy,
        registered_model_name=registered_model_name,
        spacy_model=spacy_model,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs
    )


def _load_model(path):
    import spacy

    path = os.path.abspath(path)
    return spacy.load(path)


class _SpacyModelWrapper:
    def __init__(self, spacy_model):
        self.spacy_model = spacy_model

    def predict(self, dataframe):
        """
        Only works for predicting using text categorizer.
        Not suitable for other pipeline components (e.g: parser)
        :param dataframe: pandas dataframe containing texts to be categorized
                          expected shape is (n_rows,1 column)
        :return: dataframe with predictions
        """
        if len(dataframe.columns) != 1:
            raise MlflowException("Shape of input dataframe must be (n_rows, 1column)")

        return pd.DataFrame(
            {"predictions": dataframe.iloc[:, 0].apply(lambda text: self.spacy_model(text).cats)}
        )


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``spacy`` flavor.
    """
    return _SpacyModelWrapper(_load_model(path))


def load_model(model_uri):
    """
    Load a spaCy model from a local file (if ``run_id`` is ``None``) or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :return: A spaCy loaded model
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    # Flavor configurations for models saved in MLflow version <= 0.8.0 may not contain a
    # `data` key; in this case, we assume the model artifact path to be `model.spacy`
    spacy_model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.spacy"))
    return _load_model(path=spacy_model_file_path)


@experimental
@autologging_integration(FLAVOR_NAME)
def auto_logger(
    log_models: Optional[bool] = True,
    disable: Optional[bool] = False,
    exclusive: Optional[bool] = False,
    disable_for_unsupported_versions: Optional[bool] = False,
    silent: Optional[bool] = False,
    run_name: Optional[str] = None,
    remove_config_values: List[str] = [],
    log_interval: Optional[int] = None,
    log_dataset_dir: Optional[str] = None,
    log_best_model: Optional[bool] = False
):
    """
    Autologging should be used as an entry point for spaCy to log training metrics, configurations and 
    models to MLflow. Autologging captures the following information:

    **Metrics** and **Parameters**
     - losses; scores; other_scores
     - training configs
    **Artifacts**
     - `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (Spacy model) on spcified checkpoints 
        and the final best model

    .. code-block
        :caption: Example config for spaCy

        [training.logger]
        @loggers = "spacy.MLFlow.v1"
        run_name = "monitor_spacy_training"
        remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
        log_interval = 100
        log_dataset_dir = "corpus"
        log_best_model = true

    :param log_models: If ``True``, trained models and checkpoints are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param disable: If ``True``, disables the spaCy autologging integration. If ``False``,
                    enables the spaCy autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      spaCy that have not been tested against this version of the MLflow client
                      or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during spaCy
                   autologging. If ``False``, show all events and warnings during spaCy
                   autologging.
    :param run_name: If provided, the run is started with the given name.
    :param remove_config_values: Parameters with these keys are removed from logging
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param log_interval: When provided, the pipeline object be logged into checkpoints directory as 
                        MLFlow models artifacts and the metrices will be logged at the given intervals.
    :param log_dataset_dir: When the path to the corpus/dataset directory is given, the dataset will
                            be logged into datasets directory as an artifact.
    :param log_best_model: If ``True``, the spaCy pipeline with maximum value for matric ``score`` 
                            will be logged into best-model directory as MLFlow spacy model.
    """

    import spacy
    import re

    def setup_logger(
        nlp: spacy.language.Language, stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        mlflow.start_run(run_name=run_name)
        config = nlp.config.interpolate()
        config_dot = spacy.util.dict_to_dot(config)
        for _field in remove_config_values:
            del config_dot[_field]
        pattern = re.compile('[\W]+')
        mlflow.log_params(
            {pattern.sub('.', _field): _value for _field, _value in config_dot.items()})

        def _log_metrics(metric: Dict[str, Any], parent: str = '', step: Optional[int] = None):
            for k, v in metric.items():
                if isinstance(v, (dict,)):
                    if parent:
                        _log_metrics(v, f"{parent}.{k}", step)
                    else:
                        _log_metrics(v, k, step)
                else:
                    if parent:
                        mlflow.log_metric(f"{parent}.{k}", v, step=step)
                    else:
                        mlflow.log_metric(k, v, step=step)

        if log_dataset_dir:
            mlflow.log_artifact(log_dataset_dir, "dataset")

        def log_step(info: Optional[Dict[str, Any]]):
            if info is not None and log_interval:
                if info["step"] % log_interval == 0 and info["step"] != 0:
                    _log_metrics({k: info.get(k) for k in (
                        "losses", "score", "other_scores")}, step=info.get("step"))
                    if log_models:
                        log_model(
                            nlp, f"models/checkpoints/epoch_{info['epoch']}_step_{info['step']}")
                        if log_best_model and info["score"] == max(info["checkpoints"])[0]:
                            log_model(nlp, f"models/best-model")

        def finalize() -> None:
            mlflow.end_run()

        return log_step, finalize

    return setup_logger

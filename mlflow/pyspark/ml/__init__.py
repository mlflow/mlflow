import json
from pkg_resources import resource_filename

import mlflow
from mlflow.utils import chunk_dict, truncate_dict
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    try_mlflow_log,
)

from mlflow.spark import FLAVOR_NAME


def _get_fully_qualified_class_name(instance):
    return instance.__class__.__module__ + "." + instance.__class__.__name__


def _read_log_model_allowlist():
    """
    Reads the module allowlist and returns it as a set of tuples.
    """
    file_path = resource_filename(__name__, "log_model_allowlist.txt")
    allowlist = set()
    with open(file_path) as f:
        for line in f:
            stripped = line.strip()
            is_blankline_or_comment = stripped == "" or stripped.startswith("#")
            if not is_blankline_or_comment:
                allowlist.add(stripped)
    return allowlist


_log_model_allowlist = _read_log_model_allowlist()


def _should_log_model(estimator, spark_model):
    class_name = _get_fully_qualified_class_name(estimator)
    if class_name in _log_model_allowlist:
        return True
    elif class_name == 'pyspark.ml.classification.OneVsRest':
        classifier = estimator.getClassifier()
        return _get_fully_qualified_class_name(classifier) in _log_model_allowlist
    elif class_name == 'pyspark.ml.classification.LogisticRegression':
        return spark_model.numClasses < 10
    else:
        return False


def _get_estimator_info_tags(estimator):
    """
    :return: A dictionary of MLflow run tag keys and values
             describing the specified estimator.
    """
    return {
        "estimator_name": estimator.__class__.__name__,
        "estimator_class": _get_fully_qualified_class_name(estimator),
    }


def _get_estimator_param_map(estimator):
    param_map = {param.name: estimator.getOrDefault(param)
                 for param in estimator.params if estimator.isDefined(param)}
    # TODO:
    #  handle special case: OneVsRest.classifier,CrossValidator.estimator,
    #  CrossValidator.estimatorParamMaps
    json_param_map = {name: json.dumps(param_map[name]) for name in param_map}
    return json_param_map


# NOTE: The current implementation doesn't guarantee thread-safety, but that's okay for now because:
# 1. We don't currently have any use cases for allow_children=True.
# 2. The list append & pop operations are thread-safe, so we will always clear the session stack
#    once all _SparkTrainingSessions exit.
class _SparkTrainingSession(object):
    _session_stack = []

    def __init__(self, clazz, allow_children=True):
        self.allow_children = allow_children
        self.clazz = clazz
        self._parent = None

    def __enter__(self):
        if len(_SparkTrainingSession._session_stack) > 0:
            self._parent = _SparkTrainingSession._session_stack[-1]
            self.allow_children = (
                _SparkTrainingSession._session_stack[-1].allow_children and self.allow_children
            )
        _SparkTrainingSession._session_stack.append(self)
        return self

    def __exit__(self, tp, val, traceback):
        _SparkTrainingSession._session_stack.pop()

    def should_log(self):
        """
        Returns True when at least one of the following conditions satisfies:

        1. This session is the root session.
        2. The parent session allows autologging and its class differs from this session's class.
        """
        return (self._parent is None) or (
            self._parent.allow_children and self._parent.clazz != self.clazz
        )


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging for pyspark ml estimators.

    :param log_models:
    :param disable:
    :param exclusive:
    :param disable_for_unsupported_versions:
    :param silent:
    :return:
    """
    from mlflow.utils.validation import (
        MAX_PARAMS_TAGS_PER_BATCH,
        MAX_PARAM_VAL_LENGTH,
        MAX_ENTITY_KEY_LENGTH,
    )
    from mlflow.spark import log_model
    from pyspark.ml.base import Estimator

    def _log_pretraining_metadata(estimator, params):

        if params and isinstance(params, dict):
            estimator = estimator.copy(params)

        # Chunk model parameters to avoid hitting the log_batch API limit
        for chunk in chunk_dict(
                _get_estimator_param_map(estimator),
                chunk_size=MAX_PARAMS_TAGS_PER_BATCH,
        ):
            truncated = truncate_dict(chunk, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)
            try_mlflow_log(mlflow.log_params, truncated)

        try_mlflow_log(mlflow.set_tags, _get_estimator_info_tags(estimator))

    def _log_posttraining_metadata(estimator, spark_model, params):
        if log_models and _should_log_model(estimator, spark_model):
            # TODO:
            #  1. check whitelist when log models
            #  2. support model signature ?
            try_mlflow_log(
                log_model,
                spark_model,
                artifact_path="model",
            )

    def fit_mlflow(original, self, params=None):
        _log_pretraining_metadata(self, params)
        spark_model = original(self, params)
        _log_posttraining_metadata(self, spark_model, params)
        return spark_model

    def patched_fit(original, self, params=None):
        with _SparkTrainingSession(clazz=self.__class__, allow_children=False) as t:
            if t.should_log() and not isinstance(params, (list, tuple)):
                return fit_mlflow(original, self, params)
            else:
                return original(self, params)

    safe_patch(
        FLAVOR_NAME, Estimator, 'fit', patched_fit, manage_run=True,
    )

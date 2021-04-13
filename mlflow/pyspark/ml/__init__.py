import logging
from pkg_resources import resource_filename

import mlflow
from mlflow.utils import chunk_dict, truncate_dict
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    try_mlflow_log,
)


_logger = logging.getLogger(__name__)


def _get_fully_qualified_class_name(instance):
    return instance.__class__.__module__ + "." + instance.__class__.__name__


def _read_log_model_allowlist_from_file(allowlist_file):
    allowlist = set()
    with open(allowlist_file) as f:
        for line in f:
            stripped = line.strip()
            is_blankline_or_comment = stripped == "" or stripped.startswith("#")
            if not is_blankline_or_comment:
                allowlist.add(stripped)
    return allowlist


def _read_log_model_allowlist(spark_session):
    """
    Reads the module allowlist and returns it as a set.
    """
    allowlist_file = spark_session.sparkContext._conf.get(
        "spark.databricks.mlflow.autolog.log_model.allowlistFile", None
    )
    builtin_allowlist_file = resource_filename(__name__, "log_model_allowlist.txt")
    if allowlist_file:
        try:
            return _read_log_model_allowlist_from_file(allowlist_file)
        except Exception:
            # fallback to built-in allowlist file
            _logger.warning(
                "Reading config spark.databricks.mlflow.autolog.log_model.allowlistFile "
                f"{allowlist_file} failed, fallback to built-in allowlist file"
            )
            return _read_log_model_allowlist_from_file(builtin_allowlist_file)
    else:
        return _read_log_model_allowlist_from_file(builtin_allowlist_file)


# When autolog called, `_log_model_allowlist` will be initialized.
_log_model_allowlist = None


def _get_warning_msg_for_skip_log_model(model):
    return (
        f"This model {model.uid} is not logged because it is not in "
        "allowlist or its nested models are not in allowlist."
    )


def _should_log_model(spark_model):
    # TODO: Handle PipelineModel/CrossValidatorModel/TrainValidationSplitModel
    class_name = _get_fully_qualified_class_name(spark_model)
    if class_name in _log_model_allowlist:
        if class_name == "pyspark.ml.classification.OneVsRestModel":
            return _should_log_model(spark_model.models[0])
        else:
            return True
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


def _get_instance_param_map(instance):
    from pyspark.ml.param import Params

    param_map = {
        param.name: instance.getOrDefault(param)
        for param in instance.params
        if instance.isDefined(param)
    }
    # TODO:
    #  handle special case: CrossValidator.estimatorParamMaps

    expanded_param_map = {}
    for k, v in param_map.items():
        if isinstance(v, Params):
            expanded_param_map[k] = v.uid
            internal_param_map = _get_instance_param_map(v)
            for ik, iv in internal_param_map.items():
                expanded_param_map[f"{v.uid}.{ik}"] = iv
        else:
            expanded_param_map[k] = v

    return expanded_param_map


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


def _get_warning_msg_for_fit_call_with_a_list_of_params(estimator):
    return (
        "Skip instrumentation when calling "
        + f"{_get_fully_qualified_class_name(estimator)}.fit with a list of params."
    )


@experimental
@autologging_integration(mlflow.spark.FLAVOR_NAME)
def autolog(
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging for pyspark ml estimators.
    This method is not threadsafe and assumes a
    `SparkSession
    <https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SparkSession>`_
    already exists with the
    `mlflow-spark JAR
    <http://mlflow.org/docs/latest/tracking.html#automatic-logging-from-spark-experimental>`_
    attached. It should be called on the Spark driver, not on the executors (i.e. do not call
    this method within a function parallelized by Spark). This API requires Spark 3.0 or above.

    **When is autologging performed?**
      Autologging is performed when you call ``Estimator.fit``

    **Logged information**
      **Parameters**
        - Parameters obtained by ``estimator.params``. If a param value is also an ``Estimator``,
          then params in the the wrapped estimator will also be logged, the nested param key
          will be `{estimator_uid}.{param_name}`

      **Tags**
        - An estimator class name (e.g. "LinearRegression").
        - A fully qualified estimator class name
          (e.g. "pyspark.ml.regression.LinearRegression").

      **Artifacts**
        - An MLflow Model with the :py:mod:`mlflow.spark` flavor containing a fitted estimator
          (logged by :py:func:`mlflow.spark.log_model()`).


    :param log_models: If ``True``, if trained models are in allowlist, logged as MLflow model
                       artifacts. If ``False``, trained models are not logged.
                       Note: the built-in allowlist excludes some models which model may be large
                       (including logistic regression model, tree/random forest/boosting tree Model
                       and so on), you can set the config
                       "spark.databricks.mlflow.autolog.log_model.allowlistFile" to customize the
                       allowlist file. If you want to ensure logging model far all cases, you
                       should set autolog argument log_models=False and explicitly call
                       ``mlflow.spark.log_model`` to log the model.
    :param disable: If ``True``, disables the scikit-learn autologging integration. If ``False``,
                    enables the pyspark ML autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      pyspark that have not been tested against this version of the MLflow
                      client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during pyspark ML
                   autologging. If ``False``, show all events and warnings during pyspark ML
                   autologging.
    """
    from mlflow.utils.validation import (
        MAX_PARAMS_TAGS_PER_BATCH,
        MAX_PARAM_VAL_LENGTH,
        MAX_ENTITY_KEY_LENGTH,
    )
    from mlflow.utils._spark_utils import _get_active_spark_session
    from pyspark.ml.base import Estimator

    global _log_model_allowlist

    if not _log_model_allowlist:
        _log_model_allowlist = _read_log_model_allowlist(_get_active_spark_session())

    def _log_pretraining_metadata(estimator, params):

        if params and isinstance(params, dict):
            estimator = estimator.copy(params)

        # Chunk model parameters to avoid hitting the log_batch API limit
        for chunk in chunk_dict(
            _get_instance_param_map(estimator), chunk_size=MAX_PARAMS_TAGS_PER_BATCH,
        ):
            truncated = truncate_dict(chunk, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)
            try_mlflow_log(mlflow.log_params, truncated)

        try_mlflow_log(mlflow.set_tags, _get_estimator_info_tags(estimator))

    def _log_posttraining_metadata(estimator, spark_model, params):
        if log_models:
            if _should_log_model(spark_model):
                # TODO: support model signature
                try_mlflow_log(
                    mlflow.spark.log_model, spark_model, artifact_path="model",
                )
            else:
                _logger.warning(_get_warning_msg_for_skip_log_model(spark_model))

    def fit_mlflow(original, self, *args, **kwargs):
        params = None
        if len(args) > 1:
            params = args[1]
        elif "params" in kwargs:
            params = kwargs["params"]

        if _get_fully_qualified_class_name(self).startswith("pyspark.ml.feature."):
            return original(self, *args, **kwargs)
        elif isinstance(params, (list, tuple)):
            # skip the case params is a list or tuple, this case it will call
            # fitMultiple and return a model iterator
            _logger.warning(_get_warning_msg_for_fit_call_with_a_list_of_params(self))
            return original(self, *args, **kwargs)
        else:
            estimator = self.copy(params)
            _log_pretraining_metadata(estimator, params)
            spark_model = original(self, *args, **kwargs)
            _log_posttraining_metadata(estimator, spark_model, params)
            return spark_model

    def patched_fit(original, self, *args, **kwargs):
        with _SparkTrainingSession(clazz=self.__class__, allow_children=False) as t:
            if t.should_log():
                return fit_mlflow(original, self, *args, **kwargs)
            else:
                return original(self, *args, **kwargs)

    safe_patch(
        mlflow.spark.FLAVOR_NAME, Estimator, "fit", patched_fit, manage_run=True,
    )

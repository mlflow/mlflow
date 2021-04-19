import logging
from pkg_resources import resource_filename

import mlflow
from mlflow.utils import _chunk_dict, _truncate_dict, _get_fully_qualified_class_name
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    _get_new_training_session_class,
    autologging_integration,
    safe_patch,
    try_mlflow_log,
)


_logger = logging.getLogger(__name__)
_SparkTrainingSession = _get_new_training_session_class()

AUTOLOGGING_INTEGRATION_NAME = "pyspark.ml"


def _read_log_model_allowlist_from_file(allowlist_file):
    allowlist = set()
    with open(allowlist_file) as f:
        for line in f:
            stripped = line.strip()
            is_blankline_or_comment = stripped == "" or stripped.startswith("#")
            if not is_blankline_or_comment:
                allowlist.add(stripped)
    return allowlist


def _read_log_model_allowlist():
    """
    Reads the module allowlist and returns it as a set.
    """
    from mlflow.utils._spark_utils import _get_active_spark_session

    builtin_allowlist_file = resource_filename(__name__, "log_model_allowlist.txt")
    spark_session = _get_active_spark_session()
    if not spark_session:
        _logger.info(
            "No SparkSession detected. Autologging will log pyspark.ml models contained "
            "in the default allowlist. To specify a custom allowlist, initialize a SparkSession "
            "prior to calling mlflow.pyspark.ml.autolog() and specify the path to your allowlist "
            "file via the spark.mlflow.pysparkml.autolog.logModelAllowlistFile conf."
        )
        return _read_log_model_allowlist_from_file(builtin_allowlist_file)

    allowlist_file = spark_session.sparkContext._conf.get(
        "spark.mlflow.pysparkml.autolog.logModelAllowlistFile", None
    )
    if allowlist_file:
        try:
            return _read_log_model_allowlist_from_file(allowlist_file)
        except Exception:
            # fallback to built-in allowlist file
            _logger.exception(
                "Reading from custom log_models allowlist file "
                + "%s failed, fallback to built-in allowlist file.",
                allowlist_file,
            )
            return _read_log_model_allowlist_from_file(builtin_allowlist_file)
    else:
        return _read_log_model_allowlist_from_file(builtin_allowlist_file)


# When autolog called, `_log_model_allowlist` will be initialized.
_log_model_allowlist = None


def _get_warning_msg_for_skip_log_model(model):
    return (
        f"Model {model.uid} will not be autologged because it is not allowlisted or or because "
        "one or more of its nested models are not allowlisted. Call mlflow.spark.log_model() "
        "to explicitly log the model, or specify a custom allowlist via the "
        "spark.mlflow.pysparkml.autolog.logModelAllowlistFile Spark conf "
        "(see mlflow.pyspark.ml.autolog docs for more info)."
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
    from pyspark.ml.tuning import CrossValidator, TrainValidationSplit

    param_map = {
        param.name: instance.getOrDefault(param)
        for param in instance.params
        if instance.isDefined(param)
    }
    expanded_param_map = {}
    for k, v in param_map.items():
        if isinstance(v, Params):
            # handle the case param value type inherits `pyspark.ml.param.Params`
            # e.g. param like `OneVsRest.classifier`/`CrossValidator.estimator`
            expanded_param_map[k] = v.uid
            internal_param_map = _get_instance_param_map(v)
            for ik, iv in internal_param_map.items():
                expanded_param_map[f"{v.uid}.{ik}"] = iv
        elif k in ["estimator", "estimatorParamMaps"] and isinstance(
            instance, (CrossValidator, TrainValidationSplit)
        ):
            # skip log estimator Param and its nested params because they will be
            # logged in nested runs.
            # TODO: Log `estimatorParamMaps` as JSON artifacts.
            pass
        else:
            expanded_param_map[k] = v

    return expanded_param_map


def _get_warning_msg_for_fit_call_with_a_list_of_params(estimator):
    return (
        "Skip pyspark ML autologging when calling "
        + f"{_get_fully_qualified_class_name(estimator)}.fit with a list of params,"
        + "if you want to autolog for this case, you convert code to call `fit` with "
        + "each single param map."
    )


@experimental
@autologging_integration(AUTOLOGGING_INTEGRATION_NAME)
def autolog(
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging for pyspark ml estimators.
    This method is not threadsafe.
    This API requires Spark 3.0 or above.

    **When is autologging performed?**
      Autologging is performed when you call ``Estimator.fit`` except for estimators (featurizers)
      under ``pyspark.ml.feature``.

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
          (logged by :py:func:`mlflow.spark.log_model()`). Note that large models may not be
          autologged for performance and storage space considerations, and autologging for
          Pipelines and hyperparameter tuning meta-estimators (e.g. CrossValidator) is not yet
          supported.
          See ``log_models`` param below for details.


    :param log_models: If ``True``, if trained models are in allowlist, they are logged as MLflow
                       model artifacts. If ``False``, trained models are not logged.
                       Note: the built-in allowlist excludes some models (e.g. ALS models) which
                       can be large. To specify a custom allowlist, create a file containing a
                       newline-delimited list of fully-qualified estimator classnames, and set
                       the "spark.mlflow.pysparkml.autolog.logModelAllowlistFile" Spark config
                       to the path of your allowlist file.
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

    **The default log model allowlist in mlflow**
        .. literalinclude:: ../../../mlflow/pyspark/ml/log_model_allowlist.txt
           :language: text
    """
    from mlflow.utils.validation import (
        MAX_PARAMS_TAGS_PER_BATCH,
        MAX_PARAM_VAL_LENGTH,
        MAX_ENTITY_KEY_LENGTH,
    )
    from pyspark.ml.base import Estimator

    global _log_model_allowlist

    _log_model_allowlist = _read_log_model_allowlist()

    def _log_pretraining_metadata(estimator, params):

        if params and isinstance(params, dict):
            estimator = estimator.copy(params)

        # Chunk model parameters to avoid hitting the log_batch API limit
        for chunk in _chunk_dict(
            _get_instance_param_map(estimator), chunk_size=MAX_PARAMS_TAGS_PER_BATCH,
        ):
            truncated = _truncate_dict(chunk, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)
            try_mlflow_log(mlflow.log_params, truncated)

        try_mlflow_log(mlflow.set_tags, _get_estimator_info_tags(estimator))

    def _log_posttraining_metadata(estimator, spark_model, params):
        # TODO: Log nested runs for spark ml tuning estimators
        #   (CrossValidator/TrainValidationSplit)
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

        # Do not perform autologging on direct calls to fit() for featurizers.
        # Note that featurizers will be autologged when they're fit as part of a Pipeline.
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
        AUTOLOGGING_INTEGRATION_NAME, Estimator, "fit", patched_fit, manage_run=True,
    )

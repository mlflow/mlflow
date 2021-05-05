from collections import defaultdict, namedtuple
import logging
import numpy as np
import time
from pkg_resources import resource_filename

import mlflow
from mlflow.entities import Metric, Param
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _chunk_dict, _truncate_dict, _get_fully_qualified_class_name
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    _get_new_training_session_class,
    autologging_integration,
    safe_patch,
    try_mlflow_log,
)
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING, MLFLOW_PARENT_RUN_ID
from mlflow.utils.validation import (
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_ENTITY_KEY_LENGTH,
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
    from pyspark.ml.base import Model

    # TODO: Handle PipelineModel/CrossValidatorModel/TrainValidationSplitModel
    class_name = _get_fully_qualified_class_name(spark_model)
    if class_name in _log_model_allowlist:
        if class_name == "pyspark.ml.classification.OneVsRestModel":
            return _should_log_model(spark_model.models[0])
        elif class_name == "pyspark.ml.pipeline.PipelineModel":
            return all(
                _should_log_model(stage) for stage in spark_model.stages if isinstance(stage, Model)
            )
        elif _is_parameter_search_model(spark_model):
            return _should_log_model(spark_model.bestModel)
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


def _is_parameter_search_estimator(instance):
    from pyspark.ml.tuning import CrossValidator, TrainValidationSplit

    return isinstance(instance, (CrossValidator, TrainValidationSplit))


def _is_parameter_search_model(instance):
    from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel

    return isinstance(instance, (CrossValidatorModel, TrainValidationSplitModel))


def _should_log_hierarchy(estimator):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import OneVsRest

    return isinstance(estimator, (Pipeline, OneVsRest)) or _is_parameter_search_estimator(estimator)


AutologgingEstimatorMetadata = namedtuple(
    "AutologgingEstimatorMetadata",
    ["hierarchy", "uid_to_indexed_name_map", "param_search_estimators"],
)


def _traverse_stage(stage):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import OneVsRest

    yield stage
    if isinstance(stage, Pipeline):
        for stage in stage.getStages():
            yield from _traverse_stage(stage)
    elif isinstance(stage, OneVsRest):
        yield from _traverse_stage(stage.getClassifier())
    elif _is_parameter_search_estimator(stage):
        yield from _traverse_stage(stage.getEstimator())
        yield from _traverse_stage(stage.getEvaluator())


def _get_uid_to_indexed_name_map(estimator):
    counter = defaultdict(int)
    uid_to_classname_and_count = {}
    for child in _traverse_stage(estimator):
        class_name = child.__class__.__name__
        counter[class_name] += 1
        uid_to_classname_and_count[child.uid] = (class_name, counter[class_name])
    return {
        uid: f"{class_name}_{count}" if counter[class_name] > 1 else class_name
        for uid, (class_name, count) in uid_to_classname_and_count.items()
    }


def _gen_stage_hierarchy_recursively(
    stage, uid_to_indexed_name_map,
):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import OneVsRest

    stage_name = uid_to_indexed_name_map[stage.uid]

    if isinstance(stage, Pipeline):
        sub_stages = []
        for sub_stage in stage.getStages():
            sub_hierarchy = _gen_stage_hierarchy_recursively(sub_stage, uid_to_indexed_name_map)
            sub_stages.append(sub_hierarchy)
        return {"name": stage_name, "stages": sub_stages}
    elif isinstance(stage, OneVsRest):
        classifier_hierarchy = _gen_stage_hierarchy_recursively(
            stage.getClassifier(), uid_to_indexed_name_map
        )
        return {"name": stage_name, "classifier": classifier_hierarchy}
    elif _is_parameter_search_estimator(stage):
        evaluator = stage.getEvaluator()
        tuned_estimator = stage.getEstimator()
        return {
            "name": stage_name,
            "evaluator": _gen_stage_hierarchy_recursively(evaluator, uid_to_indexed_name_map),
            "tuned_estimator": _gen_stage_hierarchy_recursively(
                tuned_estimator, uid_to_indexed_name_map
            ),
        }
    else:
        return {"name": stage_name}


def _gen_estimator_metadata(estimator):
    """
    Returns an AutologgingEstimatorMetadata object.
    The AutologgingEstimatorMetadata object includes:
     - hierarchy: the hierarchy of the estimator, it will expand
         pipeline/meta estimator/param tuning estimator
     - uid_to_indexed_name_map: a map of `uid` -> `name`, each nested instance uid will be
         mapped to a fixed name. The naming rule is using `{class_name}` if the
         instance type occurs once, or `{class_name}_{index}` if the instance type occurs
         multiple times. The index order is in line with depth-first traversing.
     - param_search_estimators: a list includes all param search estimators in the
         hierarchy tree.
    """
    uid_to_indexed_name_map = _get_uid_to_indexed_name_map(estimator)
    param_search_estimators = [
        stage for stage in _traverse_stage(estimator) if _is_parameter_search_estimator(stage)
    ]
    hierarchy = _gen_stage_hierarchy_recursively(estimator, uid_to_indexed_name_map)

    metadata = AutologgingEstimatorMetadata(
        hierarchy=hierarchy,
        uid_to_indexed_name_map=uid_to_indexed_name_map,
        param_search_estimators=param_search_estimators,
    )
    estimator._autologging_metadata = metadata

    return metadata


def _get_instance_param_map_recursively(instance, level, uid_to_indexed_name_map):
    from pyspark.ml.param import Params
    from pyspark.ml.pipeline import Pipeline

    param_map = {
        param.name: instance.getOrDefault(param)
        for param in instance.params
        if instance.isDefined(param)
    }
    expanded_param_map = {}

    is_pipeline = isinstance(instance, Pipeline)
    is_parameter_search_estimator = _is_parameter_search_estimator(instance)

    if level == 0:
        logged_param_name_prefix = ""
    else:
        logged_param_name_prefix = uid_to_indexed_name_map[instance.uid] + "."

    for param_name, param_value in param_map.items():
        logged_param_name = logged_param_name_prefix + param_name

        if is_pipeline and param_name == "stages":
            expanded_param_map[logged_param_name] = [
                uid_to_indexed_name_map[stage.uid] for stage in instance.getStages()
            ]
            for stage in instance.getStages():
                stage_param_map = _get_instance_param_map_recursively(
                    stage, level + 1, uid_to_indexed_name_map
                )
                expanded_param_map.update(stage_param_map)
        elif is_parameter_search_estimator and param_name == "estimator":
            expanded_param_map[logged_param_name] = uid_to_indexed_name_map[param_value.uid]
            # skip log estimator's nested params because they will be logged as JSON artifact,
            # and they will be logged in nested runs as well.
        elif is_parameter_search_estimator and param_name == "estimatorParamMaps":
            # this param will be saved as JSON format artifact.
            pass
        elif isinstance(param_value, Params):
            # handle the case param value type inherits `pyspark.ml.param.Params`
            # e.g. param like
            # `OneVsRest.classifier`/`CrossValidator.evaluator`
            expanded_param_map[logged_param_name] = uid_to_indexed_name_map[param_value.uid]
            internal_param_map = _get_instance_param_map_recursively(
                param_value, level + 1, uid_to_indexed_name_map
            )
            expanded_param_map.update(internal_param_map)
        else:
            expanded_param_map[logged_param_name] = param_value

    return expanded_param_map


def _get_instance_param_map(instance, uid_to_indexed_name_map):
    return _get_instance_param_map_recursively(
        instance, level=0, uid_to_indexed_name_map=uid_to_indexed_name_map
    )


def _create_child_runs_for_parameter_search(parent_estimator, parent_model, parent_run, child_tags):
    from itertools import zip_longest

    client = MlflowClient()
    # Use the start time of the parent parameter search run as a rough estimate for the
    # start time of child runs, since we cannot precisely determine when each point
    # in the parameter search space was explored
    child_run_start_time = parent_run.info.start_time
    child_run_end_time = int(time.time() * 1000)

    estimator_param_maps = parent_estimator.getEstimatorParamMaps()
    tuned_estimator = parent_estimator.getEstimator()

    metric_key, metrics = _get_param_search_metrics(parent_estimator, parent_model)
    for i in range(len(estimator_param_maps)):
        child_estimator = tuned_estimator.copy(estimator_param_maps[i])
        tags_to_log = dict(child_tags) if child_tags else {}
        tags_to_log.update({MLFLOW_PARENT_RUN_ID: parent_run.info.run_id})
        tags_to_log.update(_get_estimator_info_tags(child_estimator))

        child_run = client.create_run(
            experiment_id=parent_run.info.experiment_id,
            start_time=child_run_start_time,
            tags=tags_to_log,
        )

        params_to_log = _get_instance_param_map(
            child_estimator, parent_estimator._autologging_metadata.uid_to_indexed_name_map
        )
        param_batches_to_log = _chunk_dict(params_to_log, chunk_size=MAX_PARAMS_TAGS_PER_BATCH)
        metrics_to_log = {metric_key: metrics[i]}
        for params_batch, metrics_batch in zip_longest(
            param_batches_to_log, [metrics_to_log], fillvalue={}
        ):
            # Trim any parameter keys / values and metric keys that exceed the limits
            # imposed by corresponding MLflow Tracking APIs (e.g., LogParam, LogMetric)
            truncated_params_batch = _truncate_dict(
                params_batch, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH
            )
            truncated_metrics_batch = _truncate_dict(
                metrics_batch, max_key_length=MAX_ENTITY_KEY_LENGTH
            )
            client.log_batch(
                run_id=child_run.info.run_id,
                params=[
                    Param(str(key), str(value)) for key, value in truncated_params_batch.items()
                ],
                metrics=[
                    Metric(key=str(key), value=value, timestamp=child_run_end_time, step=0)
                    for key, value in truncated_metrics_batch.items()
                ],
            )
        client.set_terminated(run_id=child_run.info.run_id, end_time=child_run_end_time)


def _log_parameter_search_results_as_artifact(param_maps, metrics, metric_name, run_id):
    import pandas as pd
    import json

    result_dict = defaultdict(list)
    result_dict["params"] = []
    result_dict[metric_name] = metrics
    for param_map in param_maps:
        result_dict["params"].append(json.dumps(param_map))
        for param_name, param_value in param_map.items():
            result_dict[f"param.{param_name}"].append(param_value)

    results_df = pd.DataFrame.from_dict(result_dict)
    with TempDir() as t:
        results_path = t.path("search_results.csv")
        results_df.to_csv(results_path, index=False)
        try_mlflow_log(MlflowClient().log_artifact, run_id, results_path)


def _get_warning_msg_for_fit_call_with_a_list_of_params(estimator):
    return (
        "Skip pyspark ML autologging when calling "
        + f"{_get_fully_qualified_class_name(estimator)}.fit with a list of params,"
        + "if you want to autolog for this case, you convert code to call `fit` with "
        + "each single param map."
    )


def _get_tuning_param_maps(param_search_estimator, uid_to_indexed_name_map):
    tuning_param_maps = []
    for eps in param_search_estimator.getEstimatorParamMaps():
        tuning_param_maps.append(
            {f"{uid_to_indexed_name_map[k.parent]}.{k.name}": v for k, v in eps.items()}
        )
    return tuning_param_maps


def _get_param_search_metrics(param_search_estimator, param_search_model):
    """
    Return a tuple of (metric_key, metrics: Array)
    """
    from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel

    metric_key = param_search_estimator.getEvaluator().getMetricName()
    if isinstance(param_search_model, CrossValidatorModel):
        metric_key = "avg_" + metric_key
        metrics = param_search_model.avgMetrics
    elif isinstance(param_search_model, TrainValidationSplitModel):
        metrics = param_search_model.validationMetrics
    else:
        raise RuntimeError(f"Unknown parameter search model type {type(param_search_model)}.")

    return metric_key, metrics


def _log_estimator_params(param_map):
    # Chunk model parameters to avoid hitting the log_batch API limit
    for chunk in _chunk_dict(param_map, chunk_size=MAX_PARAMS_TAGS_PER_BATCH,):
        truncated = _truncate_dict(chunk, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)
        try_mlflow_log(mlflow.log_params, truncated)


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

    **How does autologging work for meta estimators?**
          When a meta estimator (e.g. `Pipeline`_, `CrossValidator`_, `TrainValidationSplit`_,
          `OneVsRest`_)
          calls ``fit()``, it internally calls ``fit()`` on its child estimators. Autologging
          does NOT perform logging on these constituent ``fit()`` calls.

          A "estimator_info.json" artifact is logged, which includes a `hierarchy` entry
          describing the hierarchy of the meta estimator. The hierarchy includes expanded
          entries for all nested stages, such as nested pipeline stages.

      **Parameter search**
          In addition to recording the information discussed above, autologging for parameter
          search meta estimators (`CrossValidator`_ and `TrainValidationSplit`_) records child runs
          with metrics for each set of explored parameters, as well as artifacts and parameters
          for the best model and the best parameters (if available).
          For better readability, the "estimatorParamMaps" param in parameter search estimator
          will be recorded inside "estimator_info" artifact, see following description.
          Inside "estimator_info.json" artifact, in addition to the "hierarchy", records 2 more
          items: "tuning_parameter_map_list": a list contains all parameter maps used in tuning,
          and "tuned_estimator_parameter_map": the parameter map of the tuned estimator.
          Records a "best_parameters.json" artifacts, contains the best parameter it searched out.
          Records a "search_results.csv" artifacts, contains search results, it is a table with
          2 columns: "params" and "metric".

    .. _OneVsRest:
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.OneVsRest.html#pyspark.ml.classification.OneVsRest
    .. _Pipeline:
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html#pyspark.ml.Pipeline
    .. _CrossValidator:
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html#pyspark.ml.tuning.CrossValidator
    .. _TrainValidationSplit:
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.TrainValidationSplit.html#pyspark.ml.tuning.TrainValidationSplit

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
    from mlflow.tracking.context import registry as context_registry
    from pyspark.ml.base import Estimator

    global _log_model_allowlist

    _log_model_allowlist = _read_log_model_allowlist()

    def _log_pretraining_metadata(estimator, params):

        if params and isinstance(params, dict):
            estimator = estimator.copy(params)

        autologging_metadata = _gen_estimator_metadata(estimator)

        artifact_dict = {}

        param_map = _get_instance_param_map(estimator, autologging_metadata.uid_to_indexed_name_map)
        if _should_log_hierarchy(estimator):
            artifact_dict["hierarchy"] = autologging_metadata.hierarchy

        for param_search_estimator in autologging_metadata.param_search_estimators:
            param_search_estimator_name = (
                f"{autologging_metadata.uid_to_indexed_name_map[param_search_estimator.uid]}"
            )
            artifact_dict[param_search_estimator_name] = {}

            artifact_dict[param_search_estimator_name][
                "tuning_parameter_map_list"
            ] = _get_tuning_param_maps(
                param_search_estimator, autologging_metadata.uid_to_indexed_name_map
            )

            artifact_dict[param_search_estimator_name][
                "tuned_estimator_parameter_map"
            ] = _get_instance_param_map_recursively(
                param_search_estimator.getEstimator(),
                1,
                autologging_metadata.uid_to_indexed_name_map,
            )

        if artifact_dict:
            try_mlflow_log(mlflow.log_dict, artifact_dict, artifact_file="estimator_info.json")

        _log_estimator_params(param_map)

        try_mlflow_log(mlflow.set_tags, _get_estimator_info_tags(estimator))

    def _log_posttraining_metadata(estimator, spark_model, params):

        if _is_parameter_search_estimator(estimator):
            try:
                # Fetch environment-specific tags (e.g., user and source) to ensure that lineage
                # information is consistent with the parent run
                child_tags = context_registry.resolve_tags()
                child_tags.update({MLFLOW_AUTOLOGGING: AUTOLOGGING_INTEGRATION_NAME})
                _create_child_runs_for_parameter_search(
                    parent_estimator=estimator,
                    parent_model=spark_model,
                    parent_run=mlflow.active_run(),
                    child_tags=child_tags,
                )
            except Exception:
                import traceback

                msg = (
                    "Encountered exception during creation of child runs for parameter search."
                    " Child runs may be missing. Exception: {}".format(traceback.format_exc())
                )
                _logger.warning(msg)

            estimator_param_maps = _get_tuning_param_maps(
                estimator, estimator._autologging_metadata.uid_to_indexed_name_map
            )

            metric_key, metrics = _get_param_search_metrics(estimator, spark_model)
            _log_parameter_search_results_as_artifact(
                estimator_param_maps, metrics, metric_key, mlflow.active_run().info.run_id
            )

            if estimator.getEvaluator().isLargerBetter():
                best_index = np.argmax(metrics)
            else:
                best_index = np.argmin(metrics)

            # Log best_param_map as JSON artifact
            best_param_map = estimator_param_maps[best_index]
            try_mlflow_log(mlflow.log_dict, best_param_map, artifact_file="best_parameters.json")

            # Log best_param_map as autologging parameters as well
            _log_estimator_params(
                {
                    f"best_{param_name}": param_value
                    for param_name, param_value in best_param_map.items()
                }
            )

        if log_models:
            if _should_log_model(spark_model):
                # TODO: support model signature
                try_mlflow_log(
                    mlflow.spark.log_model, spark_model, artifact_path="model",
                )
                if _is_parameter_search_model(spark_model):
                    try_mlflow_log(
                        mlflow.spark.log_model, spark_model.bestModel, artifact_path="best_model",
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

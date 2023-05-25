from collections import defaultdict, namedtuple, OrderedDict
import logging
import numpy as np
import os
from urllib.parse import urlparse
import weakref
import sys
import traceback
import json
from itertools import zip_longest
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.tracking.client import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from mlflow.utils import (
    _chunk_dict,
    _truncate_dict,
    _get_fully_qualified_class_name,
    _inspect_original_var_name,
)
from mlflow.utils.autologging_utils import (
    _get_new_training_session_class,
    autologging_integration,
    safe_patch,
    resolve_input_example_and_signature,
)
from mlflow.utils.autologging_utils import get_method_call_arg_value
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
    MLFLOW_AUTOLOGGING,
    MLFLOW_DATASET_CONTEXT,
    MLFLOW_PARENT_RUN_ID,
)
from mlflow.utils.rest_utils import (
    augmented_raise_for_status,
    http_request,
    MlflowHostCreds,
)
from mlflow.utils.validation import (
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_ENTITY_KEY_LENGTH,
)
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
)
from mlflow.utils.time_utils import get_current_time_millis

_logger = logging.getLogger(__name__)
_SparkTrainingSession = _get_new_training_session_class()

AUTOLOGGING_INTEGRATION_NAME = "pyspark.ml"


def _read_log_model_allowlist_from_file(allowlist_file):
    def _parse_allowlist_file(line_iter):
        allowlist = set()
        for line in line_iter:
            stripped = line.strip()
            is_blankline_or_comment = stripped == "" or stripped.startswith("#")
            if not is_blankline_or_comment:
                allowlist.add(stripped)
        return allowlist

    url_parsed = urlparse(allowlist_file)
    scheme = url_parsed.scheme
    path = url_parsed.path
    if os.name == "nt" and not url_parsed.hostname:
        path = scheme + "://" + path
        scheme = ""
    if scheme in ("file", ""):
        if not os.path.exists(path):
            raise MlflowException.invalid_parameter_value(f"{allowlist_file} does not exist")

        with open(allowlist_file) as f:
            return _parse_allowlist_file(f)
    else:
        host_creds = MlflowHostCreds(
            host=scheme + "://" + (url_parsed.hostname or ""),
            username=url_parsed.username,
            password=url_parsed.password,
        )
        response = http_request(host_creds=host_creds, endpoint=path, method="GET")
        augmented_raise_for_status(response)
        return _parse_allowlist_file(response.iter_lines(decode_unicode=True))


def _read_log_model_allowlist():
    """
    Reads the module allowlist and returns it as a set.
    """
    from mlflow.utils._spark_utils import _get_active_spark_session

    # New in 3.9: https://docs.python.org/3/library/importlib.resources.html#importlib.resources.files
    if sys.version_info.major > 2 and sys.version_info.minor > 8:
        from importlib.resources import as_file, files  # pylint: disable=lazy-builtin-import

        with as_file(files(__name__).joinpath("log_model_allowlist.txt")) as file:
            builtin_allowlist_file = file.as_posix()
    else:
        from importlib.resources import path  # pylint: disable=lazy-builtin-import

        with path(__name__, "log_model_allowlist.txt") as file:
            builtin_allowlist_file = file.as_posix()
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
                (
                    "Reading from custom log_models allowlist file %s failed, "
                    "fallback to built-in allowlist file."
                ),
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
    should_log = class_name in _log_model_allowlist
    if not should_log:
        for name in _log_model_allowlist:
            # only support one trailing *
            if name.endswith("*") and class_name.startswith(name[:-1]):
                should_log = True
                break
    if should_log:
        if class_name == "pyspark.ml.classification.OneVsRestModel":
            return _should_log_model(spark_model.models[0])
        elif class_name == "pyspark.ml.pipeline.PipelineModel":
            return all(
                _should_log_model(stage) for stage in spark_model.stages if isinstance(stage, Model)
            )
        elif _is_parameter_search_model(spark_model):
            return _should_log_model(spark_model.bestModel)
        else:
            return all(
                _should_log_model(param_value)
                for _, param_value in _get_param_map(spark_model).items()
                # Transformers are logged by default as the same behavior as PipelineModel
                if isinstance(param_value, Model)
            )
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

    return (
        isinstance(estimator, (Pipeline, OneVsRest))
        or _is_parameter_search_estimator(estimator)
        or any(_get_stage_type_params(estimator))
    )


_AutologgingEstimatorMetadata = namedtuple(
    "_AutologgingEstimatorMetadata",
    ["hierarchy", "uid_to_indexed_name_map", "param_search_estimators"],
)


def _traverse_stage(stage):
    from pyspark.ml import Pipeline

    yield stage
    if isinstance(stage, Pipeline):
        original_sub_stages = stage.getStages()

        try:
            iter(original_sub_stages)
        except TypeError:
            raise TypeError(
                f"Pipeline stages should be iterable, but found object {original_sub_stages}"
            )
        for stage in original_sub_stages:
            yield from _traverse_stage(stage)
    else:
        # General support for params that of type Params
        for _, param_value in _get_stage_type_params(stage).items():
            yield from _traverse_stage(param_value)


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


def _gen_stage_hierarchy_recursively(stage, uid_to_indexed_name_map):
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
    elif any(_get_stage_type_params(stage)):
        sub_params = {}
        for param_name, param_value in _get_stage_type_params(stage).items():
            sub_hierarchy = _gen_stage_hierarchy_recursively(param_value, uid_to_indexed_name_map)
            sub_params[param_name] = sub_hierarchy
        return {"name": stage_name, "params": sub_params}
    else:
        return {"name": stage_name}


def _gen_estimator_metadata(estimator):
    """
    Returns an `_AutologgingEstimatorMetadata` object.
    The `_AutologgingEstimatorMetadata` object includes:
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

    metadata = _AutologgingEstimatorMetadata(
        hierarchy=hierarchy,
        uid_to_indexed_name_map=uid_to_indexed_name_map,
        param_search_estimators=param_search_estimators,
    )
    estimator._autologging_metadata = metadata

    return metadata


def _get_param_map(instance):
    return {
        param.name: instance.getOrDefault(param)
        for param in instance.params
        if instance.isDefined(param)
    }


def _get_stage_type_params(instance):
    """
    Get the param map of the instance where param value is of type pyspark.ml.param.Params
    """
    from pyspark.ml.param import Params

    return {
        param.name: instance.getOrDefault(param)
        for param in instance.params
        if instance.isDefined(param) and isinstance(instance.getOrDefault(param), Params)
    }


def _get_instance_param_map_recursively(instance, level, uid_to_indexed_name_map):
    from pyspark.ml.param import Params
    from pyspark.ml.pipeline import Pipeline

    param_map = _get_param_map(instance)
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
    client = MlflowClient()
    # Use the start time of the parent parameter search run as a rough estimate for the
    # start time of child runs, since we cannot precisely determine when each point
    # in the parameter search space was explored
    child_run_start_time = parent_run.info.start_time
    child_run_end_time = get_current_time_millis()

    estimator_param_maps = parent_estimator.getEstimatorParamMaps()
    tuned_estimator = parent_estimator.getEstimator()

    metrics_dict, _ = _get_param_search_metrics_and_best_index(parent_estimator, parent_model)
    for i, est_param in enumerate(estimator_param_maps):
        child_estimator = tuned_estimator.copy(est_param)
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
        metrics_to_log = {k: v[i] for k, v in metrics_dict.items()}
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


def _log_parameter_search_results_as_artifact(param_maps, metrics_dict, run_id):
    import pandas as pd

    result_dict = defaultdict(list)
    result_dict["params"] = []
    result_dict.update(metrics_dict)
    for param_map in param_maps:
        result_dict["params"].append(json.dumps(param_map))
        for param_name, param_value in param_map.items():
            result_dict[f"param.{param_name}"].append(param_value)

    results_df = pd.DataFrame.from_dict(result_dict)
    with TempDir() as t:
        results_path = t.path("search_results.csv")
        results_df.to_csv(results_path, index=False)
        MlflowClient().log_artifact(run_id, results_path)


def _get_warning_msg_for_fit_call_with_a_list_of_params(estimator):
    return (
        "Skip pyspark ML autologging when calling "
        + f"{_get_fully_qualified_class_name(estimator)}.fit with a list of params,"
        + "if you want to autolog for this case, you convert code to call `fit` with "
        + "each single param map."
    )


def _get_tuning_param_maps(param_search_estimator, uid_to_indexed_name_map):
    tuning_param_maps = []

    def gen_log_key(param):
        if param.parent not in uid_to_indexed_name_map:
            raise ValueError(
                "Tuning params should not include params not owned by the tuned estimator, but "
                f"found a param {param}"
            )
        return f"{uid_to_indexed_name_map[param.parent]}.{param.name}"

    for eps in param_search_estimator.getEstimatorParamMaps():
        tuning_param_maps.append({gen_log_key(k): v for k, v in eps.items()})
    return tuning_param_maps


def _get_param_search_metrics_and_best_index(param_search_estimator, param_search_model):
    """
    Return a tuple of `(metrics_dict, best_index)`
    `metrics_dict` is a dict of metric_name --> metric_values for each param map
    - For CrossValidatorModel, the result dict contains metrics of avg_metris and std_metrics
      for each param map.
    - For TrainValidationSplitModel, the result dict contains metrics for each param map.

    `best_index` is the best index of trials.
    """
    from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel

    metrics_dict = {}

    metric_key = param_search_estimator.getEvaluator().getMetricName()
    if isinstance(param_search_model, CrossValidatorModel):
        avg_metrics = param_search_model.avgMetrics
        metrics_dict["avg_" + metric_key] = avg_metrics
        if hasattr(param_search_model, "stdMetrics"):
            metrics_dict["std_" + metric_key] = param_search_model.stdMetrics
    elif isinstance(param_search_model, TrainValidationSplitModel):
        avg_metrics = param_search_model.validationMetrics
        metrics_dict[metric_key] = avg_metrics
    else:
        raise RuntimeError(f"Unknown parameter search model type {type(param_search_model)}.")

    if param_search_estimator.getEvaluator().isLargerBetter():
        best_index = np.argmax(avg_metrics)
    else:
        best_index = np.argmin(avg_metrics)

    return metrics_dict, best_index


def _log_estimator_params(param_map):
    # Chunk model parameters to avoid hitting the log_batch API limit
    for chunk in _chunk_dict(param_map, chunk_size=MAX_PARAMS_TAGS_PER_BATCH):
        truncated = _truncate_dict(chunk, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)
        mlflow.log_params(truncated)


class _AutologgingMetricsManager:
    """
    This class is designed for holding information which is used by autologging metrics
    It will hold information of:
    (1) a map of "prediction result object id" to a tuple of dataset name(the dataset is
       the one which generate the prediction result) and run_id.
       Note: We need this map instead of setting the run_id into the "prediction result object"
       because the object maybe a numpy array which does not support additional attribute
       assignment.
    (2) _log_post_training_metrics_enabled flag, in the following method scope:
       `Estimator.fit`, `Model.transform`, `Evaluator.evaluate`,
       in order to avoid nested/duplicated autologging metric, when run into these scopes,
       we need temporarily disable the metric autologging.
    (3) _eval_dataset_info_map, it is a double level map:
       `_eval_dataset_info_map[run_id][eval_dataset_var_name]` will get a list, each
       element in the list is an id of "eval_dataset" instance.
       This data structure is used for:
        * generating unique dataset name key when autologging metric. For each eval dataset object,
          if they have the same eval_dataset_var_name, but object ids are different,
          then they will be assigned different name (via appending index to the
          eval_dataset_var_name) when autologging.
    (4) _evaluator_call_info, it is a double level map:
       `_metric_api_call_info[run_id][metric_name]` wil get a list of tuples, each tuple is:
         (logged_metric_key, evaluator_information)
        Evaluator information includes evaluator class name and params, these information
        will also be logged into "metric_info.json" artifacts.

    Note: this class is not thread-safe.
    Design rule for this class:
     Because this class instance is a global instance, in order to prevent memory leak, it should
     only holds IDs and other small objects references. This class internal data structure should
     avoid reference to user dataset variables or model variables.
    """

    def __init__(self):
        self._pred_result_id_to_dataset_name_and_run_id = {}
        self._eval_dataset_info_map = defaultdict(lambda: defaultdict(list))
        self._evaluator_call_info = defaultdict(lambda: defaultdict(list))
        self._log_post_training_metrics_enabled = True
        self._metric_info_artifact_need_update = defaultdict(lambda: False)

    def should_log_post_training_metrics(self):
        """
        Check whether we should run patching code for autologging post training metrics.
        This checking should surround the whole patched code due to the safe guard checking,
        See following note.

        Note: It includes checking `_SparkTrainingSession.is_active()`, This is a safe guarding
        for meta-estimator (e.g. CrossValidator/TrainValidationSplit) case:
          running CrossValidator.fit, the nested `estimator.fit` will be called in parallel,
          but, the _autolog_training_status is a global status without thread-safe lock protecting.
          This safe guarding will prevent code run into this case.
        """
        return not _SparkTrainingSession.is_active() and self._log_post_training_metrics_enabled

    def disable_log_post_training_metrics(self):
        class LogPostTrainingMetricsDisabledScope:
            def __enter__(inner_self):  # pylint: disable=no-self-argument
                # pylint: disable=attribute-defined-outside-init
                inner_self.old_status = self._log_post_training_metrics_enabled
                self._log_post_training_metrics_enabled = False

            # pylint: disable=no-self-argument
            def __exit__(inner_self, exc_type, exc_val, exc_tb):
                self._log_post_training_metrics_enabled = inner_self.old_status

        return LogPostTrainingMetricsDisabledScope()

    @staticmethod
    def get_run_id_for_model(model):
        return getattr(model, "_mlflow_run_id", None)

    @staticmethod
    def is_metric_value_loggable(metric_value):
        """
        check whether the specified `metric_value` is a numeric value which can be logged
        as an MLflow metric.
        """
        return isinstance(metric_value, (int, float, np.number)) and not isinstance(
            metric_value, bool
        )

    def register_model(self, model, run_id):
        """
        In `patched_fit`, we need register the model with the run_id used in `patched_fit`
        So that in following metric autologging, the metric will be logged into the registered
        run_id
        """
        model._mlflow_run_id = run_id

    @staticmethod
    def gen_name_with_index(name, index):
        assert index >= 0
        if index == 0:
            return name
        else:
            # Use '-' as the separator between name and index,
            # The '-' is not valid character in python var name
            # so it can prevent name conflicts after appending index.
            return f"{name}-{index + 1}"

    def register_prediction_input_dataset(self, model, eval_dataset):
        """
        Register prediction input dataset into eval_dataset_info_map, it will do:
         1. inspect eval dataset var name.
         2. check whether eval_dataset_info_map already registered this eval dataset.
            will check by object id.
         3. register eval dataset with id.
         4. return eval dataset name with index.

        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.
        """
        eval_dataset_name = _inspect_original_var_name(
            eval_dataset, fallback_name="unknown_dataset"
        )
        eval_dataset_id = id(eval_dataset)

        run_id = self.get_run_id_for_model(model)
        registered_dataset_list = self._eval_dataset_info_map[run_id][eval_dataset_name]

        for i, id_i in enumerate(registered_dataset_list):
            if eval_dataset_id == id_i:
                index = i
                break
        else:
            index = len(registered_dataset_list)

        if index == len(registered_dataset_list):
            # register new eval dataset
            registered_dataset_list.append(eval_dataset_id)

        return self.gen_name_with_index(eval_dataset_name, index)

    def register_prediction_result(self, run_id, eval_dataset_name, predict_result):
        """
        Register the relationship
         id(prediction_result) --> (eval_dataset_name, run_id)
        into map `_pred_result_id_to_dataset_name_and_run_id`
        """
        value = (eval_dataset_name, run_id)
        prediction_result_id = id(predict_result)
        self._pred_result_id_to_dataset_name_and_run_id[prediction_result_id] = value

        def clean_id(id_):
            _AUTOLOGGING_METRICS_MANAGER._pred_result_id_to_dataset_name_and_run_id.pop(id_, None)

        # When the `predict_result` object being GCed, its ID may be reused, so register a finalizer
        # to clear the ID from the dict for preventing wrong ID mapping.
        weakref.finalize(predict_result, clean_id, prediction_result_id)

    def get_run_id_and_dataset_name_for_evaluator_call(self, pred_result_dataset):
        """
        Given a registered prediction result dataset object,
        return a tuple of (run_id, eval_dataset_name)
        """
        if id(pred_result_dataset) in self._pred_result_id_to_dataset_name_and_run_id:
            dataset_name, run_id = self._pred_result_id_to_dataset_name_and_run_id[
                id(pred_result_dataset)
            ]
            return run_id, dataset_name
        else:
            return None, None

    def gen_evaluator_info(self, evaluator):
        """
        Generate evaluator information, include evaluator class name and params.
        """
        class_name = _get_fully_qualified_class_name(evaluator)
        param_map = _truncate_dict(
            _get_param_map(evaluator), MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH
        )
        return {"evaluator_class": class_name, "params": param_map}

    def register_evaluator_call(self, run_id, metric_name, dataset_name, evaluator_info):
        """
        Register the `Evaluator.evaluate` call, including register the evaluator information
        (See doc of `gen_evaluator_info` method) into the corresponding run_id and metric_name
        entry in the registry table.
        """
        evaluator_call_info_list = self._evaluator_call_info[run_id][metric_name]

        index = len(evaluator_call_info_list)
        metric_name_with_index = self.gen_name_with_index(metric_name, index)
        metric_key = f"{metric_name_with_index}_{dataset_name}"

        evaluator_call_info_list.append((metric_key, evaluator_info))

        # Set the flag to true, represent the metric info in this run need update.
        # Later when `log_eval_metric` called, it will generate a new metric_info artifact
        # and overwrite the old artifact.
        self._metric_info_artifact_need_update[run_id] = True
        return metric_key

    def log_post_training_metric(self, run_id, key, value):
        """
        Log the metric into the specified mlflow run.
        and it will also update the metric_info artifact if needed.
        """
        # Note: if the case log the same metric key multiple times,
        #  newer value will overwrite old value
        client = MlflowClient()
        client.log_metric(run_id=run_id, key=key, value=value)
        if self._metric_info_artifact_need_update[run_id]:
            evaluator_call_list = []
            for v in self._evaluator_call_info[run_id].values():
                evaluator_call_list.extend(v)

            evaluator_call_list.sort(key=lambda x: x[0])
            dict_to_log = OrderedDict(evaluator_call_list)
            client.log_dict(run_id=run_id, dictionary=dict_to_log, artifact_file="metric_info.json")
            self._metric_info_artifact_need_update[run_id] = False


# The global `_AutologgingMetricsManager` instance which holds information used in
# post-training metric autologging. See doc of class `_AutologgingMetricsManager` for details.
_AUTOLOGGING_METRICS_MANAGER = _AutologgingMetricsManager()


def _get_columns_with_unsupported_data_type(df):
    from mlflow.types.schema import DataType

    supported_spark_types = DataType.get_spark_types()
    unsupported_columns = []
    for field in df.schema.fields:
        if field.dataType not in supported_spark_types:
            unsupported_columns.append(field)
    return unsupported_columns


@autologging_integration(AUTOLOGGING_INTEGRATION_NAME)
def autolog(
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_post_training_metrics=True,
    registered_model_name=None,
    log_input_examples=False,
    log_model_signatures=True,
    log_model_allowlist=None,
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

      .. _post training metrics:

      **Post training metrics**
        When users call evaluator APIs after model training, MLflow tries to capture the
        `Evaluator.evaluate` results and log them as MLflow metrics to the Run associated with
        the model. All pyspark ML evaluators are supported.

        For post training metrics autologging, the metric key format is:
        "{metric_name}[-{call_index}]_{dataset_name}"

        - The metric name is the name returned by `Evaluator.getMetricName()`
        - If multiple calls are made to the same pyspark ML evaluator metric, each subsequent call
          adds a "call_index" (starting from 2) to the metric key.
        - MLflow uses the prediction input dataset variable name as the "dataset_name" in the
          metric key. The "prediction input dataset variable" refers to the variable which was
          used as the `dataset` argument of `model.transform` call.
          Note: MLflow captures the "prediction input dataset" instance in the outermost call
          frame and fetches the variable name in the outermost call frame. If the "prediction
          input dataset" instance is an intermediate expression without a defined variable
          name, the dataset name is set to "unknown_dataset". If multiple "prediction input
          dataset" instances have the same variable name, then subsequent ones will append an
          index (starting from 2) to the inspected dataset name.

        **Limitations**
          - MLflow cannot find run information for other objects derived from a given prediction
            result (e.g. by doing some transformation on the prediction result dataset).

      **Artifacts**
        - An MLflow Model with the :py:mod:`mlflow.spark` flavor containing a fitted estimator
          (logged by :py:func:`mlflow.spark.log_model()`). Note that large models may not be
          autologged for performance and storage space considerations, and autologging for
          Pipelines and hyperparameter tuning meta-estimators (e.g. CrossValidator) is not yet
          supported.
          See ``log_models`` param below for details.
        - For post training metrics API calls, a "metric_info.json" artifact is logged. This is a
          JSON object whose keys are MLflow post training metric names
          (see "Post training metrics" section for the key format) and whose values are the
          corresponding evaluator information, including evaluator class name and evaluator params.

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
    :param log_datasets: If ``True``, dataset information is logged to MLflow Tracking.
                         If ``False``, dataset information is not logged.
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
    :param log_post_training_metrics: If ``True``, post training metrics are logged. Defaults to
                                      ``True``. See the `post training metrics`_ section for more
                                      details.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.
    :param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with pyspark ml model artifacts during training. If
                               ``False``, input examples are not logged.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with spark ml pipeline/estimator artifacts during training.
                                 If ``False`` signatures are not logged.

                                 .. warning::

                                    Currently, only scalar Spark data types are supported. If
                                    model inputs/outputs contain non-scalar Spark data types such
                                    as ``pyspark.ml.linalg.Vector``, signatures are not logged.
    :param log_model_allowlist: If given, it overrides the default log model allowlist in mlflow.
                                This takes precedence over the spark configuration of
                                "spark.mlflow.pysparkml.autolog.logModelAllowlistFile".

    **The default log model allowlist in mlflow**
        .. literalinclude:: ../../../mlflow/pyspark/ml/log_model_allowlist.txt
           :language: text
    """
    from mlflow.tracking.context import registry as context_registry
    from pyspark.ml.base import Estimator, Model
    from pyspark.ml.evaluation import Evaluator

    global _log_model_allowlist

    if log_model_allowlist:
        _log_model_allowlist = set(model.strip() for model in log_model_allowlist)
    else:
        _log_model_allowlist = _read_log_model_allowlist()

    def _log_pretraining_metadata(estimator, params, input_df):
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
            mlflow.log_dict(artifact_dict, artifact_file="estimator_info.json")

        _log_estimator_params(param_map)

        mlflow.set_tags(_get_estimator_info_tags(estimator))

        if log_datasets:
            try:
                context_tags = context_registry.resolve_tags()
                code_source = CodeDatasetSource(context_tags)
                dataset = SparkDataset(
                    df=input_df,
                    source=code_source,
                )
                mlflow.log_input(dataset, "train")
            except Exception as e:
                _logger.warning(
                    "Failed to log training dataset information to MLflow Tracking. Reason: %s", e
                )

    def _log_posttraining_metadata(estimator, spark_model, params, input_df):
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
                msg = (
                    "Encountered exception during creation of child runs for parameter search."
                    " Child runs may be missing. Exception: {}".format(traceback.format_exc())
                )
                _logger.warning(msg)

            estimator_param_maps = _get_tuning_param_maps(
                estimator, estimator._autologging_metadata.uid_to_indexed_name_map
            )

            metrics_dict, best_index = _get_param_search_metrics_and_best_index(
                estimator, spark_model
            )
            _log_parameter_search_results_as_artifact(
                estimator_param_maps, metrics_dict, mlflow.active_run().info.run_id
            )

            # Log best_param_map as JSON artifact
            best_param_map = estimator_param_maps[best_index]
            mlflow.log_dict(best_param_map, artifact_file="best_parameters.json")

            # Log best_param_map as autologging parameters as well
            _log_estimator_params(
                {
                    f"best_{param_name}": param_value
                    for param_name, param_value in best_param_map.items()
                }
            )

        if log_models:
            if _should_log_model(spark_model):
                from mlflow.models import infer_signature
                from mlflow.pyspark.ml._autolog import (
                    cast_spark_df_with_vector_to_array,
                    get_feature_cols,
                )
                from mlflow.spark import _find_and_set_features_col_as_vector_if_needed
                from pyspark.sql import SparkSession

                spark = SparkSession.builder.getOrCreate()

                def _get_input_example_as_pd_df():
                    feature_cols = list(get_feature_cols(input_df, spark_model))
                    limited_input_df = input_df.select(feature_cols).limit(
                        INPUT_EXAMPLE_SAMPLE_ROWS
                    )
                    return cast_spark_df_with_vector_to_array(limited_input_df).toPandas()

                def _infer_model_signature(input_example_slice):
                    input_slice_df = _find_and_set_features_col_as_vector_if_needed(
                        spark.createDataFrame(input_example_slice), spark_model
                    )
                    model_output = spark_model.transform(input_slice_df).drop(
                        *input_slice_df.columns
                    )
                    # TODO: Remove this once we support non-scalar spark data types
                    unsupported_columns = _get_columns_with_unsupported_data_type(model_output)
                    if unsupported_columns:
                        _logger.warning(
                            "Model outputs contain unsupported Spark data types: "
                            f"{unsupported_columns}. Output schema is not be logged."
                        )
                        model_output = None
                    else:
                        model_output = model_output.toPandas()

                    return infer_signature(input_example_slice, model_output)

                # TODO: Remove this once we support non-scalar spark data types
                nonlocal log_model_signatures
                if log_model_signatures:
                    unsupported_columns = _get_columns_with_unsupported_data_type(input_df)
                    if unsupported_columns:
                        _logger.warning(
                            "Model inputs contain unsupported Spark data types: "
                            f"{unsupported_columns}. Model signature is not logged."
                        )
                        log_model_signatures = False

                input_example, signature = resolve_input_example_and_signature(
                    _get_input_example_as_pd_df,
                    _infer_model_signature,
                    log_input_examples,
                    log_model_signatures,
                    _logger,
                )

                mlflow.spark.log_model(
                    spark_model,
                    artifact_path="model",
                    registered_model_name=registered_model_name,
                    input_example=input_example,
                    signature=signature,
                )
                if _is_parameter_search_model(spark_model):
                    mlflow.spark.log_model(
                        spark_model.bestModel,
                        artifact_path="best_model",
                    )
            else:
                _logger.warning(_get_warning_msg_for_skip_log_model(spark_model))

    def fit_mlflow(original, self, *args, **kwargs):
        params = get_method_call_arg_value(1, "params", None, args, kwargs)

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
            # we need generate estimator param map so we call `self.copy(params)` to construct
            # an estimator with the extra params.
            from pyspark.storagelevel import StorageLevel

            estimator = self.copy(params) if params is not None else self
            input_training_df = args[0].persist(StorageLevel.MEMORY_AND_DISK)
            _log_pretraining_metadata(estimator, params, input_training_df)
            spark_model = original(self, *args, **kwargs)
            _log_posttraining_metadata(estimator, spark_model, params, input_training_df)
            input_training_df.unpersist()

            return spark_model

    def patched_fit(original, self, *args, **kwargs):
        should_log_post_training_metrics = (
            log_post_training_metrics
            and _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics()
        )
        with _SparkTrainingSession(estimator=self, allow_children=False) as t:
            if t.should_log():
                with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                    fit_result = fit_mlflow(original, self, *args, **kwargs)
                # In some cases the `fit_result` may be an iterator of spark models.
                if should_log_post_training_metrics and isinstance(fit_result, Model):
                    _AUTOLOGGING_METRICS_MANAGER.register_model(
                        fit_result, mlflow.active_run().info.run_id
                    )
                return fit_result
            else:
                return original(self, *args, **kwargs)

    def patched_transform(original, self, *args, **kwargs):
        run_id = _AUTOLOGGING_METRICS_MANAGER.get_run_id_for_model(self)
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics() and run_id:
            predict_result = original(self, *args, **kwargs)
            eval_dataset = get_method_call_arg_value(0, "dataset", None, args, kwargs)
            eval_dataset_name = _AUTOLOGGING_METRICS_MANAGER.register_prediction_input_dataset(
                self, eval_dataset
            )
            _AUTOLOGGING_METRICS_MANAGER.register_prediction_result(
                run_id, eval_dataset_name, predict_result
            )
            return predict_result
        else:
            return original(self, *args, **kwargs)

    def patched_evaluate(original, self, *args, **kwargs):
        if _AUTOLOGGING_METRICS_MANAGER.should_log_post_training_metrics():
            with _AUTOLOGGING_METRICS_MANAGER.disable_log_post_training_metrics():
                metric = original(self, *args, **kwargs)

            if _AUTOLOGGING_METRICS_MANAGER.is_metric_value_loggable(metric):
                params = get_method_call_arg_value(1, "params", None, args, kwargs)
                # we need generate evaluator param map so we call `self.copy(params)` to construct
                # an evaluator with the extra evaluation params.
                evaluator = self.copy(params) if params is not None else self
                metric_name = evaluator.getMetricName()
                evaluator_info = _AUTOLOGGING_METRICS_MANAGER.gen_evaluator_info(evaluator)

                pred_result_dataset = get_method_call_arg_value(0, "dataset", None, args, kwargs)
                (
                    run_id,
                    dataset_name,
                ) = _AUTOLOGGING_METRICS_MANAGER.get_run_id_and_dataset_name_for_evaluator_call(
                    pred_result_dataset
                )
                if run_id and dataset_name:
                    metric_key = _AUTOLOGGING_METRICS_MANAGER.register_evaluator_call(
                        run_id, metric_name, dataset_name, evaluator_info
                    )
                    _AUTOLOGGING_METRICS_MANAGER.log_post_training_metric(
                        run_id, metric_key, metric
                    )
                    if log_datasets:
                        try:
                            context_tags = context_registry.resolve_tags()
                            code_source = CodeDatasetSource(context_tags)

                            dataset = SparkDataset(
                                df=pred_result_dataset,
                                source=code_source,
                            )
                            tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value="eval")]
                            dataset_input = DatasetInput(
                                dataset=dataset._to_mlflow_entity(), tags=tags
                            )
                            client = MlflowClient()
                            client.log_inputs(run_id, [dataset_input])
                        except Exception as e:
                            _logger.warning(
                                "Failed to log evaluation dataset information to MLflow Tracking. "
                                "Reason: %s",
                                e,
                            )
            return metric
        else:
            return original(self, *args, **kwargs)

    safe_patch(
        AUTOLOGGING_INTEGRATION_NAME,
        Estimator,
        "fit",
        patched_fit,
        manage_run=True,
    )

    if log_post_training_metrics:
        safe_patch(
            AUTOLOGGING_INTEGRATION_NAME,
            Model,
            "transform",
            patched_transform,
            manage_run=False,
        )
        safe_patch(
            AUTOLOGGING_INTEGRATION_NAME,
            Evaluator,
            "evaluate",
            patched_evaluate,
            manage_run=False,
        )
